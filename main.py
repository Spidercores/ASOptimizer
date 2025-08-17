import tensorflow as tf
import pandas as pd
import numpy as np
import warnings, os
import shutil
import tqdm
import time
import random
import torch
from scipy import stats
from absl import flags, app
from libml import models, utils
from libml.data import DATASETS
from sklearn import metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NCCL_P2P_DISABLE'] = '1'

seed=42
tf.random.set_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

class ASOptimizer(models.EGT_Model):
    def train(self, dataset, node_dim, edge_dim, model_height, num_head, num_vnode,max_length):
        self.dataset = dataset

        NUM_GPU = len(utils.get_available_gpus())
        print(f"Num of GPUs: {NUM_GPU}")

        mirrored_strategy = tf.distribute.MirroredStrategy()
        BATCH_SIZE_PER_REPLICA = int(FLAGS.batch/NUM_GPU)
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

        print(GLOBAL_BATCH_SIZE)

        with mirrored_strategy.scope():

            train_data = self.dataset.train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            train_dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_data)

            test_data = self.dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            test_dist_dataset = mirrored_strategy.experimental_distribute_dataset(test_data)

            screen_data = self.dataset.screen.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            screen_dist_dataset = mirrored_strategy.experimental_distribute_dataset(screen_data)


        checkpoint_dir = './checkpoints/training_checkpoints_1209'

        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "best_checkpoint")
        checkpoint_prefix_roche = os.path.join(checkpoint_dir, "best_checkpoint_roche")

        best_acc, best_epoch, best_corr = 0, 0, -1

        with mirrored_strategy.scope():
            model = self.EGT_Backbone(node_dim,edge_dim,model_height, num_head, num_vnode,max_length)
            model.summary()

            margin = 0.05
            opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)


        @tf.function
        def train_step(front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges):
            with tf.GradientTape() as tape:
                front_out, _ = model([front_nodes, front_adjs, front_edges], training=True)
                back_out, _ = model([back_nodes, back_adjs, back_edges], training=True)

                preds = back_out - front_out
                results = tf.cast(preds < 0, dtype=tf.int32)

                acc = tf.reduce_sum(tf.cast(tf.equal(results, 1), dtype=tf.float32)) / GLOBAL_BATCH_SIZE

                loss = tf.reduce_sum(tf.maximum(margin + back_out - front_out, 0)) / GLOBAL_BATCH_SIZE


            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

            return loss, acc

        def eval_step(front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges):
            front_out, _ = model([front_nodes, front_adjs, front_edges])
            back_out, _ = model([back_nodes, back_adjs, back_edges])


            preds = back_out - front_out
            results = tf.cast(preds < 0, dtype=tf.int32)

            acc = tf.reduce_sum(tf.cast(tf.equal(results, 1), dtype=tf.float32)) / GLOBAL_BATCH_SIZE

            loss = tf.reduce_sum(tf.maximum(margin + back_out - front_out, 0))  / GLOBAL_BATCH_SIZE

            return loss, acc

        def screen_step(merge_nodes, merge_adjs, merge_edges, front_ids, gts):
            merge_out, _ = model([merge_nodes, merge_adjs, merge_edges])

            return merge_out, front_ids, gts

        with mirrored_strategy.scope():
            # @tf.function
            def distributed_train_step(front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges):
                per_replica_losses, per_replica_accs = mirrored_strategy.run(train_step,
                                                                             args=(
                                                                             front_nodes, front_adjs, front_edges,
                                                                             back_nodes, back_adjs, back_edges))
                return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                                axis=None), mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                                     per_replica_accs,
                                                                                     axis=None)

            # @tf.function
            def distributed_eval_step(front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges):
                per_replica_losses, per_replica_accs = mirrored_strategy.run(eval_step, args=(
                front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges))
                return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                                axis=None), mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                                     per_replica_accs,
                                                                                     axis=None)
            
            @tf.function
            def distributed_screen_step(front_nodes, front_adjs, front_edges, front_ids, gts):
                per_replica_preds, per_replica_ids, per_replica_gts = mirrored_strategy.run(screen_step, args=(
                front_nodes, front_adjs, front_edges, front_ids, gts))
                return mirrored_strategy.gather(per_replica_preds, axis=0), mirrored_strategy.gather(per_replica_ids, axis=0), mirrored_strategy.gather(per_replica_gts, axis=0)

                                                                                     
            df_save = pd.DataFrame([],columns=['epoch','tr_loss','tr_acc','test_loss','test_acc','roche-pear'])
            with mirrored_strategy.scope():
                for epoch in range(FLAGS.epochs):
                    losses = []
                    accs = []
                    losses_val = []
                    accs_val = []

                    preds = []
                    local_ids = []
                    inhibitions = []

                    for element in train_dist_dataset:

                        data = element[0]

                        loss, acc = distributed_train_step(
                            data['front_feat'], data['front_adj'], data['front_e_feat'],
                            data['back_feat'], data['back_adj'], data['back_e_feat'])

                        losses.append(loss)
                        accs.append(acc)


                    for element in test_dist_dataset:
                        
                        data = element[0]
                        
                        loss_val, acc_val = distributed_eval_step(
                            data['front_feat'], data['front_adj'], data['front_e_feat'],
                            data['back_feat'], data['back_adj'], data['back_e_feat'])
                        

                        losses_val.append(loss_val)
                        accs_val.append(acc_val)

                        
                    print('*' * 40)
                    print(f'Epoch: {epoch}')
                    print(f'Training loss: {np.mean(losses):.4f}, Training accuracy: {np.mean(accs):.4f}')
                    print(f'Validation loss: {np.mean(losses_val):.4f}, Validation accuracy: {np.mean(accs_val):.4f}')

                    for element in screen_dist_dataset:
                        
                        data = element
                        
                        pred, eval_id, gts = distributed_screen_step(data['front_feat'], data['front_adj']
                            ,data['front_e_feat'], data['pairs_id'],  data['Inhibition'])

                        local_preds = tf.distribute.get_strategy().experimental_local_results(pred)
                        local_eval_ids = tf.distribute.get_strategy().experimental_local_results(eval_id)
                        local_gts = tf.distribute.get_strategy().experimental_local_results(gts)

                        for replica_preds, replica_ids, replica_gts in zip(local_preds, local_eval_ids, local_gts):
                            preds.append(replica_preds.numpy())
                            local_ids.append(replica_ids.numpy())
                            inhibitions.append(replica_gts.numpy())

                    preds = np.reshape(np.array(preds),-1)
                    local_ids =  np.reshape(np.array(local_ids),-1)
                    inhibitions =  np.reshape(np.array(inhibitions),-1)

                    print('Pearson corr:{:.4f}' .format(np.corrcoef(preds,inhibitions)[0, 1]))
                    print('Spearman correlation:{:.4f}' .format(stats.spearmanr(preds,inhibitions)[0]))

                    this_acc = np.mean(accs_val)
                    roche_corr = np.corrcoef(preds,inhibitions)[0, 1]

                    df_save.loc[epoch] = [
                        epoch,
                        np.mean(losses),
                        np.mean(accs),
                        np.mean(losses_val),
                        this_acc, roche_corr]

                    if this_acc > best_acc:
                        best_epoch, best_acc = epoch, this_acc
                        model.save_weights(checkpoint_prefix)
                        print(f"[*] Best epoch: {best_epoch}")

                    if roche_corr > best_corr:
                        best_epoch_roche = epoch
                        best_corr = roche_corr
                        model.save_weights(checkpoint_prefix_roche)
                        print("[*] Best epoch: {}".format(best_epoch_roche))
                        
                    df_save['epoch'] = df_save['epoch'].astype(int)
                    df_save.to_csv(f'{checkpoint_dir}/log.csv', float_format='%.4f', index=False)


    def screen(self, dataset,  node_dim, edge_dim, model_height, num_head, num_vnode,max_length):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings(action='ignore')

        self.dataset = dataset
        NUM_GPU = len(utils.get_available_gpus())

        screen_data = self.dataset.screen.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        checkpoint_dir = './checkpoints/training_checkpoints'
        

        model = self.EGT_Backbone(node_dim, edge_dim, model_height, num_head, num_vnode,max_length)
        model.summary()
        model.load_weights(os.path.join(checkpoint_dir, 'best_checkpoint'))

        @tf.function
        def screen_step(merge_nodes, merge_adjs, merge_edges):
            merge_out, _ = model([merge_nodes, merge_adjs, merge_edges])

            return merge_out
        
        df_save = pd.DataFrame([], columns=['Preds', 'ids', 'gt'])
        outputs_val = []
        front_ids = []
        Inhibitions = []
 
        for element in screen_data:

            data = element

            output_rank = screen_step(data['front_feat'], data['front_adj'],  data['front_e_feat'])

            Inhibitions.append(data['Inhibition'])
            outputs_val.append(output_rank)
            front_ids.append(data['pairs_id'])


        preds = np.reshape(np.array(outputs_val),-1)
        front_ids =  np.reshape(np.array(front_ids),-1)
        inhibitions =  np.reshape(np.array(Inhibitions),-1)

        print(preds)
        print(inhibitions)

        print('Pearson corr:{:.4f}' .format(np.corrcoef(preds,inhibitions)[0, 1]))
        print('Spearman correlation:{:.4f}' .format(stats.spearmanr(preds,inhibitions)[0]))

        df_save['Preds'] = preds
        df_save['ids'] = front_ids
        df_save['gt'] = inhibitions        

    def test(self, dataset, node_dim, edge_dim, model_height, num_head, num_vnode,max_length):
        self.dataset = dataset

        NUM_GPU = len(utils.get_available_gpus())
        print(f"Num of GPUs: {NUM_GPU}")
        
        mirrored_strategy = tf.distribute.MirroredStrategy()
        BATCH_SIZE_PER_REPLICA = int(FLAGS.batch/NUM_GPU)
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

        with mirrored_strategy.scope():
            test_data = self.dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            test_dist_dataset = mirrored_strategy.experimental_distribute_dataset(test_data)

        checkpoint_dir = './checkpoints/training_checkpoints'
        with mirrored_strategy.scope():
            model = self.EGT_Backbone(node_dim, edge_dim, model_height, num_head, num_vnode,max_length)
            model.summary()
            model.load_weights(os.path.join(checkpoint_dir, 'best_checkpoint'))


        margin = 0.1

        @tf.function
        def eval_step(front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges):
            front_out, _ = model([front_nodes, front_adjs, front_edges],training=False)
            back_out, _ = model([back_nodes, back_adjs, back_edges],training=False)

            preds = back_out - front_out
            results = tf.cast(preds < 0, dtype=tf.int32)

            acc = tf.reduce_sum(tf.cast(tf.equal(results, 1), dtype=tf.float32)) / GLOBAL_BATCH_SIZE

            loss = tf.reduce_sum(tf.maximum(margin + back_out - front_out, 0)) / GLOBAL_BATCH_SIZE

            return loss, acc

        with mirrored_strategy.scope():
            @tf.function
            def distributed_eval_step(front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges):
                per_replica_losses, per_replica_accs = mirrored_strategy.run(eval_step, args=(
                front_nodes, front_adjs, front_edges, back_nodes, back_adjs, back_edges))
                return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                                axis=None), mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                                     per_replica_accs,
                                                                                     axis=None)
            with mirrored_strategy.scope():
                
                losses_val = []
                accs_val = []

                for element in test_dist_dataset:

                    data = element[0]

                    loss_val, acc_val = distributed_eval_step(data['front_feat'], 
                        data['front_adj'], data['front_e_feat'], 
                        data['back_feat'], data['back_adj'], data['back_e_feat'])


                    losses_val.append(loss_val)
                    accs_val.append(acc_val)

                print('*' * 40)
                print(f'Validation loss: {np.mean(losses_val):.4f}, Validation accuracy: {np.mean(accs_val):.4f}')

def main(argv):
    del argv
    dataset = DATASETS['chemical_engineering']()
    log_width = utils.ilog2(587)

    model = ASOptimizer()
    if FLAGS.mode == 'train':
        model.train(
            dataset=dataset,
            node_dim=FLAGS.node_dim,
            edge_dim=FLAGS.edge_dim,
            model_height=FLAGS.model_height,
            num_head=FLAGS.num_head,
            num_vnode=FLAGS.num_vnode,
            max_length=FLAGS.max_length,
        )
    elif FLAGS.mode == 'test':
        model.test(
            dataset=dataset,
            node_dim=FLAGS.node_dim,
            edge_dim=FLAGS.edge_dim,
            model_height=FLAGS.model_height,
            num_head=FLAGS.num_head,
            num_vnode=FLAGS.num_vnode,
            max_length=FLAGS.max_length,
        )

    elif FLAGS.mode == 'screen':
        model.screen(
            dataset=dataset,
            node_dim=FLAGS.node_dim,
            edge_dim=FLAGS.edge_dim,
            model_height=FLAGS.model_height,
            num_head=FLAGS.num_head,
            num_vnode=FLAGS.num_vnode,
            max_length=FLAGS.max_length,
        )

if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_string('mode', 'train', '[train or test or screen]')
    flags.DEFINE_string('load_dir', 'y-m-d-h-m-s', 'Load the checkpoint files in this directory')
    flags.DEFINE_integer('epochs', 200, 'Number of residual layers per stage.')
    flags.DEFINE_integer('max_length', 516, 'max node size.')
    flags.DEFINE_integer('node_dim', 64, 'Node dimension of Inputs.')
    flags.DEFINE_integer('edge_dim', 64, 'Edge dimension of Inputs.')
    flags.DEFINE_integer('model_height', 12, 'Height of model.')
    flags.DEFINE_integer('num_head', 32, 'Number of heads.')
    flags.DEFINE_integer('num_vnode', 8, 'Number of virtual nodes.')
    flags.DEFINE_integer('batch', 2, 'Batch size.')
    flags.DEFINE_float('lr', 0.00001, 'Learning rate.')
    flags.DEFINE_float('wd', 0.0000, 'Weight decay.')
    flags.DEFINE_float('type', 0, 'index.')

    app.run(main)
