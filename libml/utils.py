from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

import tensorflow as tf
import numpy as np
import os
import warnings

_GPUS = None

def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        local_device_protos = device_lib.list_local_devices()
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS

def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def ilog2(x):
    """Integer log2."""
    return int(np.ceil(np.log2(x)))


def get_logger(name=__file__):
    import logging
    import logging.handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return logger

def random_init(shape, dtype=None, k=1.0, filters=16):
    return K.random_normal(shape, stddev=tf.math.rsqrt(k * k * filters), dtype=dtype)





class ExponentialMovingAverage(Callback):
    """create a copy of trainable weights which gets updated at every
       batch using exponential weight decay. The moving average weights along
       with the other states of original model(except original model trainable
       weights) will be saved at every epoch if save_mv_ave_model is True.
       If both save_mv_ave_model and save_best_only are True, the latest
       best moving average model according to the quantity monitored
       will not be overwritten. Of course, save_best_only can be True
       only if there is a validation set.
       This is equivalent to save_best_only mode of ModelCheckpoint
       callback with similar code. custom_objects is a dictionary
       holding name and Class implementation for custom layers.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of the training, the moving
       averaged weights are transferred to the original model.
       """
    def __init__(self, decay=0.999, filepath='temp_weight.hdf5',
                 save_mv_ave_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto',
                 save_weights_only=False, custom_objects={}, ema_model=None):
        self.decay = decay
        self.filepath = filepath
        self.verbose = verbose
        self.save_mv_ave_model = save_mv_ave_model
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.custom_objects = custom_objects  # dictionary of custom layers
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        self.ema_model = ema_model
        super(ExponentialMovingAverage, self).__init__()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        self.sym_trainable_weights = self.model.trainable_weights
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}
        if self.verbose:
            print('Created a copy of model weights to initialize moving'
                  ' averaged weights.')

    def on_batch_end(self, batch, logs={}):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we can optionally save the moving averaged model,
        but the weights will NOT be transferred to the original model. This
        happens only at the end of training. We also need to transfer state of
        original model to model2 as model2 only gets updated trainable weight
        at end of each batch and non-trainable weights are not transferred
        (for example mean and var for batch normalization layers)."""
        if self.save_mv_ave_model:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('saving moving average model to %s'
                                  % (filepath))
                        self.best = current
                        self._make_mv_model(filepath)
                        if self.save_weights_only:
                            self.ema_model.save_weights(filepath, overwrite=True)
                        else:
                            self.ema_model.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving moving average model to %s' % (epoch, filepath))
                self._make_mv_model(filepath)
                if self.save_weights_only:
                    self.ema_model.save_weights(filepath, overwrite=True)
                else:
                    self.ema_model.save(filepath, overwrite=True)

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self, filepath):
        """ Create a model with moving averaged weights. Other variables are
        the same as original mode. We first save original model to save its
        state. Then copy moving averaged weights over."""
        # self.model.save(filepath, overwrite=True)
        # self.emamodel2 = clone_model(self.model)#load_model(filepath, custom_objects=self.custom_objects)

        for w2, w in zip(self.ema_model.trainable_weights, self.model.trainable_weights):
            K.set_value(w2, self.mv_trainable_weights_vals[w.name])

        # return model2