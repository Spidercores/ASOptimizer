# ASOptimizer

11/06/2024 - Our paper "ASOptimizer: Optimizing antisense oligonucleotides through deep learning for IDO1 gene regulation" has been published at *Molecular Therapy Nucleic Acids*.

06/12/2024 - The code for chemical engineering and the corresponding experimental data are publicly available for reference and use.

To access ASOptimizer, we have developed a web server that runs ASOptimizer on the backend and provided free access to it. You can access it through the following link: http://asoptimizer.s-core.ai/

> **ASOptimizer consists of a database and two computational models: *sequence engineering* and *chemical engineering***.

> **This repository focuses exclusively on the *chemical engineering* , providing related data and source code**.

## 📂 Key Components

- **Chemical Engineering**  
  Algorithms and data pipelines for molecular structure analysis and optimization.
  To learn more about sequence engineering described in the MTNA paper, please refer to [URL](https://github.com/minkang23/ASOptimizer_sequence).

- **Dataset**  
  Includes experimentally validated chemical data and simulation results.

## Data Sources
The data used in this project is sourced from the following:

1. [Dataset from *Spidercore Inc.*] ([URL](https://github.com/Spidercores/ASOptimizer/tree/main/dataset) to dataset)  
   - Description: A curated dataset of in vitro experimental results, primarily gathered from published data samples on Lens.org through an iterative data collection process.
   - Source: Spidercore Inc. Special thanks to our dedicated team members for their efforts in compiling and organizing this dataset.  

2. [Dataset source from *Roche.*] ([URL](https://www.sciencedirect.com/science/article/pii/S2162253119304068))  
   - Description: This dataset contains 256 experimental results of ASO experiments trageting HIF1A gene discussed in the following publication:
   - Source: Papargyri, N., Pontoppidan, M., Andersen, M.R., Koch, T., and Hagedorn, P.H.
(2020). Chemical Diversity of Locked Nucleic Acid-Modified Antisense
Oligonucleotides Allows Optimization of Pharmaceutical Properties. Mol. Ther.
Nucleic Acids 19, 706–717.


Please refer to the original sources for more details on the data and its terms of use.

## 🚀 Getting Started

### Clone the Repository
```bash
git clone https://github.com/Spidercores/ASOptimizer.git
cd ASOptimizer
mkdir data
```

###  Install Dependencies
1. Ensure you have Python 3.8 and tensorflow 2.3.0.
2. Install the required Python packages:

```bash
pip3 install -r requirements.txt
```

### Data preprocessing
- data_type: 'training'

  Creates a paired dataset (graph representation) for chemical engineering tasks (D_train, D_test).

- data_type: 'screening'
  
  Generates a dataset (graph representation) from experimental data provided by Roche.
  
```bash
python3 libpreprocess/make_tfrecords.py --data_type 'training'
python3 libpreprocess/make_tfrecords.py --data_type 'screening'
```

### Run Training and Evaluations

- mode: 'train'

Train ASOptimizer using Dtrain.

- mode: 'test'

Test ASOptimizer using Dtest.

- mode: 'screen'

Evaluate and generate scores for the desired ASO using Dscreen. The data used for screening is currently set to Roche's dataset, stored in the screening folder.

```bash
python3 main.py --mode 'train'
python3 main.py --mode 'test'
python3 main.py --mode 'screen'
```

### Checkpoints
We provide a pre-trained model checkpoint on our paper. These checkpoint can be used to evaluate directly for inference.  

1. **Model Checkpoint (`checkpoints/training_checkpoints`)**  
```bash
  checkpoint_dir = './checkpoints/training_checkpoints'
  with mirrored_strategy.scope():
      model = self.EGT_Backbone(node_dim, edge_dim, model_height, num_head, num_vnode,max_length)
      model.summary()
      model.load_weights(os.path.join(checkpoint_dir, 'best_checkpoint'))
```

### Model architecture

The model architecture used in this project is based on the implementation from [this GitHub repository](https://github.com/shamim-hussain/egt). The code has been adapted and extended to suit the specific requirements of our project.
We express our gratitude to the original authors for their valuable contributions to the open-source community.

### Citing this work
Please cite the following paper if you find the code and data useful:
```bash
@article{ASOptimizer2024, 
    title = {ASOptimizer: Optimizing antisense oligonucleotides through deep learning for IDO1 gene regulation}, 
    author = {Hwang, Gyeongjo and Kwon, Mincheol and Seo, Dongjin and Kim, Dae Hoon and Lee, Daehwan and Lee, Kiwon and Kim, Eunyoung and Kang, Mingeun and Ryu, Jin-Hyeob}, 
    journal = {Molecular Therapy - Nucleic Acids}, 
    volume = {35}, 
    number = {2}, 
    pages = {102186}, 
    year = {2024}, 
    month = jun, 
    DOI = {10.1016/j.omtn.2024.102186}, 
    ISSN = {2162-2531}, 
    publisher = {Elsevier BV}, 
    url = {http://dx.doi.org/10.1016/j.omtn.2024.102186}}   
```

### Get in Touch
If you have any questions not addressed in this repository or if you encounter any errors or inconsistencies in the data, please feel free to reach out to our team at *minkang@spidercore.io* and *hkj4276@spidercore.io*.

We would love to hear your thoughts and learn how ASOptimizer has benefited your research. Please share your experiences with us at *kiwon@spidercore.io*.

### User Licence
> User License: Creative Commons Attribution – NonCommercial – NoDerivs (CC BY-NC-ND 4.0) | Elsevier's open access license policy