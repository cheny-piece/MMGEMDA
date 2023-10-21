## Paper title
Metapath-aggregated multilevel graph embedding for miRNA‒disease association prediction

### Dependencies

Recent versions of the following packages for Python 3 are required:
* python 3.8.16 
* pytorch 1.12.0 
* networkx 3.0 
* scikit-learn 1.2.1 
* numpy 1.23.5
* scipy 1.10.1

### Usage
1. run 0FeatureGenerate.py to generate feautures for miRNA and disease
2. run 1preprocess_md.py to generate metapaths' information
3. run 2preprocess_md_negsampling_Kfold.py to split the dataset into 5 folds for cross-validation
4. run 3run_MD.py to obtain the predition results.


### Citing

