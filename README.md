# GSEA

[NLPCC 2024] The source codes for `GSEA: Global Structure-Aware Graph Neural Networks for Entity Alignment`.

## Getting Started

### Datasets
We use entity alignment benchmark datasets **OpenEA** which can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA). You need to put the prepared data into `../data/` folder.

### Dependencies
+ Python 3
+ PyTorch
+ networkx==2.5.1
+ Scipy
+ Numpy
+ Pandas
+ Scikit-learn

You can automatically download corresponding dependencies by following scripts:
```
conda create -n GSEA python=3.6
conda activate GSEA
conda install -n GSEA pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3.1 -c pytorch # change according to your need here
pip install -r .\requirements.txt
```

### Running
To run GSEA, please use the following scripts (ps: --task is an argument):
```
python train.py --task en_fr_15k
python train.py --task en_de_15k
python train.py --task d_w_15k
python train.py --task d_y_15k
```

To run 5-fold cross-validation, please use the following script:
```
python run_fold.py --task en_fr_15k
```


> If you have any difficulty or question in running code and reproducing experimental results, please email to wangcunda1213@163.com.

## Acknowledgement
We refer to the codes of these repos: GAEA, GCN-Align, OpenEA, MuGNN, IMEA. Thanks for their great contributions!
