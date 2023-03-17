# Multi-IVE: Privacy Enhancement of Multiple Soft-Biometrics in Face Embeddings
This repository contains the source code to reproduce the following paper:
```BibTeX
@inproceedings{multi-ive,
  title={Multi-IVE: Privacy Enhancement of Multiple Soft-Biometrics in Face Embeddings},
  author={Melzi, Pietro and Shahreza, Hatef Otroshi and Rathgeb, Christian and Tolosana, Ruben and Vera-Rodriguez, Ruben and Fierrez, Julian and Marcel, S{\'e}bastien and Busch, Christoph},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={323--331},
  year={2023}
}
```

## Usage
To create a virtual environment run the command:

```
$ conda env create --name envname --file=environment.yml
```

## Training

To generate masks that represent the order of feature elimination run the command:

```
python main.py
```

Masks can be generated for the following settings:
* no domain transformation
* pca domain transformation with k={0, 3, 5}
* ica domain transformation

## Evaluation

Use the file [evaluate_IVE.py](https://github.com/otroshi/multi-ive/blob/main/evaluate_IVE.py) to evaluate Multi-IVE as in the paper. 

Use the file [first_components_pca.py](https://github.com/otroshi/multi-ive/blob/main/first_components_pca.py) to evaluate Multi-IVE in the scenario of principal component elimination in the order of their importance. 

Use the file [rdm_mask.py](https://github.com/otroshi/multi-ive/blob/main/rdm_mask.py) to evaluate Multi-IVE in the scenario of random feature elimination.  
