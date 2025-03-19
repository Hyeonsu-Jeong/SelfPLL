# [ICLR2025] Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels 

This repository contains code for our paper: "Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels".

## Abstract

We investigate the mechanisms of self-distillation in multi-class classification, particularly in the context of linear probing with fixed feature extractors where traditional feature learning explanations do not apply. Our theoretical analysis reveals that multi-round self-distillation effectively performs label averaging among instances with high feature correlations, governed by the eigenvectors of the Gram matrix derived from input features. This process leads to clustered predictions and improved generalization, mitigating the impact of label noise by reducing the model's reliance on potentially corrupted labels. We establish conditions under which multi-round self-distillation achieves 100\% population accuracy despite label noise. Furthermore, we introduce a novel, efficient single-round self-distillation method using refined partial labels from the teacher's top two softmax outputs, referred to as the PLL student model. This approach replicates the benefits of multi-round distillation in a single round, achieving comparable or superior performance--especially in high-noise scenarios--while significantly reducing computational cost. 


## Environments

- Python 3.12.2
- PyTorch 2.2.2

## Usage

We provide the following commands to run the code:

```bash
python train_linear.py [OPTIONS]
```
### Options for `train_linear.py`

| Option                             | Type  | Default   | Description                                                                                        |
|------------------------------------|-------|-----------|----------------------------------------------------------------------------------------------------|
| `-d, --dataset DATASET`            | str   | -         | Specify dataset  |
| `-j, --workers N`                  | int   | 4         | Number of data loading workers.                                                                     |
| `--epochs N`                       | int   | 200       | Number of total epochs to run.                                                                      |
| `--train-batch N`                  | int   | 128       | Training batch size.                                                                                |
| `--test-batch N`                   | int   | 100       | Test batch size.                                                                                    |
| `--lr, --learning-rate LR`         | float | 5e-4         | Initial learning rate.                                                                              |
| `--gamma GAMMA`                    | float | 0.98      | Learning decay value.                                                                               |
| `--momentum M`                     | float | 0.9       | Momentum.                                                                                           |
| `--weight-decay W, --wd W`         | float | 1e-4      | Weight decay.                                                                                       |
| `-c, --checkpoint PATH`            | str   | -         | Path to save checkpoint.                                                                            |
| `--depth DEPTH`                    | int   | 34        | Model depth.                                                                                        |
| `--manualSeed MANUALSEED`          | int   | 0         | Manual seed.                                                                                        |
| `--gpu-id GPU_ID`                  | str   | 0         | ID(s) for CUDA_VISIBLE_DEVICES.                                                                     |
| `--dist DIST`                      | int   | 0         | Distillation count (0: teacher).                                                                    |
| `--last LAST`                      | int   | 0         | Use best/last checkpoint of teacher model (0: best, 1: last).                                       |
| `--ctype CTYPE`                    | str   | 'cc'         | Corruption type  |
| `--corruption CORR`                | float | 0.5         | Label corruption ratio, range [0, 1].                                                               |
| `--partial PARTIAL`                | bool  | False     | Specify PLL student mode 
---

```bash
python numerical_solving.py [OPTIONS]
```

### Options for `numerical_solving.py`

| Option                         | Type   | Default   | Description                                                                               |
|--------------------------------|--------|-----------|-------------------------------------------------------------------------------------------|
| `--k NUM_CLASSES`              | int    | 4         | Specify the number of classes.                                                            |
| `--n NUM_SAMPLES`              | int    | 500       | Number of samples per class.                                                              |
| `--lbd REG_PARAM`              | float  | 1e-5      | Regularization parameter.                                                                 |
| `--c INTRA_CLASS_CORR`         | float  | 0.4       | Intra-class correlation.                                                                  |
| `--d INTER_CLASS_CORR`         | float  | 0.1       | Inter-class correlation.                                                                  |
| `--eta CORR_RATIO`             | float  | 0.5       | Corruption ratio.                                                                         |
| `--epsilon NOISE_LEVEL`        | float  | 0.05      | Noise level of the corruption matrix.                                                     |
| `--max_dist DIST_ROUNDS`       | int    | 10        | Total distillation rounds.                                                                |
| `--max_iter MAX_ITERATIONS`    | int    | 100,000   | Maximum number of iterations.                                                             |
| `--approx USE_APPROX`          | bool   | False     | Use linear approximation.                                                                 |

## How to reproduce our empirical results

Please refer to the [Training Recipe](./recipe.MD).

## References

We are grateful to the [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) repository for providing resources that were essential to our project.
