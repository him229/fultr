# FULTR: Policy-Gradient Training of Fair and Unbiased Ranking Functions

The implementation for SIGIR 2021 paper ([arxiv](https://arxiv.org/pdf/1911.08054.pdf)):  

Policy-Gradient Training of Fair and Unbiased Ranking Functions

Himank Yadav*, Zhengxiao Du*, Thorsten Joachims (*: equal contribution)



## Installation
Clone the repo
```shell
git clone https://github.com/him229/fultr
cd fultr
```
Please first install PyTorch, and then install other dependencies by
```shell
pip install -r requirements.txt
```
## Getting Started

Script `main.sh` contains commands for running various experiments in the paper.

## Data

`datasets` folder contains links to download the datasets used for experiments and code we used to transform the datasets to make them suitable for training. 

`transformed_datasets` folder contains the final version of the transformed dataset that we directly use for training.