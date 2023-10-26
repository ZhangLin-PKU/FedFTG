#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python ../train.py --seed 1024 --model_arch 'resnet18'  --method 'FedProxGAN' --dataset 'CIFAR10' --print_freq 5 \
  --save_period 200 \
  --n_client 100 --rule 'Dirichlet' --alpha 0.3  --sgm 0 \
  --localE 5  --comm_amount 1000  --active_frac 0.1 --bs 50 \
  --lr 0.1  --weight_decay 1e-3 --mu 1e-4 --lr_decay 0.998