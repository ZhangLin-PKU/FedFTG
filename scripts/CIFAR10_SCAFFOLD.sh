#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python ../train.py --seed 1024 --model_arch 'resnet18'  --method 'SCAFFOLD' --dataset 'CIFAR10' --print_freq 5 \
  --save_period 200 \
  --n_client 100 --rule 'Dirichlet' --alpha 0.6  --sgm 0 \
  --n_minibatch 50  --comm_amount 1000  --active_frac 0.1 --bs 50 \
  --lr 0.1  --weight_decay 1e-3 --n_minibatch 50 --lr_decay 0.998
