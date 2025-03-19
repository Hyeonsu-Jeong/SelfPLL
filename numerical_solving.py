from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np
import torch.nn.functional as F
import torch
import pickle
from utils import Bar

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=0, keepdim=True)[0])  
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

def linearlize(x):
    K, _ = x.shape
    return x / K + 1 / K

def objective(target, pred, correlation, Knl, approx=False):
    if approx:
        return torch.norm(pred - linearlize((target - pred) @ correlation / (Knl)), p=2)**2
    return torch.norm(pred - softmax((target - pred) @ correlation / (Knl)), p=2)**2

def get_correlation_matrix(K, n, c, d, epsilon):
    
    correlation = d * torch.ones((K * n, K * n))
    for i in range(K):
        correlation[i * n:(i + 1) * n, i * n:(i + 1) * n] = c
    
    correlation += torch.rand(K * n, K * n) * 2 * epsilon - epsilon
    
    for i in range(K * n):
        correlation[i, i] = 1

    return correlation
    
def numerical_solving(K, n, lbd, c, d, eta, epsilon=0.05, max_dist=10, max_iter=100000, approx=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if approx == True:
        learning_rate = 1e-2
    else:
        learning_rate = 1e-3

    correlation = get_correlation_matrix(K, n, c, d, epsilon)
    correlation = correlation.to(device)
    
    given_Y = torch.zeros((K, K * n), device=device)
    
    for i in range(K*n):
        if torch.rand(1).item() < eta:
            new_label = torch.randint(0, K - 1, (1,)).item()
            if new_label >= i // n:
                new_label += 1
            given_Y[new_label, i] = 1
        else:
            given_Y[i // n, i] = 1

    output_path = os.path.join('outputs', 'linear' if approx else 'softmax')
    os.makedirs(output_path, exist_ok=True)
    
    
    with open(os.path.join(output_path, f'{K}_{n}_{lbd}_{c}_{d}_{eta}_{epsilon}_{0}.pkl'), 'wb') as f:
        pickle.dump(given_Y, f)       
    target = given_Y.clone()
    
    
    dist_list = list(range(1, max_dist+1))
    for dist in dist_list:
        
        print(f'eta = {eta}, dist = {dist}, approx = {approx}')
        
        Y = torch.rand((K, K * n), device=device, requires_grad=True)
        Y.data = Y.data / torch.sum(Y.data, dim=0, keepdim=True)
    
        optimizer = torch.optim.Adam([Y], lr=learning_rate)
        
        bar = Bar('Processing', max=max_iter)
        for k in range(max_iter):
            optimizer.zero_grad()
            loss = objective(target, Y, correlation, K*n*lbd, approx)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                Y.data = torch.maximum(Y.data, torch.tensor(1e-6, device=device))  # Non-negative constraint
                Y.data = Y.data / torch.sum(Y.data, dim=0, keepdim=True)  # Normalize columns again
                
            if loss.item() < 1e-5:
                print(f'\nConverged after {k + 1} iterations with loss {loss.item():.6f}.')
                break
            
            
            bar.suffix = f'({k + 1}/{max_iter}) Loss: {loss.item():.6f}'
            bar.next()
        bar.finish()
        
        clean_acc = 0
        noisy_acc = 0
        acc = 0
        
        clean_cnt = 0
        clean_correct = 0
        noisy_cnt = 0
        noisy_correct = 0
        
        for j in range(K*n):
            if torch.argmax(given_Y[:, j]).item() == j // n:
                clean_cnt += 1
                if torch.argmax(Y[:, j]).item() == j // n:
                    clean_correct += 1
            else:
                noisy_cnt += 1
                if torch.argmax(Y[:, j]).item() == j // n:
                    noisy_correct += 1
                    
        if clean_cnt > 0:
            clean_acc = clean_correct / clean_cnt * 100
            
        if noisy_cnt > 0:
            noisy_acc = noisy_correct / noisy_cnt * 100
            
        acc = (clean_correct + noisy_correct) / (clean_cnt + noisy_cnt) * 100
        print(f'Clean accuracy: {clean_acc:.2f}%, Noisy accuracy: {noisy_acc:.2f}%, Accuracy: {acc:.2f}%')        
        target = Y.clone()
        target = target.detach()
        with open(os.path.join(output_path, f'{K}_{n}_{lbd}_{c}_{d}_{eta}_{epsilon}_{dist}.pkl'), 'wb') as f:
            pickle.dump(Y, f)
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=4, type=int, help='number of classes')
    parser.add_argument('--n', default=500, type=int, help='number of samples per class')
    
    parser.add_argument('--lbd', default=1e-5, type=float, help='regularization parameter')
    parser.add_argument('--c', default=0.4, type=float, help='intra-class correlation')
    parser.add_argument('--d', default=0.1, type=float, help='inter-class correlation')
    parser.add_argument('--eta', default=0.5, type=float, help='corruption ratio')
    parser.add_argument('--epsilon', default=0.05, type=float, help='noise level of corruption matrix')
    parser.add_argument('--max_dist', default=10, type=int, help='total distillation rounds')
    parser.add_argument('--max_iter', default=100000, type=int, help='maximum iteration')
    
    parser.add_argument('--approx', default=False, type=bool, help='use linear approximation')
    args = parser.parse_args()
    
    numerical_solving(args.k, args.n, args.lbd, args.c, args.d, args.eta, args.epsilon, args.max_dist, args.max_iter, args.approx)
    