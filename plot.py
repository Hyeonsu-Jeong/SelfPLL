import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_csv(ctype='sym', wd=0):

    
    df = pd.read_csv(f'results/{ctype}.csv', header=0)

    WDs = sorted(df['WD'].unique())
    Ds = sorted(df['Dataset'].unique())
    Dists = sorted(df['Distill'].unique())
    Cs = sorted(df['Corruption'].unique())
    LRs = sorted(df['LR'].unique())

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

    flat_ax = axes.flat
    d = {'caltech101':'Caltech-101', 'caltech256':'Caltech-256', 'cifar100':'CIFAR-100', 'flowers102':'Flowers-102', 'food101':'Food-101', 'stanfordcars':'StanfordCars', }
    if wd == 'base':
            W = {'cifar100': 5e-5, 'caltech256':5e-5, 'caltech101':5e-4, 'flowers102':5e-4, 'stanfordcars':5e-4, 'food101':5e-5}
    elif wd == 'small':
        W = {'cifar100': 1e-4, 'caltech256':1e-4, 'caltech101':1e-3, 'flowers102':1e-3, 'stanfordcars':1e-3, 'food101':1e-4}
    else:
        W = {'cifar100': 2e-5, 'caltech256':2e-5, 'caltech101':2e-4, 'flowers102':2e-4, 'stanfordcars':2e-4, 'food101':2e-5}
        
    Cs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    bar_width = 1.8  # 막대 너비 설정
    colors = ['#CCCCCC', '#888888', '#444444', 'black', 'C1']
    for idx, ax in enumerate(flat_ax):
        
        base = None
        for dist_idx, dist in enumerate(Dists):
            acc = np.zeros((len(Cs), 3))
            
            for j, corr in enumerate(Cs):
                sub_df = df[(df['Dataset'] == Ds[idx]) & (df['Distill'] == dist) & (df['Corruption'] == corr) & (df['WD'] == W[Ds[idx]])]
                acc[j] = sub_df['Test Acc.']
                
            if dist == '0':
                base = acc
                
            else:
                mu = np.mean(acc-base, axis=1)
                std = np.std(acc-base, axis=1)
                x = 100*Cs - (3-dist_idx)*bar_width
                
                ax.errorbar(x, mu, yerr=std, fmt='none', color='grey', capsize=3)
                ax.bar(x, mu, bar_width, label=f't={dist}' if dist != 'p' else 'PLL', color=colors[dist_idx-1])

        ax.set_ylim(-0.2*ax.get_ylim()[1], ax.get_ylim()[1])
        ax.axhline(0, color='red', linestyle='-', linewidth=2, label='base')
        legend_properties = {'weight': 'bold', 'size': 12}
        ax.legend(prop=legend_properties)
        ax.set_title(d[Ds[idx]], weight='bold', fontsize=20)
        
        if idx % 3 == 0:
            ax.set_ylabel('Distillation Gain (%)', fontsize=15)
        
        if idx // 3 == 1:
            ax.set_xlabel('Corruption Ratio '+r'$\eta$' +' (%)', fontsize=15)
            
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
    plt.tight_layout()
    if ctype == 'cc':
        ctype = 'superclass'
    plt.savefig(f'fig/{ctype}_{wd}.pdf')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctype', default='cc', type=str)
    parser.add_argument('--wd', default='base', type=str)
    args = parser.parse_args()
    
    plot_csv(args.ctype, args.wd)