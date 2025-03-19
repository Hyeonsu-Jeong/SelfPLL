import pandas as pd
import numpy as np
import sys

def convert_to_latex_table_with_max(data, caption):
    header = data[0]
    num_cols = len(header)
    
    max_positions = [np.argmax([float(row[i].split('$')[0]) for row in data[1:]]) for i in range(1, num_cols)]
    
    latex_table = "\\begin{minipage}[b]{\\linewidth}\n"
    
    latex_table += "\\caption{"+caption+"}\n"
    latex_table += "\\centering\n"
    latex_table += "\\resizebox{\\linewidth}{!}{\n"
    latex_table += "\\begin{tabular}{c" + "|c"*(num_cols-1) + "}\n"
    latex_table += "\\toprule\n"
    
    latex_table += " & ".join(header) + " \\\\\n"
    latex_table += "\\midrule\n"
    
    for row_idx, row in enumerate(data[1:]):
        for idx in range(1,len(row)):
            row[idx] = row[idx]+"\\%"
        for idx, max_pos in enumerate(max_positions):
            if max_pos == row_idx:
                row[idx+1] = " \\textbf{"+row[idx+1]+"}"
        row_str = " & ".join(row)
        if row_str.startswith('PLL'):
            latex_table += '\\midrule\n'
        latex_table += f"{row_str}  \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}}\n"
    latex_table += "\\end{minipage}\n"
    latex_table += "\\vspace{10pt}\n"
    
    print(latex_table)

def convert_to_latex(value):
    exponent = 0
    while value < 1:
        value *= 10
        exponent -= 1
    return f"{value:.0f}\\times 10^{{{exponent}}}"

def create_table(ctype, wdtype):
    df = pd.read_csv(f'results/{ctype}.csv', header=0)
    with open(f'results/{ctype}_{wdtype}.txt', 'w') as f:
        sys.stdout = f
        if wdtype == 'base':
            W = {'cifar100': 5e-5, 'caltech256':5e-5, 'caltech101':5e-4, 'flowers102':5e-4, 'stanfordcars':5e-4, 'food101':5e-5}
        elif wdtype == 'small':
            W = {'cifar100': 1e-4, 'caltech256':1e-4, 'caltech101':1e-3, 'flowers102':1e-3, 'stanfordcars':1e-3, 'food101':1e-4}
        else:
            W = {'cifar100': 2e-5, 'caltech256':2e-5, 'caltech101':2e-4, 'flowers102':2e-4, 'stanfordcars':2e-4, 'food101':2e-5}
        
        d = {'caltech101':'Caltech-101', 'caltech256':'Caltech-256', 'cifar100':'CIFAR-100', 'flowers102':'Flowers-102', 'food101':'Food-101', 'stanfordcars':'StanfordCars', }
                    
        WDs = sorted(df['WD'].unique())
        Ds = sorted(df['Dataset'].unique())
        Dists = sorted(df['Distill'].unique())
        Cs = sorted(df['Corruption'].unique())
        
        print('\\begin{table}[htbp]')
        print('\\centering')
        for dataset in Ds:
            if ctype == 'cc':
                ctype_str = 'superclass'
            elif ctype == 'sym':
                ctype_str = 'symmetric'
            else:
                ctype_str = 'asymmetric'
            
            caption = f'(Details of Fig.~\\ref{{fig:{ctype if ctype != 'cc' else 'superclass'}_{wdtype}}}) Test accuracy of {d[dataset]} dataset applying {ctype_str} label corruption, where weight decay value $\\lambda={convert_to_latex(W[dataset])}.$'
            
            rows = []
            header = ['Distillation Step'] +[str(x) for x in Cs]
            rows.append(header)
            for dist in Dists:
                if dist == 'p':
                    row = ['PLL']
                elif dist == '0':
                    row = ['1 (Teacher)']
                else:
                    row = [str(int(dist)+1)]
                sub_df = df[(df['Dataset'] == dataset) & (df['Distill'] == dist) & (df['WD'] == W[dataset])]    
                for corruption in Cs:
                    mu = np.mean(sub_df[sub_df['Corruption'] == corruption]['Test Acc.'])
                    sigma = np.std(sub_df[sub_df['Corruption'] == corruption]['Test Acc.'])
                    row.append(f'{mu:.2f}$\\pm${sigma:.2f}')
                
                rows.append(row)
            convert_to_latex_table_with_max(rows, caption)
        print('\\end{table}')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctype', default='cc', type=str)
    parser.add_argument('--wd', default='base', type=str)
    args = parser.parse_args()
    
    create_table(args.ctype, args.wd)