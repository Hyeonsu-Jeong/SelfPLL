# Training Recipe

※ Run the following code in the *root* directory.

## Training Individual Models with Three Different Seeds (0, 1, 2)

```bash
./train.sh --lr [LR] --wd [WD]
```

## Reproducing Figure 2 (main text)

```bash
./train.sh --dataset cifar100 --wd 5e-5 --lr 0.05 --ctype cc
./train.sh --dataset caltech256 --wd 5e-5 --lr 0.01 --ctype cc
./train.sh --dataset food101 --wd 5e-5 --lr 0.01 --ctype cc

./train.sh --dataset caltech101 --wd 5e-4 --lr 0.01 --ctype cc
./train.sh --dataset stanfordcars --wd 5e-4 --lr 0.1 --ctype cc
./train.sh --dataset flowers102 --wd 5e-4 --lr 0.05 --ctype cc

python plot.py --ctype cc --wd base
python table.py --ctype cc --wd base # for corresponding tables in Appendix G.3
```

## Reproducing Figure 6(a) (Appendix C)
```bash
./train.sh --dataset cifar100 --wd 5e-5 --lr 0.05 --ctype sym
./train.sh --dataset caltech256 --wd 5e-5 --lr 0.01 --ctype sym
./train.sh --dataset food101 --wd 5e-5 --lr 0.01 --ctype sym

./train.sh --dataset caltech101 --wd 5e-4 --lr 0.01 --ctype sym
./train.sh --dataset stanfordcars --wd 5e-4 --lr 0.1 --ctype sym
./train.sh --dataset flowers102 --wd 5e-4 --lr 0.05 --ctype sym

python plot.py --ctype sym --wd base
python table.py --ctype sym --wd base # for corresponding tables in Appendix G.3
```

## Reproducing Figure 6(b) (Appendix C)
```bash
./train.sh --dataset cifar100 --wd 5e-5 --lr 0.05 --ctype asym
./train.sh --dataset caltech256 --wd 5e-5 --lr 0.01 --ctype asym
./train.sh --dataset food101 --wd 5e-5 --lr 0.01 --ctype asym

./train.sh --dataset caltech101 --wd 5e-4 --lr 0.01 --ctype asym
./train.sh --dataset stanfordcars --wd 5e-4 --lr 0.1 --ctype asym
./train.sh --dataset flowers102 --wd 5e-4 --lr 0.05 --ctype asym

python plot.py --ctype asym --wd base
python table.py --ctype asym --wd base # for corresponding tables in Appendix G.3
```

## Reproducing Figure 7(a) (Appendix G.1)
```bash
./train.sh --dataset cifar100 --wd 2e-5 --lr 0.05 --ctype cc
./train.sh --dataset caltech256 --wd 2e-5 --lr 0.01 --ctype cc
./train.sh --dataset food101 --wd 2e-5 --lr 0.01 --ctype cc

./train.sh --dataset caltech101 --wd 2e-4 --lr 0.01 --ctype cc
./train.sh --dataset stanfordcars --wd 2e-4 --lr 0.1 --ctype cc
./train.sh --dataset flowers102 --wd 2e-4 --lr 0.05 --ctype cc

python plot.py --ctype cc --wd small
python table.py --ctype cc --wd small # for corresponding tables in Appendix G.3
```

## Reproducing Figure 7(b) (Appendix G.1)
```bash
./train.sh --dataset cifar100 --wd 2e-5 --lr 0.05 --ctype sym
./train.sh --dataset caltech256 --wd 2e-5 --lr 0.01 --ctype sym
./train.sh --dataset food101 --wd 2e-5 --lr 0.01 --ctype sym

./train.sh --dataset caltech101 --wd 2e-4 --lr 0.01 --ctype sym
./train.sh --dataset stanfordcars --wd 2e-4 --lr 0.1 --ctype sym
./train.sh --dataset flowers102 --wd 2e-4 --lr 0.05 --ctype sym

python plot.py --ctype sym --wd small
python table.py --ctype sym --wd small # for corresponding tables in Appendix G.3

```

## Reproducing Figure 7(c) (Appendix G.1)
```bash
./train.sh --dataset cifar100 --wd 2e-5 --lr 0.05 --ctype asym
./train.sh --dataset caltech256 --wd 2e-5 --lr 0.01 --ctype asym
./train.sh --dataset food101 --wd 2e-5 --lr 0.01 --ctype asym

./train.sh --dataset caltech101 --wd 2e-4 --lr 0.01 --ctype asym
./train.sh --dataset stanfordcars --wd 2e-4 --lr 0.1 --ctype asym
./train.sh --dataset flowers102 --wd 2e-4 --lr 0.05 --ctype asym

python plot.py --ctype asym --wd small
python table.py --ctype asym --wd small # for corresponding tables in Appendix G.3
```

## Reproducing Figure 8(a) (Appendix G.1)
```bash
./train.sh --dataset cifar100 --wd 1e-4 --lr 0.05 --ctype cc
./train.sh --dataset caltech256 --wd 1e-4 --lr 0.01 --ctype cc
./train.sh --dataset food101 --wd 1e-4 --lr 0.01 --ctype cc

./train.sh --dataset caltech101 --wd 1e-3 --lr 0.01 --ctype cc
./train.sh --dataset stanfordcars --wd 1e-3 --lr 0.1 --ctype cc
./train.sh --dataset flowers102 --wd 1e-3 --lr 0.05 --ctype cc

python plot.py --ctype cc --wd_type big
python table.py --ctype cc --wd_type big # for corresponding tables in Appendix G.3
```

## Reproducing Figure 8(b) (Appendix G.1)
```bash
./train.sh --dataset cifar100 --wd 1e-4 --lr 0.05 --ctype sym
./train.sh --dataset caltech256 --wd 1e-4 --lr 0.01 --ctype sym
./train.sh --dataset food101 --wd 1e-4 --lr 0.01 --ctype sym

./train.sh --dataset caltech101 --wd 1e-3 --lr 0.01 --ctype sym
./train.sh --dataset stanfordcars --wd 1e-3 --lr 0.1 --ctype sym
./train.sh --dataset flowers102 --wd 1e-3 --lr 0.05 --ctype sym

python plot.py --ctype asym --wd_type big
python table.py --ctype asym --wd_type big # for corresponding tables in Appendix G.3
```

## Reproducing Figure 8(c) (Appendix G.1)
```bash
./train.sh --dataset cifar100 --wd 1e-4 --lr 0.05 --ctype asym
./train.sh --dataset caltech256 --wd 1e-4 --lr 0.01 --ctype asym
./train.sh --dataset food101 --wd 1e-4 --lr 0.01 --ctype asym

./train.sh --dataset caltech101 --wd 1e-3 --lr 0.01 --ctype asym
./train.sh --dataset stanfordcars --wd 1e-3 --lr 0.1 --ctype asym
./train.sh --dataset flowers102 --wd 1e-3 --lr 0.05 --ctype asym

python plot.py --ctype asym --wd_type big
python table.py --ctype asym --wd_type big # for corresponding tables in Appendix G.3
```
