# Data preprocessing for CIFAR-100 dataset

## 📋 Source
https://github.com/KarhouTam/FL-bench/tree/master/data

## 🗺 Steps

Please go through the following steps in Linux Python environment. If running in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` in installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

The steps 1 and 2 describe how to reproduce our data partition and train-test split. If followed correctly, a file named `partition.pkl` will be generated, which should be identical to the pkl files in this directory. For simplicity, you may want to skip steps 1 and 2, and directly use the pkl files here instead.

1. git clone from https://github.com/KarhouTam/FL-bench/tree/master, and install Fl-bench environment, i.e., `pip install -r .environment/requirements.txt`.

2. follow the steps in https://github.com/KarhouTam/FL-bench/tree/master/data. Particularly: 
    
    - run `python generate_data.py -d cifar100 --iid 1 -cn 100 -tr 0.1` for iid partition.
    - run `python generate_data.py -d cifar100 -a 0.1 -cn 100 -tr 0.1` for non-iid partition following Dirichlet distribution. The parameter `-a` or `--alpha` controlls heterogeneity. Smaller alpha implies stonger heterogeinty. For our experiments we used alpha values 1.0, 0.5, and 0.1.

3. place the generated `partition.pkl` files according to the path `--cifar100_path` in `utils.py`.