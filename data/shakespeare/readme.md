# Data preprocessing for Shakespeare dataset

## 📋 Source
https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare

## 🗺 Steps

Please go through the following steps in Linux Python environment that has already installed `numpy` and `pillow` packages. If running in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` in installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

1. git clone from https://github.com/TalwalkarLab/leaf/tree/master, and dive into `leaf/data/shakespeare/`.

2. run `./preprocess.sh -s niid --sf 1.0 -k 0 -t user --tf 0.9 --smplseed 0 --spltseed 0`. After the processing is finished, there should be two new folders, namely `train` and `test`, containing `all_data_0_0_keep_0_train_9.json` and `all_data_0_0_keep_0_test_9.json` respectively.

3. place the json files according to the paths `--shakespeare_train_path` and `--shakespeare_test_path` in `utils.py`.