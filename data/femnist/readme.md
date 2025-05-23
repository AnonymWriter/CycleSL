# Data preprocessing for FEMNIST dataset

## 📋 Source
https://github.com/TalwalkarLab/leaf/tree/master/data/femnist

## 🗺 Steps

Please go through the following steps in Linux Python environment that has already installed `numpy` and `pillow` packages. If running in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` in installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

1. git clone from https://github.com/TalwalkarLab/leaf/tree/master, and dive into `leaf/data/femnist/`.

2. run `./preprocess.sh -s niid --sf 1.0 -k 0 -t sample --tf 0.9 --smplseed 0 --spltseed 0`. After the processing is finished, there should be two new subfolders, namely `train` and `test`, with each containing 36 json files. The file names should be `all_data_xx_niid_0_keep_0_train_9.json` and `all_data_xx_niid_0_keep_0_test_9.json` respectively, with `xx` ranging from 0 to 35.

3. place the `train` and `test` folders according to the path variables `--femnist_train_path` and `--femnist_test_path` in `utils.py`.
