# Data preprocessing for CelebA dataset

## 📋 Source
https://github.com/TalwalkarLab/leaf/tree/master/data/celeba

## 🗺 Steps

Please go through the following steps in Linux Python environment that has already installed `numpy` and `pillow` packages. If running in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` in installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

1. git clone from https://github.com/TalwalkarLab/leaf/tree/master, and dive into `leaf/data/celeba/`. Create here a new subfolder `data/raw/`.

2. from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, download the image zip file `img_align_celeba.zip` and the metadata files `identity_CelebA.txt` and `list_attr_celeba.txt`, and place them in `data/raw/` directory. Unzip the images into a subfolder `img_align_celeba`.

3. run `./preprocess.sh -s niid --sf 1.0 -k 0 -t sample --tf 0.9 --smplseed 0 --spltseed 0`. After the processing is finished, there should be two new folders, namely `train` and `test`, containing `all_data_niid_0_keep_0_train_9.json` and `all_data_niid_0_keep_0_test_9.json` respectively.

4. place the json files and the image folder according to the paths `--celeba_train_path`, `--celeba_test_path` and `--celeba_image_path` in `utils.py`.