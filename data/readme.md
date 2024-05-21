# Instructions on data preprocessing

We conducted experiments using three datasets: FEMNIST, CelebA, and Shakespeare. All datasets can be obtained from https://leaf.cmu.edu/ together with bash code for reproducible data split.

We always did user-dependent data split, which means we have a proportion of held-out samples per client rather than a held-out set of clients. This corresponds to `-t sample` argument in bash code. The main reason for conducting such data split is that we emulated sparse attendance, which means only 5% clients participate in each round while the other 95% clients don't update their models in that round. In such a case model validation involving those 95% clients is meaningless.

The train-test split rate is 90%-10%. This corresponds to `--tf 0.9` argument in bash code.

Please dive into the respective subdirectory for further instructions on each dataset.

Once data processing is done, please place the data according to the paths in `utils.py`.
