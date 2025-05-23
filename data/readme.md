# Instructions on data preprocessing

We conducted experiments using four datasets: FEMNIST, CelebA, Shakespeare, and CIFAR-100 (the OpenImage and CIFAR-10 tasks were deprecated). The first three datasets can be obtained from [LEAF](https://leaf.cmu.edu/) together with bash code for reproducible data partition and train-test split. The CIFAR-100 dataset was downloaded and partitioned with [FL-bench](https://github.com/KarhouTam/FL-bench/tree/master).

We always did sample-wise data split, which means we have a proportion of held-out samples per client rather than a held-out set of clients. This corresponds to the argument `-t sample` in bash code for LEAF and `-sp sample` for FL-bench. The main reason for conducting such data split is that we emulated sparse attendance, which means only 5% clients participate in each round while the other 95% clients don't update their models in that round. In such a case model validation involving those 95% clients is meaningless. The train-test split rates were 90%-10% for all tasks. This corresponds to the argument `--tf 0.9` in bash code for LEAF and  `-tr 0.1` for FL-bench.

Please dive into the respective subdirectory for further instructions on each dataset.

Once data processing is done, please place the data according to the paths in `utils.py`.
