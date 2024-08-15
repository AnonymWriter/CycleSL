# find train/ test/ validation/ -maxdepth 1 -type f -print0 | xargs -0 mv -t all_samples/

import json
import random
import pandas as pd

def load_csv(clients: dict[str, list], prefix: str, csv_path: str) -> None:
    """
    Read csv file and add samples to clients.

    Arguments:
        clients (dict[str, list]): client data dictionary.
        prefix (str): prefix for client id.
        csv_path (str): path to csv file.
    """
    
    df = pd.read_csv(csv_path)
    
    for c, p, l in zip(df['client_id'], df['sample_path'], df['label_id']):
        k = prefix + str(c)
        if k not in clients:
            clients[k] = []

        clients[k].append((p, l))


clients = {}

# collect all samples
load_csv(clients, 'train', 'client_data_mapping/train_20.csv')
load_csv(clients, 'test' , 'client_data_mapping/test_20.csv')
load_csv(clients, 'val'  , 'client_data_mapping/validation_20.csv')
print('number of all clients:', len(clients))

# user-dependent train-test-split, ratio 90% - 10%
train_dict = {}
test_dict = {}
for c, samples in clients.items():
    # discard clients who have only 1 sample - cannot do train-test-split for such clients
    if len(samples) <= 1:
        continue

    random.seed(0) # reproducibility
    random.shuffle(samples)
    
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    test_samples  = samples[split:]

    train_dict[c] = train_samples
    test_dict [c] = test_samples

print('number of clients with more than 1 sample:', len(train_dict))
print('number of training samples:', sum([len(v) for v in train_dict.values()]))
print('number of test samples:', sum([len(v) for v in test_dict.values()]))

with open("train_path.json", "w") as outfile: 
    json.dump(train_dict, outfile)

with open("test_path.json", "w") as outfile: 
    json.dump(test_dict, outfile)