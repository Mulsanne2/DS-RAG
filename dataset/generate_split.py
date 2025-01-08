import os
import numpy as np
from sklearn.model_selection import train_test_split


def generate_split(num_nodes, path):

    # Split the dataset into train, val, and test sets
    indices = np.arange(num_nodes)
    # np.random.shuffle(indices)
    print("# total samples: ", len(indices))

    # Create a folder for the split
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/indices.txt', 'w') as file:
        file.write('\n'.join(map(str, indices)))