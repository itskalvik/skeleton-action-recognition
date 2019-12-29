import argparse
import os
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

paris = {
    'xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    )
}

datasets = ['xsub', 'xview']
sets = ['train', 'val']

def gendata(datapath):
    for dataset in datasets:
        for set in sets:
            print(dataset, set)
            data = np.load(os.path.join(datapath, '{}/{}_data_joint.npy'.format(dataset, set)))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                os.path.join(datapath, '{}/{}_data_bone.npy'.format(dataset, set)),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))

            fp_sp[:, :C, :, :, :] = data
            for v1, v2 in tqdm(paris[dataset]):
                v1 -= 1
                v2 -= 1
                fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Joint to Bone Converter.')
    parser.add_argument('--data_path', default='../data/ntu/')
    arg = parser.parse_args()
    gendata(arg.data_path)
