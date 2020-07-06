import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(features, label):
    feature = {
        'features':
        _bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
        'label':
        _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(
        feature=feature)).SerializeToString()


def gen_tfrecord_data(num_shards, label_path, data_path, shuffle):
    dest_folder = data_path[:-4]
    label_path = Path(label_path)
    if not (label_path.exists()):
        print('Label file does not exist')
        return

    data_path = Path(data_path)
    if not (data_path.exists()):
        print('Data file does not exist')
        return

    try:
        with open(label_path) as f:
            _, labels = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            _, labels = pickle.load(f, encoding='latin1')

    # Datashape: Total_samples, 3, 300, 25, 2
    data = np.load(data_path, allow_pickle=True, mmap_mode='r')
    labels = np.array(labels)

    if len(labels) != len(data):
        print("Data and label lengths didn't match!")
        print("Data size: {} | Label Size: {}".format(data.shape,
                                                      labels.shape))
        return -1

    print("Data shape:", data.shape)
    if shuffle:
        p = np.random.permutation(len(labels))
        labels = labels[p]
        data = data[p]

    dest_folder = Path(dest_folder)
    if not (dest_folder.exists()):
        os.mkdir(dest_folder)

    tfrecord_data_path = os.path.join(
        dest_folder,
        data_path.name.split(".")[0] + "-{}.tfrecord")
    shard = 0
    writer = None
    for i in tqdm(range(len(labels))):
        if i % (len(labels) // num_shards) == 0:
            writer = tf.io.TFRecordWriter(tfrecord_data_path.format(shard))
            shard += 1
        writer.write(serialize_example(data[i], labels[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NTU-RGB-D Data TFRecord Converter')
    parser.add_argument('--num-shards',
                        type=int,
                        default=40,
                        help='number of files to split dataset into')
    parser.add_argument('--data-path',
                        default='../data/ntu/xview/{}_data_joint.npy',
                        help='path to npy file with data')
    parser.add_argument('--label-path',
                        default='../data/ntu/xview/{}_label.pkl',
                        help='path to pkl file with labels')
    arg = parser.parse_args()

    for part in ['train', 'val']:
        if 'train' in part:
            shuffle = True
        else:
            shuffle = False
        gen_tfrecord_data(arg.num_shards, arg.label_path.format(part),
                          arg.data_path.format(part), shuffle)
