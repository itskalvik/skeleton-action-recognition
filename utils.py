from data_gen.gen_joint_data import *
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix
import io
import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import PIL
import torch
import yaml
matplotlib.use('Agg')


'''
Extract joint data from Microsoft Azure Kinect pose data
Args:
    filename: str, file name of json data
Returns:
    data: numpy array; shape-(num_frames, num_joints, 3), joint position data
    edges: list of tuples, tuple shape-(3), source joint, dest joint,
                                            dia of part (modeled as ellipsoid)
'''
def preprocess_azure_kinect(filename):
    # define edges of the skeleton graph
    edges = [(1, 0), (2, 1), (3, 2), (4, 2),
             (5, 4), (6, 5), (7, 6), (8, 7),
             (9, 8), (10, 7), (11, 2), (12, 11),
             (13, 12), (14, 13), (15, 14), (16, 15),
             (17, 14), (18, 0), (19, 18), (20, 19),
             (21, 20), (22, 0), (23, 22),
             (24, 23), (25, 24), (26, 3)]

    # read json file
    with open(filename) as f:
        json_file = json.load(f)

    # gather joint positions and timestamps of all joints
    data = []
    for frame in json_file['frames']:
        if frame['num_bodies'] > 0:
            data.append(frame['bodies'][0]['joint_positions'])
    data = np.array(data)
    data = data * 0.001 # convert data from meters to 
    return data, edges

'''
Extract joint data from NTU pose data
Args:
    filename: str, file name of json data
Returns:
    data: numpy array; shape-(num_frames, num_joints, 3), joint position data
    edges: list of tuples, tuple shape-(3), source joint, dest joint,
                                            dia of part (modeled as ellipsoid)
'''
def preprocess_ntu(filename):
    edges = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7),
            (7, 21), (7, 22), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23),
            (11, 24), (0, 16), (0, 12), (12, 13), (13, 14), (14, 15), (16, 17),
            (17, 18), (18, 19)]

    data = read_xyz(filename, max_body_kinect, num_joint)
    data = np.transpose(data, (3, 1, 2, 0))
    return data, edges


'''
Smooths data with a gaussian window and upsamples data frame rate with cubic
interpolation
Args:
    data: numpy array; shape-(num_frames, num_joints, 3), joint data
    num_pad_frames: int, number of interpolated frames to insert between real frames
    sigma: int, sigma value for gaussian smoothing
Returns:
    data: numpy array; shape-(num_frames, num_joints, 3), padded data
'''
def pad_frames(data, num_pad_frames=1, sigma=3):
  T, V, C = data.shape
  f = interp1d(np.linspace(0, 1, T),
              gaussian_filter1d(data, sigma, axis=1),
              'cubic',
              axis=-3)
  data = f(np.linspace(0, 1, num_pad_frames * T))
  return data


class Dataset(torch.utils.data.Dataset):
    """
    pytorch Dataset object to fetch and serve data samples from NTU Dataset.
    The method also increaces data frame rate with interpolation.

    Args:
        data_path: str, path to npy file containing the NTU Dataset
        label_path: str, path to pickle file containing NTU Dataset labels
        num_pad_frames: int, number of interpolated frames to insert between
            consecutive real frames in each data sample
        sigma: int, sigma of gaussian filter used to smooth the data before
            interpolating extra frames
    """
    def __init__(self, data_path, label_path, num_pad_frames=250, sigma=3):
        self.sigma = sigma
        self.num_pad_frames = num_pad_frames

        label_path = Path(label_path)
        if not (label_path.exists()):
            print('Label file does not exist')

        data_path = Path(data_path)
        if not (data_path.exists()):
            print('Data file does not exist')

        with open(label_path, 'rb') as f:
            _, labels = pickle.load(f, encoding='latin1')

        self.data = np.load(data_path, allow_pickle=True, mmap_mode='r')
        self.labels = np.array(labels)

        self.T = self.data.shape[-3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        X = torch.from_numpy(self.pad_frames(X))
        y = torch.as_tensor(self.labels[index])
        return X.type(torch.FloatTensor), y

    def pad_frames(self, data):
        f = interp1d(np.linspace(0, 1, self.T),
                     gaussian_filter1d(data, self.sigma, axis=-3),
                     'cubic',
                     axis=-3)
        data = f(np.linspace(0, 1, self.num_pad_frames * self.T))
        return data


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_confusion_matrix(y_true, y_pred):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.
  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    cm = confusion_matrix(y_true, y_pred)

    figure = plt.figure(figsize=(25, 25))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Confusion matrix")
    tick_marks = np.arange(60)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                   decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = np.asarray(image)
    return image


def save_arg(arg):
    arg_dict = vars(arg)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    with open(os.path.join(arg.log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)
