import os
import io
import PIL
import yaml
import itertools
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from pathlib import Path
import pickle
import torch


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
