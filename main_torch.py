import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import yaml
import os

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import io
import PIL
import itertools
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_parser():
    parser = argparse.ArgumentParser(
        description='Skeleton-Based Action Recognition')
    parser.add_argument(
        '--base-lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument(
        '--num-classes', type=int, default=60, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument(
        '--num-epochs', type=int, default=80, help='total epochs to train')
    parser.add_argument(
        '--log-dir',
        default="logs/",
        help='folder to store model-definition/training-logs/hyperparameters')
    parser.add_argument(
        '--data-path',
        default="data/ntu/xview/{}_data_joint.npy",
        help='path to data files')
    parser.add_argument(
        '--label-path',
        default="data/ntu/xview/{}_label.pkl",
        help='path to label files')
    parser.add_argument(
        '--notes',
        default="",
        help='run details')
    parser.add_argument(
        '--cos-anneal-epochs',
        type=int,
        default=10,
        help='number of epochs for the Cosine Annealing LR cycle')
    parser.add_argument(
        '--lambda-train-epoch',
        type=int,
        default=1000,
        help='epoch to training the radar_lambda')
    parser.add_argument(
        '--loc-train-epoch',
        type=int,
        default=1000,
        help='epoch to training the radar_loc')
    return parser


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
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path,
                 num_pad_frames=250,
                 sigma=3):
        label_path = Path(label_path)
        if not (label_path.exists()):
            print('Label file does not exist')

        data_path = Path(data_path)
        if not (data_path.exists()):
            print('Data file does not exist')

        with open(label_path, 'rb') as f:
            _, labels = pickle.load(f, encoding='latin1')

        self.data = np.load(data_path,
                            allow_pickle=True,
                            mmap_mode='r')
        print(self.data.shape)
        self.labels = np.array(labels)

        self.T = self.data.shape[-3]
        self.sigma = sigma
        self.num_pad_frames = num_pad_frames

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
        data = f(np.linspace(0, 1, self.num_pad_frames*self.T))
        return data


class Spectrogram(torch.nn.Module):
    def __init__(self, radar_lambda=1e-3, radar_loc=[0.,0.,0.], image_size=256):
        super().__init__()
        self.radar_lambda = torch.nn.Parameter(torch.as_tensor(radar_lambda),
                                               requires_grad=False)
        self.radar_loc = torch.nn.Parameter(torch.as_tensor(radar_loc),
                                            requires_grad=False)
        self.window = torch.nn.Parameter(torch.hann_window(256),
                                         requires_grad=False)
        self.image_size = image_size

        edges = [(0, 1), (1, 20), (20, 2), (2, 3),
                (20, 4), (4, 5), (5, 6), (6, 7),
                (7, 21), (7, 22), (20, 8), (8, 9),
                (9, 10), (10, 11), (11, 23), (11, 24),
                (0, 16), (0, 12), (12, 13), (13, 14),
                (14, 15), (16, 17), (17, 18), (18, 19)]
        self.src, self.dst = map(list, zip(*edges))

    def forward(self, x):
        joint1 = x[:, :, :, self.src]
        joint2 = x[:, :, :, self.dst]
        radar_dist = torch.abs(joint1-self.radar_loc.reshape(-1, 1, 1, 1))
        distances = torch.norm(radar_dist, dim=1)
        A = self.radar_loc.reshape(-1, 1, 1, 1)-((joint1+joint2)/2)
        B = joint2-joint1
        A_dot_B = torch.sum(A*B, dim=1)
        A_sum_sqrt = torch.norm(A, dim=1)
        B_sum_sqrt = torch.norm(B, dim=1)
        ThetaAngle = torch.acos(A_dot_B / ((A_sum_sqrt * B_sum_sqrt)+1e-6))
        PhiAngle = torch.asin((self.radar_loc[1]-joint1[:, 1])/
                              (torch.norm(radar_dist[:, :2], dim=1)+1e-6))

        c = torch.mean(torch.norm(joint1-joint2, dim=1), dim=2, keepdim=True)
        c = torch.pow(c, 2)
        rcs = (np.pi*c)/((torch.sin(ThetaAngle)**2)*(torch.cos(PhiAngle)**2) + \
                         (torch.sin(ThetaAngle)**2)*(torch.sin(PhiAngle)**2) + \
                       c*(torch.cos(ThetaAngle)**2))**2

        amp = torch.sqrt(rcs)
        theta = -1*4*np.pi*distances/self.radar_lambda
        phase_data = torch.stack((amp*torch.cos(theta),
                                  amp*torch.sin(theta)),
                                 dim=4)
        phase_data = torch.sum(phase_data, dim=[-2, -3])
        TF1 = torch.stft(phase_data[..., 0],
                          n_fft=256,
                          hop_length=16,
                          win_length=256,
                          window=self.window,
                          center=True,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=False)
        TF2 = torch.stft(phase_data[..., 1],
                          n_fft=256,
                          hop_length=16,
                          win_length=256,
                          window=self.window,
                          center=True,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=False)
        TF = TF1+TF2
        TF = torch.norm(TF, dim=-1)
        TF = torch.log(TF+1e-6)
        TF = torch.roll(TF, 128, dims=1)
        TF = TF.unsqueeze(dim=1)
        TF = torch.nn.functional.interpolate(TF, self.image_size)
        return TF


def resnet18(num_classes=60, image_size=256):
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64,
                                  kernel_size=7,
                                  stride=2,
                                  padding=3,
                                  bias=False)
    torch.nn.init.kaiming_normal_(model.conv1.weight,
                                  mode='fan_out',
                                  nonlinearity='relu')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = torch.nn.Sequential(Spectrogram(image_size=image_size), model)
    return model


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()

    base_lr            = arg.base_lr
    num_classes        = arg.num_classes
    num_epochs         = arg.num_epochs
    log_dir            = arg.log_dir
    data_path          = arg.data_path
    label_path         = arg.label_path
    batch_size         = arg.batch_size
    notes              = arg.notes
    lambda_train_epoch = arg.lambda_train_epoch
    loc_train_epoch    = arg.loc_train_epoch
    cos_anneal_epochs  = arg.cos_anneal_epochs

    run_params = dict(vars(arg))
    del run_params['data_path']
    del run_params['label_path']
    del run_params['log_dir']
    if lambda_train_epoch > num_epochs:
        del run_params['lambda_train_epoch']
    if loc_train_epoch > num_epochs:
        del run_params['loc_train_epoch']
    sorted(run_params)

    run_params   = str(run_params).replace(" ", "").replace("'", "").replace(",", "-")[1:-1]
    if not notes:
        run_params   += "-" + notes
    log_dir      = os.path.join(arg.log_dir, run_params)
    arg.log_dir  = log_dir

    #copy hyperparameters and model definition to log folder
    save_arg(arg)

    numpy_datasets = {x: Dataset(data_path=data_path.format(x),
                                 label_path=label_path.format(x)) \
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(numpy_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=16) \
                   for x in ['train', 'val']}

    writer = SummaryWriter(log_dir=log_dir)
    model = resnet18(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=len(dataloaders['train'])*cos_anneal_epochs,
                                                              eta_min=1e-4)
    lr_scheduler.last_epoch = num_epochs-1

    # add graph to tb
    writer.add_graph(model, numpy_datasets['train'][0][0].unsqueeze(0))
    writer.close()

    # assign available gpus to model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda")
    model.to(device)

    #start training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        if epoch > lambda_train_epoch:
            for key, value in model.named_parameters():
                if 'radar_lambda' in key:
                    value.requires_grad = True

        if epoch > loc_train_epoch:
            for key, value in model.named_parameters():
                if 'radar_loc' in key:
                    value.requires_grad = True

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            val_preds = []
            for i, data in enumerate(tqdm(dataloaders[phase])):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        val_preds.extend(preds.data.cpu().numpy())

                running_loss += loss.item()
                running_corrects += torch.sum(preds==labels.data)
                writer.add_scalar('{}_cross_entropy_loss'.format(phase),
                                  loss.item(),
                                  epoch*len(dataloaders[phase])+i)
                writer.add_scalar('{}_acc'.format(phase),
                                  torch.sum(preds==labels.data).double()/inputs.size(0),
                                  epoch*len(dataloaders[phase])+i)

            if phase=='val':
                conf_mat = get_confusion_matrix(dataloaders[phase].dataset.labels,
                                                val_preds)
                writer.add_image('confusion_matrix',
                                 conf_mat,
                                 epoch,
                                 dataformats='HWC')
                writer.close()

            epoch_loss = running_loss/len(dataloaders[phase])
            epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)
            writer.add_scalar('{}_epoch_cross_entropy_loss'.format(phase),
                              epoch_loss,
                              epoch)
            writer.add_scalar('{}_epoch_acc'.format(phase),
                              epoch_acc,
                              epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        lr_scheduler.step()
