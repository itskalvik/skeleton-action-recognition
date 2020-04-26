import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import argparse
import inspect
import shutil
import os

from utils import *

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
        '--num-filters', type=int, default=64, help='number of base filters in model')
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
        '--model-type',
        default="resnet",
        help='model to train')
    parser.add_argument(
        '--lr_cycle',
        type=int,
        default=10,
        help='number of epochs for the cyclic LR cycle')
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


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()
    arg.model_type = 'model.'+arg.model_type.strip()+'.Model'

    run_params = dict(vars(arg))
    del run_params['data_path']
    del run_params['label_path']
    del run_params['log_dir']
    if arg.lambda_train_epoch > arg.num_epochs:
        del run_params['lambda_train_epoch']
    if arg.loc_train_epoch > arg.num_epochs:
        del run_params['loc_train_epoch']
    sorted(run_params)

    run_params   = str(run_params).replace(" ", "").replace("'", "").replace(",", "-")[1:-1]
    if len(arg.notes) > 0:
        run_params += "-" + arg.notes
    arg.log_dir  = os.path.join(arg.log_dir, run_params)

    #copy hyperparameters and model definition to log folder
    save_arg(arg)
    Model = import_class(arg.model_type)
    shutil.copy2(inspect.getfile(Model), arg.log_dir)
    shutil.copy2(os.path.abspath(__file__), arg.log_dir)

    numpy_datasets = {x: Dataset(data_path=arg.data_path.format(x),
                                 label_path=arg.label_path.format(x)) \
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(numpy_datasets[x],
                                                  batch_size=arg.batch_size,
                                                  shuffle=True,
                                                  num_workers=10) \
                   for x in ['train', 'val']}

    writer = SummaryWriter(log_dir=arg.log_dir)
    model = Model(num_classes=arg.num_classes, num_filters=arg.num_filters)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.base_lr)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                     base_lr=1e-4,
                                                     max_lr=arg.base_lr,
                                                     step_size_up=arg.lr_cycle,
                                                     cycle_momentum=False)

    # add graph to tb
    writer.add_graph(model, numpy_datasets['train'][0][0].unsqueeze(0))
    writer.close()

    # assign available gpus to model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda")
    model.to(device)

    #start training
    for epoch in range(arg.num_epochs):
        print('Epoch {}/{}'.format(epoch+1, arg.num_epochs))
        print('-' * 10)

        if epoch > arg.lambda_train_epoch:
            for key, value in model.named_parameters():
                if 'radar_lambda' in key:
                    value.requires_grad = True

        if epoch > arg.loc_train_epoch:
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
