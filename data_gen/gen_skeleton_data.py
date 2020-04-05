from modules.input_reader import VideoReader
from modules.parse_poses import parse_poses
from modules.inference_engine_pytorch import InferenceEnginePyTorch

import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

import numpy as np
import os
import cv2

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body_true = 2
num_joint = 19
max_frame = 300

R = np.array([[ 0.9261, -0.0471, -0.374 ],
              [-0.0258,  0.9818, -0.1879],
              [ 0.3761,  0.1837,  0.9081]])
t = np.array([[ -46.8],[8.],[-343.2]])

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def read_xyz(file, input_scale=0.237037037037037, fx=1669.54, stride=8):
    data = np.zeros((max_frame, max_body_true, num_joint, 3), dtype=np.float32)
    frame_provider = VideoReader(file)
    for i, frame in enumerate(frame_provider):
        if frame is None:
            break

        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:448]
        inference_result = net.infer(scaled_img)
        poses_3d, _ = parse_poses(inference_result, input_scale, stride, fx, True)
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], num_joint, -1)[:, :, 0:4]
            person_idx = np.argsort(np.count_nonzero(poses_3d[:, :, 3]==-1, axis=1))
            poses_3d = poses_3d[person_idx[:2]]

            for j, pose in enumerate(poses_3d):
                data[i, j] = pose[:, :3]

    data = data.transpose(3, 0, 2, 1)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s))
        fp[i, :, 0:data.shape[1], :, :] = data

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/ntu/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()
    net = InferenceEnginePyTorch("models/human-pose-estimation-3d.pth", args.device)

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
