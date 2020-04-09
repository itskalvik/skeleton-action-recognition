import sys
sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

from scipy import signal
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import cv2
import os

sys.path.extend(['./pose_est_3d'])
from modules.input_reader import VideoReader
from modules.parse_poses import parse_poses
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from pose_tracker import pose_tracker

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                     31, 34, 35, 38]
training_cameras = [2, 3]
max_body_true = 2
num_joint = 19
max_frame = 300

'''
Rotation matrix and transformation matrix from calibration results for
Microsoft Kinect V2.

Villena-Martinez, Victor et al. “A Quantitative Comparison of Calibration
Methods for RGB-D Sensors Using Different Technologies.” Sensors (2017).
'''
R = np.array([[ 0.9261, -0.0471, -0.374 ],
              [-0.0258,  0.9818, -0.1879],
              [ 0.3761,  0.1837,  0.9081]])
R_inv = np.linalg.inv(R)
t = np.array([[ -46.8],[8.],[-343.2]])

# deep learning model to extract pose from rgb images
net = InferenceEnginePyTorch("pose_est_3d/models/human-pose-estimation-3d.pth",
                              "GPU")

'''
Transform given pose data with rotation matrix R and transformation
matrix T.
Args:
    poses_3d: pose data
    R: rotation matrix
    t: transformation matrix
Returns:
    data: numpy array; transformed pose matrix
'''
def rotate_poses(poses_3d, R, t):
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d


'''
Extract pose data from video data and preprocess it
Args:
    filename: path to video file from which, pose data is extracted
    input_scale: scale factor along the horizontal/vertical axis
Returns:
    data: numpy array; transformed pose matrix
'''
def read_xyz(filename, input_scale=0.2370, fx=1669.54, stride=8, batch_size=32):
    tracker = pose_tracker(data_frame=max_frame, num_joint=num_joint)

    total_frames = []
    frame_provider = VideoReader(filename)
    for i, frame in enumerate(frame_provider):
        if frame is None:
            break
        scaled_img = cv2.resize(frame,
                                dsize=None,
                                fx=input_scale,
                                fy=input_scale)
        scaled_img = scaled_img[:, 0:448]
        total_frames.append(scaled_img)
    total_frames = np.array(total_frames)
    total_frames = np.array_split(total_frames, (len(total_frames)//batch_size)+1)

    features = []
    heatmaps = []
    pafs = []
    for batch_frames in total_frames:
        inference_result = net.infer_batch(batch_frames)
        features.extend(inference_result[0])
        heatmaps.extend(inference_result[1])
        pafs.extend(inference_result[2])
    features = np.array(features)
    heatmaps = np.array(heatmaps)
    pafs = np.array(pafs)

    for i in range(len(features)):
        poses_3d, _ = parse_poses([features[i], heatmaps[i], pafs[i]],
                                  input_scale,
                                  stride,
                                  fx,
                                  True)
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d = poses_3d.reshape(poses_3d.shape[0], num_joint, -1)
            person_idx = np.argsort(np.count_nonzero(poses_3d[:, :, 3]==-1,
                                                     axis=1))
            poses_3d = poses_3d[person_idx[:max_body_true]]
            tracker.update(poses_3d[:, :, :3], i+1)
        data = tracker.get_skeleton_sequence()*0.01
        data = pre_normalization(np.expand_dims(data, axis=0),
                                 zaxis=[2, 0],
                                 xaxis=[9, 3],
                                 center=2)[0]

    if data.shape[-1] == 1:
        tmp = np.zeros((3, max_frame, num_joint, max_body_true), dtype=np.float32)
        tmp[:, :, :, 0:1] = data
        data = tmp

    return data

'''
Extract pose data from video data and save labels/data to disk
Args:
    data_path: path to folder in which NTU dataset's rgb data is in
    out_path: path to folder in which data and labels are stored in
    benchmark: str, xview/xsub data partition
    part: str, eval/train data partition
    shuffle: bool, enable/disable data shuffling
'''
def gendata(data_path, out_path, benchmark='xview', part='eval', shuffle=False):
    sample_name = []
    sample_label = []
    print("Getting Filenames and Labels")
    for filename in tqdm(os.listdir(data_path)):
        if not filename.endswith('avi'):
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
            sample_name.append(os.path.join(data_path, filename))
            sample_label.append(action_class - 1)

    if shuffle:
        p = np.random.permutation(len(sample_label))
        sample_label = np.array(sample_label)[p]
        sample_name = np.array(sample_name)[p]

    with open('{}/{}_label_rgb.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    print("Extracting Pose Data")
    data = np.zeros((len(sample_name), 3, max_frame, num_joint, max_body_true),
                    dtype=np.float32)
    for i, file in enumerate(tqdm(sample_name)):
        data[i] = read_xyz(file)
    np.save('{}/{}_data_joint_rgb.npy'.format(out_path, part), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_rgb/')
    parser.add_argument('--out_folder', default='../data/ntu/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print("\n", b, p)
            gendata(
                arg.data_path,
                out_path,
                benchmark=b,
                part=p,
                shuffle=True if 'train' in p else False)
