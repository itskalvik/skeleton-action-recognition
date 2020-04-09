from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy import signal
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle
import json
import os

import sys
sys.path.extend(['../'])
from data_gen.gen_joint_data import *

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
    edges = [(1, 0, 0.3), (2, 1, 0.3), (3, 2, 0.1), (4, 2, 0.1),
             (5, 4, 0.1), (6, 5, 0.1), (7, 6, 0.1), (8, 7, 0.05),
             (9, 8, 0.05), (10, 7, 0.05), (11, 2, 0.1), (12, 11, 0.1),
             (13, 12, 0.1), (14, 13, 0.1), (15, 14, 0.05), (16, 15, 0.05),
             (17, 14, 0.05), (18, 0, 0.1), (19, 18, 0.15), (20, 19, 0.12),
             (21, 20, 0.1), (22, 0, 0.1), (23, 22, 0.15),
             (24, 23, 0.12), (25, 24, 0.1), (26, 3, 0.1)]

    # read json file
    with open(filename) as f:
        json_file = json.load(f)

    # gather joint positions and timestamps of all joints
    data = []
    for frame in json_file['frames']:
        if frame['num_bodies'] > 0:
            data.append(frame['bodies'][0]['joint_positions'])
    data = np.array(data)
    data = data * 0.001 # convert data from meters to mm
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
    edges = [(0, 1, 0.3), (1, 20, 0.3), (20, 2, 0.3), (2, 3, 0.1),
            (20, 4, 0.1), (4, 5, 0.1), (5, 6, 0.1), (6, 7, 0.1),
            (7, 21, 0.05), (7, 22, 0.05), (20, 8, 0.1), (8, 9, 0.1),
            (9, 10, 0.1), (10, 11, 0.1), (11, 23, 0.05), (11, 24, 0.05),
            (0, 16, 0.1), (0, 12, 0.1), (12, 13, 0.15), (13, 14, 0.12),
            (14, 15, 0.1), (16, 17, 0.15), (17, 18, 0.12), (18, 19, 0.1)]

    data = read_xyz(filename, max_body_kinect, num_joint)
    data = np.transpose(data, (3, 1, 2, 0))
    return data, edges


'''
Smooths data with a gaussian window and upscales data frame rate with cubic
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
    data = data.reshape(T, V*C)
    f = interp1d(np.linspace(0, 1, T),
                 gaussian_filter1d(data, sigma, axis=0),
                 'cubic',
                 axis=0)
    data = f(np.linspace(0, 1, num_pad_frames*T))
    data = np.reshape(data, [-1, V, C])
    return data


'''
Calculates the backscattered RCS of an ellipsoid
with semi-axis lengths of a, b, and c.
The source code is based on Radar Systems Analysis and Design Using
MATLAB, By B. Mahafza, Chapman & Hall/CRC 2000.

Args:
    a: float, file name of json data
    b: float, file name of json data
    c: float, file name of json data
    phi: numpy array; shape-(num_frames,), phase of part for each frame
    theta: numpy array; shape-(num_frames,), angle of part for each frame

Returns:
    rcs: numpy array; shape-(num_frames,), radar backscatter of part
'''
def rcsellipsoid(a,b,c,phi,theta):
    a = np.square(a)
    b = np.square(b)
    c = np.square(c)
    rcs = (np.pi*a*b*c)/(a*(np.sin(theta)**2)*(np.cos(phi)**2) + \
                         b*(np.sin(theta)**2)*(np.sin(phi)**2) + \
                         c*(np.cos(theta)**2))**2
    return rcs


'''
Calculates the phase for each part in the skeleton.
Args:
    joint1: numpy array, joint 1 of each part
    joint2: numpy array, joint 2 of each part
    radar_loc: list, shape: (3,), location of radar
    radar_lambda: float, radar wavelength
    rangeres: float, range resolution

Returns:
    rcs: numpy array; shape-(num_frames,), radar backscatter of part
'''
def skeleton_phase(joint1, joint2, part_dia,
                   radar_loc=[0, 0, 0],
                   radar_lambda=0.001,
                   rangeres=0.01):
    radar_dist = np.abs(joint1-radar_loc)
    distances = np.linalg.norm(radar_dist, axis=-1)
    A = radar_loc-((joint1+joint2)/2)
    B = joint2-joint1
    A_dot_B = (A*B).sum(axis=-1)
    A_sum_sqrt = np.linalg.norm(A, axis=-1)
    B_sum_sqrt = np.linalg.norm(B, axis=-1)
    ThetaAngle = np.arccos(A_dot_B / ((A_sum_sqrt * B_sum_sqrt)+1e-6))
    PhiAngle = np.arcsin((radar_loc[1]-joint1[:, :, 1])/ \
                         (np.linalg.norm(radar_dist[:, :, :2], axis=-1)+1e-6))
    rcs = rcsellipsoid(part_dia,
                       part_dia,
                       np.linalg.norm(joint1-joint2, axis=-1).mean(axis=0),
                       PhiAngle,
                       ThetaAngle)
    amp = np.sqrt(rcs)
    phase = amp*np.exp(-1j*4*np.pi*distances/radar_lambda)
    indices = np.floor(distances/rangeres).astype(int)-1

    return phase, indices


'''
Calculates the backscattered RCS of a skeleton sequence
Args:
    data: numpy array; shape-(num_frames, num_joints, 3), joint position data
    edges: list of tuples, tuple shape-(3), source joint, dest joint,
                                            dia of part (modeled as ellipsoid)
    radar_loc: list, shape-(3), radar location
    radar_lambda: float, radar wavelength
    rangeres: float, range resolution
    num_pad_frames: int, number of interpolated frames to insert between real frames
Returns:
    TF; numpy array; shape-(num_range_bins, num_frames), backscattered RCS
    range_map; numpy array; range map
'''
def synthetic_spectrogram(data, edges,
                          radar_lambda=0.02,
                          rangeres=0.01,
                          radar_loc=[0,0,0],
                          num_pad_frames=25):
    src, dst, part_dia = map(list, zip(*edges))
    data = pad_frames(data, num_pad_frames)
    num_frames = data.shape[0]
    num_parts = len(edges)
    phase, indices = skeleton_phase(data[:, src],
                                    data[:, dst],
                                    part_dia,
                                    radar_loc,
                                    radar_lambda,
                                    rangeres)
    data = np.zeros((np.max(indices)+1, num_frames), dtype=np.complex64)
    for i in range(num_parts):
        data[indices[:, i], np.arange(num_frames)] += phase[:, i]

    range_map = np.flip(20*np.log10(np.abs(data)+1e-6), axis=0)
    data = np.sum(data, axis=0)

    _, _, TF = signal.stft(data,
                           window=signal.gaussian(512, std=16),
                           nperseg=512,
                           noverlap=512-16,
                           nfft=512,
                           return_onesided=False)
    TF = np.fft.fftshift(np.abs(TF), 0)
    TF = 20*np.log10(TF+1e-6)
    return TF, range_map


'''
Calculates the backscattered RCS of a skeleton sequence from ntu dataset
Args:
    data: numpy array; shape-(num_frames, num_joints, 3), joint position data
    edges: list of tuples, tuple shape-(3), source joint, dest joint,
                                            dia of part (modeled as ellipsoid)
    radar_loc: list, shape-(3), radar location
    radar_lambda: float, radar wavelength
    rangeres: float, range resolution
    num_pad_frames: int, number of interpolated frames to insert between real frames
Returns:
    TF; numpy array; shape-(num_range_bins, num_frames), backscattered RCS
    range_map; numpy array; range map
'''
def synthetic_spectrogram_ntu(data, edges,
                              radar_lambda=0.02,
                              rangeres=0.01,
                              radar_loc=[0,0,0],
                              num_pad_frames=25):
    src, dst, part_dia = map(list, zip(*edges))
    num_parts = len(edges)
    phase_data = np.zeros((600, 300*num_pad_frames), dtype=np.complex64)
    for person in data:
        if person.sum() == 0:
            continue
        person = pad_frames(person, num_pad_frames)
        phase, indices = skeleton_phase(person[:, src],
                                        person[:, dst],
                                        part_dia,
                                        radar_loc,
                                        radar_lambda,
                                        rangeres)
        for i in range(num_parts):
            phase_data[indices[:, i], np.arange(person.shape[0])] += phase[:, i]
    data = phase_data

    range_map = np.flip(20*np.log10(np.abs(data)+1e-6), axis=0)
    data = np.sum(data, axis=0)

    _, _, TF = signal.stft(data,
                           window=signal.gaussian(224, std=16),
                           nperseg=224,
                           noverlap=224-16,
                           nfft=224,
                           return_onesided=False)
    TF = np.fft.fftshift(np.abs(TF), 0)
    TF = 20*np.log10(TF+1e-6)
    return TF, range_map


def gendata(data_path, out_path, num_shards, benchmark='xview', part='eval'):
    print("Generating Spectrograms")
    with multiprocessing.Pool() as pool:
        pool.daemon = True
        spectrogram_data = list(tqdm(pool.imap(synthetic_spectrogram, sample_name),
                                total=len(sample_name)))
        pool.close()
        pool.join()

    print("Normalizing Data")
    if part == 'train':
        stat_dict[benchmark] = {}
        stat_dict[benchmark]['mean'] = np.mean(spectrogram_data)
    spectrogram_data -= stat_dict[benchmark]['mean']
    if part == 'train':
        stat_dict[benchmark]['std'] = np.std(spectrogram_data)
    spectrogram_data /= stat_dict[benchmark]['std']

    tfrecord_data_path = os.path.join(out_path,
                                      part+"_data_spectrograms",
                                      part+"_data_spectrograms_{}.tfrecord")

    if not os.path.exists(os.path.join(out_path, part+"_data_spectrograms")):
        os.makedirs(os.path.join(out_path, part+"_data_spectrograms"))

    shard = 0
    writer = None
    print("Generating TFRecords")
    for i in tqdm(range(len(sample_label))):
        if i % (len(sample_label)//num_shards) == 0:
            writer = tf.io.TFRecordWriter(tfrecord_data_path.format(shard))
            shard += 1
        writer.write(serialize_example(spectrogram_data[i], sample_label[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--out_folder', default='../data/ntu/')
    parser.add_argument('--num-shards',
                        type=int,
                        default=40,
                        help='number of files to split dataset into')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    stat_dict = dict()
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
                arg.num_shards,
                benchmark=b,
                part=p)

    f = open(os.path.join(arg.out_folder, "normalization_stats.pkl"),"wb")
    pickle.dump(stat_dict,f)
    f.close()
    print("\nNormalization Stats")
    print(stat_dict)
