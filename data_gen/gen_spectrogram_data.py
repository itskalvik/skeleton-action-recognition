from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from skimage.transform import resize
from scipy import signal
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle
import os

import sys
sys.path.extend(['../'])
from data_gen.gen_joint_data import *

edges = [(0, 1), (1, 20), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7),
        (7, 21), (7, 22), (20, 8), (8, 9),
        (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 16), (0, 12), (12, 13), (13, 14),
        (14, 15), (16, 17), (17, 18), (18, 19)]

src, dst = map(list, zip(*edges))
num_parts = len(edges)

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
def skeleton_phase(joint1, joint2,
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
    rcs = rcsellipsoid(1.0,
                       1.0,
                       np.linalg.norm(joint1-joint2, axis=-1).mean(axis=0),
                       PhiAngle,
                       ThetaAngle)
    amp = np.sqrt(rcs)
    phase = amp*np.exp(-1j*4*np.pi*distances/radar_lambda)
    return phase


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
def synthetic_spectrogram_ntu(data,
                              radar_lambda=4e-4,
                              rangeres=0.01,
                              radar_loc=[0,0,0],
                              num_pad_frames=500,
                              image_size=224):
    data = np.transpose(data, (3, 1, 2, 0))
    phase_data = np.zeros((300*num_pad_frames), dtype=np.complex64)
    for person in data:
        if person.sum() == 0:
            continue
        person = pad_frames(person, num_pad_frames)
        phase = skeleton_phase(person[:, src],
                               person[:, dst],
                               radar_loc,
                               radar_lambda,
                               rangeres)
        phase_data[:phase.shape[0]] += phase.sum(axis=-1)
    _, _, TF = signal.stft(phase_data,
                           window=signal.gaussian(224, std=16),
                           nperseg=224,
                           noverlap=224-16,
                           nfft=224,
                           return_onesided=False)
    TF = np.fft.fftshift(np.abs(TF), 0)
    TF = 20*np.log(TF+1e-6)
    TF = resize(TF, (image_size, image_size)).astype(np.float32)
    return TF


def gendata(data_path, part):
    data = np.load(data_path)

    print("Generating Spectrograms")
    with multiprocessing.Pool() as pool:
        spectrogram_data = list(tqdm(pool.imap(synthetic_spectrogram_ntu, data),
                                total=len(data)))
    del data

    print("Normalizing Data")
    if part == 'train':
        stat_dict['mean'] = np.mean(spectrogram_data)
    spectrogram_data -= stat_dict['mean']
    if part == 'train':
        stat_dict['std'] = np.std(spectrogram_data)
    spectrogram_data /= stat_dict['std']

    np.save(data_path[:-4]+'_spectrograms.npy', spectrogram_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/ntu/xview/{}_data_joint.npy')

    stat_dict = dict()
    arg = parser.parse_args()

    for part in ['train', 'val']:
        gendata(arg.data_path.format(part), part)
