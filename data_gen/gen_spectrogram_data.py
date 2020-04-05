import sys
sys.path.extend(['../'])
from data_gen.gen_joint_data import *
from data_gen.gen_tfrecord_data import *

from scipy import signal
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# define edges of the skeleton graph
edges = [(0, 1, 0.3), (1, 20, 0.3), (20, 2, 0.3), (2, 3, 0.1),
        (20, 4, 0.1), (4, 5, 0.1), (5, 6, 0.1), (6, 7, 0.1),
        (7, 21, 0.05), (7, 22, 0.05), (20, 8, 0.1), (8, 9, 0.1),
        (9, 10, 0.1), (10, 11, 0.1), (11, 23, 0.05), (11, 24, 0.05),
        (0, 16, 0.1), (0, 12, 0.1), (12, 13, 0.15), (13, 14, 0.12),
        (14, 15, 0.1), (16, 17, 0.15), (17, 18, 0.12), (18, 19, 0.1)]
radar_lambda=0.02
rangeres=0.01
radar_loc=[0,0,0]
pad_frames=15


# function to apply moving average on n dim data
# along first axis
def moving_average(a, n=3) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


'''
Function to preprocess data from microsoft kinect azure
to generate spectrograms from virtual radar

Args:
    filename: str, file name of json data
    pad_frames: int, number of interpolated frames to insert
                between real frames
Returns:
    data: numpy array; shape-(num_frames, num_joints, 3), joint position data
    edges: list of tuples, tuple shape-(4), vertices in edge, dia of part, length of
           part (modeled as ellipsoid)
    total_time: float, total time of data sample in secs
'''
def preprocess_ntu(data, pad_frames=1, window_len=15):
    # denoise data with moving average
    data = data.reshape(-1, num_joint*3)
    num_frames = data.shape[0]-window_len+1

    # linear interpolation to fill in the extra/pad frames
    tmp_data = []
    for i in range(num_joint*3):
        tmp = moving_average(data[:, i], window_len)
        tmp = np.interp(np.arange(0, pad_frames*num_frames),
                        np.arange(0, pad_frames*num_frames, pad_frames),
                        tmp)
        tmp_data.append(tmp)
    data = np.array(tmp_data).T
    data = np.reshape(data, [-1, num_joint, 3])

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
Calculates the backscattered RCS of a skeleton sequence

Args:
    joint1: numpy array; shape-(num_frames, 3), joint to calc backscattes
    joint2: numpy array; shape-(num_frames, 3), joint that belongs to part of interest
    edges: list of tuples, tuple shape-(4), vertices in edge, dia of part, length of
           part (modeled as ellipsoid)
    radar_loc: list, shape-(3), radar location
    radar_lambda: float, radar wavelength
    rangeres: float, range resolution

Returns:
    TF; numpy array; shape-(512, 283), spectrogram
'''
def synthetic_spectrogram(filename):
    src, dst, part_dia = map(list, zip(*edges))
    joint_data = read_xyz(filename, max_body_kinect, num_joint)
    joint_data = np.transpose(joint_data, (3, 1, 2, 0))
    data = np.zeros((600, max_frame*pad_frames), dtype=np.complex64)

    for i in range(2):
        person = preprocess_ntu(joint_data[i], pad_frames)
        if person.sum() == 0:
            continue

        joint1 = person[:, src]
        joint2 = person[:, dst]
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

        for i in range(joint1.shape[1]):
            data[indices[:, i], np.arange(joint1.shape[0])] += phase[:, i]

    range_map = np.flip(20*np.log10(np.abs(data)+1e-6), axis=0)
    data = np.sum(data, axis=0)

    _, _, TF = signal.stft(data,
                           window=signal.gaussian(224, std=16),
                           nperseg=224,
                           noverlap=224-20,
                           nfft=224,
                           return_onesided=False)

    TF = np.roll(TF, TF.shape[0]//2, axis=0)
    TF = np.flip(TF, axis=0)
    TF = 20*np.log10(np.abs(TF)+1e-6)
    return TF.astype(np.float32)[:, :224]


def gendata(data_path, out_path, num_shards, ignored_sample_path=None,
            benchmark='xview', part='eval', shuffle=False):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    print("Getting Filenames and Labels")
    for filename in tqdm(os.listdir(data_path)):
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
            sample_name.append(os.path.join(data_path, filename))
            sample_label.append(action_class - 1)

    if shuffle:
        p = np.random.permutation(len(sample_label))
        sample_label = np.array(sample_label)[p]
        sample_name = np.array(sample_name)[p]

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
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
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
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
                shuffle=True if 'train' in p else False)

    f = open(os.path.join(arg.out_folder, "normalization_stats.pkl"),"wb")
    pickle.dump(stat_dict,f)
    f.close()
    print("\nNormalization Stats")
    print(stat_dict)
