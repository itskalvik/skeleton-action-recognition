import torch
import numpy as np
from nnAudio.Spectrogram import STFT
"""
Defines default edges of the graph data, here the edges correspond to the
NTU RGB+D dataset's skeleton structure. Some vertices and edges with minimal
impact to spectrograms were removed.
http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
"""
edges = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7),
         (7, 21), (7, 22), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23),
         (11, 24), (0, 16), (0, 12), (12, 13), (13, 14), (14, 15), (16, 17),
         (17, 18), (18, 19)]


class VirtualRadar(torch.nn.Module):
    """pytorch layer that generates radar returns from graph data. Each edge is
    modeled as a ellipsoid. The radar returns form each ellipsoid is computed
    seperately and superimposed on to a single Spectrogram. The radar returns
    do not model the interaction of the signals bouncing off multiple ellipsoids

    References:
      Radar cross section back scatter or radar returns are computed for an
      ellipsoid following the definition mentioned in Radar Systems Analysis
      and Design Using MATLAB, By B. Mahafza, Chapman & Hall/CRC 2000.

    Notes:
      Method to compute stft on complex values using STFT function which
      only supports float values.

      s = a + bj
      a_real, a_imag = stft(a)
      b_real, b_imag = stft(b)
      stft(s) = (a_real-b_imag) + (a_imag+b_real)j
    """
    def __init__(self,
                 edges=edges,
                 wavelength=1e-3,
                 radar_location=[0., 0., 0.],
                 train_wavelength=False,
                 train_radar_location=False,
                 train_stft_kernel=False,
                 n_fft=256,
                 hop_length=16,
                 device='cuda:0'):
        """
        Args:
          edges: A list of tuples. Defines edges of the graph data.
          wavelength: A float value. Wavelength of the virtual radar
          radar_location: A list of 3 float values. X, Y and Z coordinates of
            of the virtual radarfrom
          train_wavelength: A bool value. If true, updates wavelength using
            gradint descent
          train_radar_location: A bool value. If true, updates radar_location
            using gradint descent
          train_stft_kernel: A bool value. Determine if the STFT kenrels are
            trainable or not. If True, the gradients for STFT kernels will
            also be caluclated and the STFT kernels will be updated during model
            training.
          n_fft: A int value. The window size of FFT in STFT.
          hop_length: A int value. The hop (or stride) size in STFT.
          device: A str value. Choose which device to initialize this layer.
        """
        super().__init__()
        self.wavelength = torch.nn.Parameter(torch.as_tensor(wavelength),
                                             requires_grad=train_wavelength)
        self.radar_location = torch.nn.Parameter(
            torch.as_tensor(radar_location),
            requires_grad=train_radar_location)
        self.src, self.dst = map(list, zip(*edges))
        self.stft = STFT(n_fft=n_fft,
                         freq_bins=n_fft,
                         hop_length=hop_length,
                         output_format='Complex',
                         trainable=train_stft_kernel,
                         device=device)
        self.n_fft = n_fft

    def forward(self, x):
        """
        Computes forward propogation of the layer
        Args:
           x: A tensor with shape
                    (batch_size, num_features, timesteps, vertices, num_graphs)
              num_features: 3, (x, y, z) coordinates of each vertex.
              num_graphs: 1 or higher, number of seperate graph entities present
                in the data.
                example: If each graph modeled a human skeleton, radar return
                    generated from multiple humans can be superimposed onto
                    the same Spectrogram.
        """

        source_joints = x[:, :, :, self.src]
        destination_joints = x[:, :, :, self.dst]

        radar_ellipsoid_vector = torch.abs(source_joints -
                                           self.radar_location[:, None, None,
                                                               None])
        distances = torch.norm(radar_ellipsoid_vector, dim=1)

        A = self.radar_location[:, None, None, None]-\
                ((source_joints+destination_joints)/2)
        B = destination_joints - source_joints
        theta = torch.acos(torch.sum(A*B, dim=1) /\
                        ((torch.norm(A, dim=1) * torch.norm(B, dim=1))+1e-6))
        phi = torch.asin(
            (self.radar_location[1] - source_joints[:, 1]) /
            (torch.norm(radar_ellipsoid_vector[:, :2], dim=1) + 1e-6))

        c = torch.mean(torch.norm(source_joints - destination_joints, dim=1),
                       dim=2,
                       keepdim=True)
        c = torch.pow(c, 2)
        rcs = (np.pi*c)/((torch.sin(theta)**2)*(torch.cos(phi)**2) + \
                         (torch.sin(theta)**2)*(torch.sin(phi)**2) + \
                       c*(torch.cos(theta)**2))**2

        amp = torch.sqrt(rcs)
        theta = 4 * np.pi * distances / self.wavelength

        phase_data = torch.stack(
            (amp * torch.cos(theta), amp * torch.sin(theta)), dim=4)
        phase_data = torch.sum(phase_data, dim=[2, 3])
        stft_phase_real = self.stft(phase_data[..., 0])
        stft_phase_imag = self.stft(phase_data[..., 1])
        phase_data = torch.stack(
            (stft_phase_real[..., 0] - stft_phase_imag[..., 1],
             stft_phase_real[..., 1] + stft_phase_imag[..., 0]),
            dim=-1)

        phase_data = torch.norm(phase_data, dim=-1)
        phase_data = torch.log(phase_data + 1e-6)
        phase_data = torch.roll(phase_data, self.n_fft // 2, dims=1)
        return phase_data
