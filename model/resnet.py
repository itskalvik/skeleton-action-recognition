import torch
import torchvision
import numpy as np

'''
torch resnet 18 model with Skeleton to Spectrogram generator before model
'''
class Model(torch.nn.Module):
    def __init__(self, num_classes=60, image_size=256, pretrained=True):
        super(Model, self).__init__()
        base_model = torchvision.models.resnet18(pretrained=pretrained)
        base_model.conv1 = torch.nn.Conv2d(1, 64,
                                           kernel_size=7,
                                           stride=2,
                                           padding=3,
                                           bias=False)
        torch.nn.init.kaiming_normal_(base_model.conv1.weight,
                                      mode='fan_out',
                                      nonlinearity='relu')
        base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
        self.base_model = base_model
        self.spectrogram = Spectrogram(image_size=image_size)

    def forward(self, x):
        x = self.spectrogram(x)
        x = self.base_model(x)
        return x

'''
Convert Skeleton data to Spectrogram
'''
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
