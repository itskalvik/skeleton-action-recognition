# NTU RGB+D Dataset Action Recognition with GNNs and CNNs
graph neural networks(tensorflow) and spatial convolutional neural
networks(pytorch). The spatial convolutional neural networks use a novel virtual
radar method to convert graph data in the NTU dataset into spectrograms (images).

## Directory Structure
- `data_gen/`: These scripts are used to convert NTU dataset from their native
               format to numpy tensors.

  - `data_gen/gen_joint_data.py`: Converts NTU data from skeleton format to
                                  joint data and saves it in numpy file

  - `data_gen/gen_bone_data.py`: Computes bone data from joint data           

  - `data_gen/gen_motion_data.py`: Computes motion data from joint and bone data

  - `data_gen/gen_tfrecord_data.py`: Converts prepossessed numpy dataset to
                                     tensorflow's tfrecord fromat. Needed to
                                     train with GNNs in this repo

- `graph/`: Methods to define graph adjacency or incidence matrix for graph
            neural networks. Incidence'
- `layers/virtual_radar.py`: Custom pytorch layer to compute spectrograms from
                             temporal graph data by simulating a virtual
                             radar to capture the graph's motion
- `models/`: Neural network definitions. All Graph Neural Networks are defined
             in tensorflow. Vanilla Spatial Convolutional Networks are defined
             in pytorch.'
- `main_gnn.py`:  Script to train on NTU dataset with graph neural networks
- `main_spectrogram.py`: Script to train on NTU dataset with spatial
                         convolutional neural networks. Since the NTU data is
                         natively represented as graphs, it is converted to
                         spectrogram(images) with the custom VirtualRadar layer.

## NTU RGB+D dataset

The [NTU RGB+D dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf) can be downloaded from [here](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp).
We'll only need the Skeleton data (~ 5.8G).

## References

- Parts of the code is based on the following repos
  - [mmskeleton](https://github.com/open-mmlab/mmskeleton)
  - [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
  - [ST-GCN](https://github.com/kdkalvik/ST-GCN)
