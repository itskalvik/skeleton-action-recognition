#!/bin/bash

python3 gen_tfrecord_data.py --label-path ../data/ntu/xview/train_label.pkl --shuffle True --data-path ../data/ntu/xview/train_data_joint_real.npy --dest-folder ../data/ntu/xview/train_data_joint_real_smooth
python3 gen_tfrecord_data.py --label-path ../data/ntu/xview/val_label.pkl --shuffle False --data-path ../data/ntu/xview/val_data_joint_real.npy --dest-folder ../data/ntu/xview/val_data_joint_real_smooth
