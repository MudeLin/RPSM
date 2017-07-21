#!/bin/bash
mkdir /data/human3.6m/ImageList
python collect_all_filelist.py /data/human3.6m/norm_img_list /data/human3.6m/ImageList/all_camera_train.txt train
python collect_all_filelist.py /data/human3.6m/norm_img_list /data/human3.6m/ImageList/all_camera_test.txt test
