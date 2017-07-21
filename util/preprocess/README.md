


# Data preprocess
Assuming the generated label for training and validating will be stored in /data/human3.6m/train_valid_data.
THe generated label for each actions are stored in /data/hhuman3.6m/norm_img_list
The generated image sequence are stored in /data/human3.6m/linux_square_imgs.

1. Download data and develop code from Human3.6m official site.
2. Following the instruction in the develop code to set up the environment.
3. Put 'extract_imgs_and_labels.m' into 'H36MDemo' subfoler of the develop code, cd into this folder and run 'extract_imgs_and_labels' to extract images and labels.
4. Run 'crop_square_img.py' to squarely crop the subject in the image.
5. Run 'split_by_action.py' to split the data by actions and split training and testing sets.
6. Run 'substract_root_joint.sh'.
7. Run 'convert_to_square.sh' to convert the image to square crop imgs
8. Run 'collect_all_filelist.sh' to collect files for train and test
9. Run  'bash mean_normalized.sh'  to mean normlize the labels

After the above steps, there will be two files contains the img_label list for training and testing.

# Create H5 files
simply run the create_h5.lua scripts in `../util/`, files contains the lables and 
sequence infomation should be stored in `../data/h3m/tempo/`

# Preprocessed data
You could also download the data that preprocessed by above process at [H3.6M data](https://pan.baidu.com/s/1bpvSLBp)
