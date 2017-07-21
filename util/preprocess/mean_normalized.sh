
mkdir /data/human3.6m/train_valid_data

python ./cal_mean_limb.py /data/human3.6m/ImageList/all_camera_train.txt \
			/data/human3.6m/train_valid_data/mean_limb.txt

python ./mean_limb_scale.py /data/human3.6m/ImageList/all_camera_train.txt \
			/data/human3.6m/train_valid_data/train_mean_limb_scaled.txt \
			/data/human3.6m/train_valid_data/mean_limb.txt

python cal_max_min.py /data/human3.6m/train_valid_data/train_mean_limb_scaled.txt \
			/data/human3.6m/train_valid_data/train_point_max.txt \
			/data/human3.6m/train_valid_data/train_point_min.txt

python ./normalize_label.py /data/human3.6m/train_valid_data/train_mean_limb_scaled.txt \
			/data/human3.6m/train_valid_data/train_all.txt \
			/data/human3.6m/train_valid_data/train_point_max.txt \
			/data/human3.6m/train_valid_data/train_point_min.txt
echo `printf '%s,' {1..50}``printf '51\n'` > tmp_index.txt
cat tmp_index.txt /data/human3.6m/train_valid_data/train_point_max.txt \
	/data/human3.6m/train_valid_data/train_point_min.txt \
	> /data/human3.6m/train_valid_data/train_point_max_min.csv
rm tmp_index.txt

python split_train_valid.py /data/human3.6m/train_valid_data/train_all.txt    \
		/data/human3.6m/train_valid_data/train.txt                   \
		/data/human3.6m/train_valid_data/valid.txt           

