#!/bin/bash


NUM_SUBJECT=15
	NEW_DATA_LIST_ROOT=/data/human3.6m/new_img_list
	DST_DATA_LIST_ROOT=/data/human3.6m/norm_img_list
	echo $NEW_DATA_LIST_ROOT,$DST_DATA_LIST_ROOT
	if [ ! -d $NEW_DATA_LIST_ROOT ]; then
	  python ./split_by_action.py 
	fi

	if [ ! -d $DST_DATA_LIST_ROOT ]; then
	  mkdir $DST_DATA_LIST_ROOT 
	fi


	for ((i=1;i<=$NUM_SUBJECT;++i))
	do
	  echo $i
	  python ./substract_root_joint.py  $NEW_DATA_LIST_ROOT/train_$i.txt $DST_DATA_LIST_ROOT/train_$i
	  python ./substract_root_joint.py  $NEW_DATA_LIST_ROOT/test_$i.txt  $DST_DATA_LIST_ROOT/test_$i
	done
