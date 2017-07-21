#!/bin/bash

for fn in `find /data/human3.6m/norm_img_list/ *.txt`;
do
  if [ -f $fn ]; then
    echo $fn
    sed -i 's/linux_cropped_img/linux_square_img/g' $fn
  fi
done
