
import os
import sys
import numpy as np
import copy
from read_file import read_labels

        
def cal_mean_limb(src, dst):
  parent_node = {0:0,
                1:0,2:1,3:2,
                4:0,5:4,6:5,
                7:0,8:7,9:8,10:9,
                11:8,12:11,13:12,
                14:8,15:14,16:15}
  pose_indexes = [0, 1,2,3, 4,5,6, 7,8,9,10, 11,12,13, 14,15,16];
  fns, labels = read_labels(src)
  labels = np.array(labels)
  labels = np.reshape(labels,[labels.shape[0], labels.shape[1]/3, 3])
  limb_length = np.zeros((labels.shape[0], labels.shape[1]))
  for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
      parent_id = parent_node[j];
      limb_length[i][j] = np.linalg.norm(labels[i][j] - labels[i][parent_id])

  mean_limb_length = np.mean(limb_length,axis = 0)
  print mean_limb_length
  with open(dst,'w') as f:
    mean_limb_length = mean_limb_length.tolist()
    mean_limb_length = [str(x) for x in mean_limb_length]
    line = ','.join(mean_limb_length)
    f.write(line + '\n')
    print line
      


if __name__ == "__main__":
  if len(sys.argv ) != 3:
    print "Error!"
    print "Usage: python cal_mean_limb.py src dst"
    
  else:

    ori_label_file = sys.argv[1]
    dst_file = sys.argv[2]
    cal_mean_limb(ori_label_file,dst_file)
