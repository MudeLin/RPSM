import sys
import numpy as np
from read_file import read_labels

def read_mean_lim(mean_limb_file):
	mean_limb = open(mean_limb_file,'r').readline()
	mean_limb = mean_limb.split(',')
	mean_limb = [float(x) for x in mean_limb]

	return mean_limb


def mean_limb_scale(preds, mean_limb_file):
	parent_node = {0:0,
                1:0,2:1,3:2,
                4:0,5:4,6:5,
                7:0,8:7,9:8,10:9,
                11:8,12:11,13:12,
                14:8,15:14,16:15}
	pose_indexes = [0, 1,2,3, 4,5,6, 7,8,9,10, 11,12,13, 14,15,16];
	mean_limb = read_mean_lim(mean_limb_file)
	origin_preds = preds.copy()
	for k in range(preds.shape[0]):
		for  i in range(1,len(mean_limb)):
			
			my_length = np.linalg.norm(origin_preds[k][i] - origin_preds[k][parent_node[i]])
			scale = mean_limb[i] / my_length 
			# print i, scale
			preds[k][i] = (origin_preds[k][i] - origin_preds[k][parent_node[i]])*scale + preds[k][parent_node[i]]
	return preds

def mean_scale(src,dst, mean_limb_file):
  fns, labels = read_labels(src)
  labels = np.array(labels)
  labels = np.reshape(labels,[labels.shape[0], labels.shape[1]/3, 3])
  labels = mean_limb_scale(labels, mean_limb_file)
  labels = np.reshape(labels,[labels.shape[0], labels.shape[1] * labels.shape[2]])
  with open(dst,'w') as dst_f:
  	for i in range(len(fns)):
  		write_str = labels[i] 
  		write_str = [str(x) for x in write_str]
  		write_str = ','.join(write_str)
  		write_str = fns[i] +' ' +write_str
  		dst_f.write(write_str + '\n')

if __name__ == "__main__":
  if len(sys.argv ) != 4:
    print "Error!"
    print "Usage: python mean_limb_scale.py src dst mean_limb_file"
  else:

    ori_label_file = sys.argv[1]
    dst_file = sys.argv[2]
    mean_limb_file = sys.argv[3]
    mean_scale(ori_label_file,dst_file, mean_limb_file)