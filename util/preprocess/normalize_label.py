'''
   Generate label for multi task, that is regerssion on the labels while predicting the pose classifications
   pose classification class is get by cluster on the laber, current cluster number is 128.
   # 1, get max min value
   # 2, normalize label
   # concat cluster ground truth
'''
from read_file import read_labels
import numpy as np
import os
import sys

kMaxValueFile = None
kMinValueFile = None

def get_max_min():
  max_value = np.genfromtxt(kMaxValueFile, delimiter=',')
  min_value = np.genfromtxt(kMinValueFile, delimiter=',')

  return max_value,min_value

num_valid = 3000
valid_ratio = 0.03

def normalize_label(gt_fn, dst_gt_fn, valid_data = False):
  max_value,min_value = get_max_min()
  fns = []
 
  dst_file = open(dst_gt_fn,'w')
  sample_id = 0
  fns,gt_labels = read_labels(gt_fn)
  for i in range(len(fns)):
    if valid_data and np.random.uniform() > valid_ratio:
      continue
    if valid_data and sample_id >= num_valid:
      break
    fn = fns[i]
    labels = gt_labels[i]
    labels = np.array(labels)
    labels = (labels - min_value)/(max_value - min_value)
    labels = labels.tolist()
    labels = [str(x) for x in labels]
    label_str = ','.join(labels)
    dst_file.write(fn + ' ' + label_str + '\n')
    sample_id += 1
    # print sample_id

if __name__ == '__main__':
  if len(sys.argv) != 5:
    print("Error!")
    print("Usage: python normalize_label.py src dst max_value_file min_value_file")
  else:
    train_gt_fn = sys.argv[1]
    dst_train_gt_fn = sys.argv[2]
    kMaxValueFile = sys.argv[3]
    kMinValueFile = sys.argv[4]
    normalize_label(train_gt_fn,dst_train_gt_fn)






