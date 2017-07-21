
import os
import sys
import numpy as np
import copy

def gen_relative(ori_label_file,dst_folder):
  if os.path.isdir(dst_folder) != True:
    os.mkdir(dst_folder)
  if os.path.exists(ori_label_file) != True:
    print "Error!"
    print "Input file not exists"
    return 
  ori_label_file = open(ori_label_file,'r')
  dst_label_file = open(dst_folder + os.path.sep + 'filename_gt.txt','w')
  dst_filenames = open(dst_folder + os.path.sep + 'filenames.txt','w')
  dst_root_location = open(dst_folder + os.path.sep + 'root_locations.txt','w')
  for line in ori_label_file.readlines():
    elem = line.strip().split(' ')
    fn = elem[0]
    label = map(lambda x:float(x), elem[1].split(','))
    label = np.array(label)
    label = label.reshape([np.size(label)/3,3])
    root_loc = copy.deepcopy(label[0])
    for i in range(np.size(label,0)):
      label[i] = label[i] - root_loc

    root_loc_label = root_loc.tolist()

    root_loc_label = map(lambda x:str(x),root_loc_label)

    relative_label = np.reshape(label,np.size(label)).tolist()
    relative_label = map(lambda x:str(x), relative_label) 

    dst_filenames.write(fn + '\n')
    dst_root_location.write(','.join(root_loc_label) + '\n')
    dst_label_file.write(fn + ' ' + ','.join(relative_label) + '\n')
    

if __name__ == "__main__":
  if len(sys.argv ) != 3:
    print "Error!"
    print "Usage: python gen_relative.py ori_d3_label_file dst_folder"
  else:

    ori_label_file = sys.argv[1]
    dst_folder = sys.argv[2]
    gen_relative(ori_label_file,dst_folder)
