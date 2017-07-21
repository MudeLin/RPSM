
import os
import sys

# phase = (train, test)
def collect_all_filelist(root_folder,dst_file,phase):
	for folder,_,fns in os.walk(root_folder):
	  # print folder,phase
	  if phase in folder:
	    cur_gt_file = open(folder + '/filename_gt.txt','r')
	    for li in cur_gt_file.readlines():
	      dst_file.write(li)

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print "Error! \n Usage: python collect_all_filelist.py root_folder dst_file phase"
	root_folder = sys.argv[1]
	dst_file = open(sys.argv[2],'w')
	phase = sys.argv[3]
	collect_all_filelist(root_folder,dst_file,phase)
