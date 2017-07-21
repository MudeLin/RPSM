from read_file import read_labels
import numpy as np
import sys

def cal_max_min(src, dst_max, dst_min):
	train_data = src
	fns, labels = read_labels(train_data)
	labels = np.array(labels)
	max_value = labels.max(0)
	min_value = labels.min(0)
	with open(dst_max,'w') as f:
		max_value[0] = float(0.0001)
		max_value[1] = float(0.0001)
		max_value[2] = float(0.0001)
		write_str = [str(x) for x in max_value]
		f.write(','.join(write_str) + '\n')
		

	with open(dst_min,'w') as f:
		write_str = [str(x) for x in min_value]
		f.write(','.join(write_str) + '\n')
		

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print('Error')
		print("Usage: python cal_max_min.py src dst_max dst_min")
	else:
		src = sys.argv[1]
		dst_max = sys.argv[2]
		dst_min = sys.argv[3]
		cal_max_min(src, dst_max, dst_min)