import sys

import numpy as np
import os

kValidNum = 15000

def split_train_valid(src_train_file, dst_train_file, dst_valid_file):
	with open(src_train_file,'r') as src_train_f:
		with open(dst_train_file,'w') as dst_train_f:
			with open(dst_valid_file,'w') as dst_valid_f:
				lines = src_train_f.readlines()
				num_lin = len(lines)
				for lid in range(num_lin):
					if lid >= kValidNum :
						dst_train_f.write(lines[lid])
					else:
						dst_valid_f.write(lines[lid])
				print("Total train lines: ", num_lin - kValidNum)
				print("Saved to ", dst_train_file)
				print("Total valid lines: ", kValidNum)
				print("Saved to ", dst_valid_file)

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Error!")
		print("Usage: python split_train_valid.py src_train_file dst_train_file dst_valid_file ")
	else:

		split_train_valid(sys.argv[1], sys.argv[2], sys.argv[3])