def read_labels(gt_label_path):
  with open(gt_label_path,'r') as gt_label_file:
    labels = []
    filenames = []
    for lin in gt_label_file.readlines():
      label = lin.strip().split(' ')[1]
      label = label.split(',')
      label = map(lambda x:float(x),label)
      labels.append(label)
      filenames.append(lin.split(' ')[0])
       
    return filenames,labels
