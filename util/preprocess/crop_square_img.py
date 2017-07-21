import cv2
import os

def square_bbox(bbox):
  x = bbox[0]
  y = bbox[1]
  w = bbox[2]
  h = bbox[3]
  if w > h:
    border = (w -h ) /2
    y = max(0, y - border)
    h = w
  else:
    border = (h-w) /2
    x =  max(0, x - border)
    w = h 
  return [x,y,w,h]

def crop_square_img(file_list,img_root, new_root_folder):
  with open(file_list,'r') as bbox_file:
    for lin in bbox_file.readlines():
      elem = lin.split(' ')
      fn = elem[0]
      bbox = elem[1].split(',')
      
      fn_elem = fn.split('/') 
      fn_prefix = '/'.join(fn_elem[4:-1])
      fn_name = fn_elem[-1]
      new_prefix = new_root_folder   + '/'+ fn_prefix 
      if not os.path.isdir(new_prefix):
        os.makedirs(new_prefix)
      new_fn = new_prefix + '/' + fn_name
      bbox = map(lambda x:float(x), bbox)
      
      
      img_path = os.path.abspath(img_root  + '/' + fn)
      img = cv2.imread(img_path)
      bbox = square_bbox(bbox)      
      square_img = img[bbox[1]:bbox[1] + bbox[3],bbox[0]: bbox[0] + bbox[2]]
      cv2.imwrite(new_fn, square_img) 

if __name__ == '__main__':
  img_root = '/data/human3.6m/Release-v1.1/H36MDemo'
  new_img_root = '/data/human3.6m/linux_square_imgs'
  
  if not os.path.isdir(new_img_root):
    os.mkdir(new_img_root)

  for phase in ['train','val']:
    for camera in ['1','2','3','4']:
      file_list = '/data/human3.6m/img_list/linux_accv_%s_camera_%sbbox.txt'%(phase,camera)
      crop_square_img(file_list, img_root, new_img_root)

  
      
