import cv2
import glob
import os
from PIL import Image
import numpy as np

def folder_gray2BGR(folder):
  imgz_paths = []
  for path in glob.glob(os.path.join(folder,'*bmp')):
    imgz_paths.append(path)
  for path in glob.glob(os.path.join(folder,'*jpg')):
    imgz_paths.append(path)
  for path in glob.glob(os.path.join(folder,'*jpeg')):
    imgz_paths.append(path)
  for path in imgz_paths:
    img = cv2.imread(path)
    cv2.imwrite(path,img)
  
def makedir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def remove_objects(img, lower_size=None, upper_size=None):
  # find all objects
  nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
  sizes = stats[1:, -1]
  _img = np.zeros((labels.shape))
  # process all objects, label=0 is background, objects are started from 1
  for i in range(1, nlabels):
  # remove small objects
    if (lower_size is not None) and (upper_size is not None):
      if lower_size < sizes[i - 1] and sizes[i - 1] < upper_size:
        _img[labels == i] = 255
    elif (lower_size is not None) and (upper_size is None):
      if lower_size < sizes[i - 1]:
        _img[labels == i] = 255
    elif (lower_size is None) and (upper_size is not None):
      if sizes[i - 1] < upper_size:
        _img[labels == i] = 255
  return _img
def anaume(img):
  kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
  th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_ellipse)
  th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_ellipse)
  th_clean = remove_objects(th, lower_size=30000, upper_size=None)
  th_clean = th_clean.astype(np.uint8)
  th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)
  th_clean_not = cv2.bitwise_not(th_clean)
  th_clean_not_clean = remove_objects(th_clean_not, lower_size=None, upper_size=10000)
  th_clean_not_clean = th_clean_not_clean.astype(np.uint8)
  anaume_img = cv2.bitwise_or(th_clean, th_clean_not_clean)
  return anaume_img

def anaume_FixedThreshold(img, fixed_threshold):
  kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  ret, th = cv2.threshold(gray, fixed_threshold, 255, cv2.THRESH_BINARY_INV)
  th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_ellipse)
  th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_ellipse)
  th_clean = remove_objects(th, lower_size=30000, upper_size=None)
  th_clean = th_clean.astype(np.uint8)
  th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)
  th_clean_not = cv2.bitwise_not(th_clean)
  th_clean_not_clean = remove_objects(th_clean_not, lower_size=None, upper_size=10000)
  th_clean_not_clean = th_clean_not_clean.astype(np.uint8)
  anaume_img = cv2.bitwise_or(th_clean, th_clean_not_clean)
  return anaume_img

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
        
def make_square_file(in_path, out_path):
  out_img_path = out_path + "\\" + os.path.basename(in_path)
  im = Image.open(in_path)
  im_new = expand2square(im, 0)
  im_new.save(out_img_path, quality=95)

def output_overall_length(img_path, skeleton_name="aaa", skeleton_path="skeleton_path"):
  anaume = cv2.imread(img_path)
  skeleton = cv2.ximgproc.thinning(anaume, thinningType=cv2.ximgproc.THINNING_GUOHALL)
  if skeleton_path:
    cv2.imwrite(os.path.join(skeleton_path, skeleton_name + '.bmp'), skeleton)
  
  overall_length = cv2.countNonZero(skeleton)
  return overall_length