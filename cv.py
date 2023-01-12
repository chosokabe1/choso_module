import cv2
import glob
import os
from PIL import Image
import numpy as np
import shutil

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
  
def copy_folder(from_folder, to_folder):
  for from_file in glob.glob(os.path.join(from_folder,'*')):
    shutil.copy(from_file, to_folder)

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
        
def padding_0_min_size_img(img, min_size):
  h, w, _ = img.shape
  flag = False
  if h < min_size and w < min_size:
    newimg = np.zeros((min_size, min_size, 3))
    start_h = int((min_size - h) / 2)
    fin_h = int((min_size + h) / 2)
    start_w = int((min_size - w) / 2)
    fin_w = int((min_size + w) / 2)

    newimg[start_h:fin_h, start_w:fin_w] = img
    flag = True
  
  elif w < min_size:
    newimg = np.zeros((h, min_size, 3))
    start = int((min_size - w) / 2)
    fin = int((min_size + w) / 2)
    newimg[:, start:fin] = img
    flag = True

  elif h < min_size:
    newimg = np.zeros((min_size, min_size, 3))
    start_h = int((min_size - h) / 2)
    fin_h = int((min_size + h) / 2)
    newimg[start_h:fin_h, :] = img
    flag = True

  if flag:
    return newimg
  else:
    return img


def make_square_file(in_path, out_path):
  out_img_path = out_path + "\\" + os.path.basename(in_path)
  im = Image.open(in_path)
  im_new = expand2square(im, 0)
  im_new.save(out_img_path, quality=95)

def output_overall_length(anaume, skeleton_name="aaa", skeleton_path="skeleton_path"):
  skeleton = cv2.ximgproc.thinning(anaume, thinningType=cv2.ximgproc.THINNING_GUOHALL)
  if skeleton_path:
    cv2.imwrite(os.path.join(skeleton_path, skeleton_name + '.bmp'), skeleton)
  
  overall_length = cv2.countNonZero(skeleton)
  return overall_length

def folder_bright_kakutyou(in_folder, out_folder):
  gamma1 = 0.5
  gamma2 = 0.75
  gamma3 = 1.25
  gamma4 = 1.5
  img2gamma1 = np.zeros((256,1),dtype=np.uint8)
  img2gamma2 = np.zeros((256,1),dtype=np.uint8)
  img2gamma3 = np.zeros((256,1),dtype=np.uint8)
  img2gamma4 = np.zeros((256,1),dtype=np.uint8)

  for i in range(256):
    img2gamma1[i][0] = 255 * (float(i)/255) ** (1.0 / gamma1)
    img2gamma2[i][0] = 255 * (float(i)/255) ** (1.0 / gamma2)
    img2gamma3[i][0] = 255 * (float(i)/255) ** (1.0 / gamma3)
    img2gamma4[i][0] = 255 * (float(i)/255) ** (1.0 / gamma4)


  for idx,in_path in enumerate(glob.glob(os.path.join(in_folder, '*'))):
    img = cv2.imread(in_path)
    if idx % 4 == 0:
      gamma_img = cv2.LUT(img,img2gamma1)
    elif idx % 4 == 1:
      gamma_img = cv2.LUT(img,img2gamma2)
    elif idx % 4 == 2:
      gamma_img = cv2.LUT(img,img2gamma3)
    else:
      gamma_img = cv2.LUT(img,img2gamma4)

    cv2.imwrite(os.path.join(out_folder, os.path.basename(in_path)), img)
    cv2.imwrite(os.path.join(out_folder, os.path.splitext(os.path.basename(in_path))[0] +'_bright.jpg'), gamma_img)