#!/usr/bin/env python
# -*- coding:utf-8 -*-
import openslide
import numpy as np
import time
import pandas as pd
import scipy.io as scio
import os
import cv2
from skimage import io
from openslide.deepzoom import DeepZoomGenerator
import warnings
import math

warnings.filterwarnings('error')
time_start = time.time()

fild_id_downloaded = os.listdir('D:\\Data\\LUSC_Data\\LUSC_20')  ## svs_address
num = fild_id_downloaded.__len__()

names = []
level_count = []
max_power = []
level_downsamples = []
level_dimensions = []
seqs = []

loc = 'D:\\Data\\LUSC_Data\\LUSC_20.mat\\'  ##  file location
if not os.path.exists(loc):
    os.makedirs(loc)

for n in range(num):
    print(n + 1)
    file_name = fild_id_downloaded[n]
    print(file_name)
    save_path = 'D:\\Data\\LUSC_Data\\LUSC_ms\\' + file_name[0:12] + '\\'  ## save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = 'D:\\Data\\LUSC_Data\\LUSC_20\\' + file_name  ## single_svs_address
    slide = openslide.OpenSlide(path)

    [w1, h1] = slide.level_dimensions[0]  # size at highest magnification

    print(w1, h1)
    w = math.ceil(w1 / 2)
    h = math.ceil(h1 / 2)
    print(w, h)
    highth = 512
    pix = highth
    num_h = int(h / pix)
    num_w = int(w / pix)
    data_gen = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
    location = np.zeros((num_h, num_w))
    seq = 1
    for i in range(num_h):
        for j in range(num_w):
            img = np.array(data_gen.get_tile(data_gen.level_count - 1, (j, i)))  ## level_count-1 is the maximum magnification
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h1, s1, v1 = cv2.split(hsv_image)
            s2 = cv2.blur(s1, (5, 5))
            ret, saturation_thresholded_mask1 = cv2.threshold(s2, 15, 255, cv2.THRESH_BINARY)
            roi1 = saturation_thresholded_mask1 / 255
            if np.sum(roi1) / (highth * highth) > 0.5:
                try:
                     image_save_path = "%s%d%s" % (save_path, seq, '.jpg')
                     io.imsave(image_save_path, img)
                     location[i, j] = 1
                     seq = seq + 1
                except Warning:
                    pass
    seqs.append(seq)
    print(seq)

    dataNew = loc + file_name[0:12] + '.mat'  ############### Location
    scio.savemat(dataNew, {'location': location})
    names.append(file_name[0:12])

dic = {
    'name': names,
    'num': seqs,
}
patient_names = pd.DataFrame(dic)
patient_names.to_csv('D:\\Data\\LUSC_Data\\LUSC_supp_20.csv', index=None) # Files that count the number of paths cropped

time_end = time.time()
print('totally cost', time_end - time_start)
