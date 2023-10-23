import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import mahotas
import os, time, random

data_path = 'D:\\Data\\LUSC_Data\\LUSC_his' # File location of path after unified magnification and histogram matching
patient_list1 = os.listdir(data_path)
patient_list2 = [i[0:12] for i in patient_list1]

for m in range(len(patient_list1)):
    image_list = os.listdir(os.path.join(data_path, patient_list1[m]))
    print('patient %d' % m)
    for times in range(int(len(image_list))):
        image_path = os.path.join(data_path,patient_list1[m], image_list[times])
        image = mpimg.imread(image_path)
        lbp_feature_r = mahotas.features.lbp(image[:,:,0],1,9)
        lbp_feature_g = mahotas.features.lbp(image[:,:,1],1,9)
        lbp_feature_b = mahotas.features.lbp(image[:,:,2],1,9)
        feature_lbp = (lbp_feature_r + lbp_feature_g + lbp_feature_b)/3   # LBP features

        image_uint8 = image.astype('uint8')
        feature_pftas = mahotas.features.pftas(image_uint8)  # pftas features
        feature_lbp = feature_lbo[np.newaxis, :]
        feature_pftas = feature_pftas[np.newaxis,:]
        if times == 0:
            one_patient_lbp = feature_lbp
            one_patient_pftas = feature_pftas
        else:
            one_patient_lbp = np.vstack((one_patient_lbp,feature_lbp))
            one_patient_pftas = np.vstack((one_patient_pftas, feature_pftas))

    one_patient1_lbp = np.mean(one_patient_lbp,axis=0)
    one_patient1_pftas = np.mean(one_patient_pftas, axis=0)
    if m == 0:
        feature_matrix_lbp = one_patient1_lbp
        feature_matrix_pftas = one_patient1_pftas
    else:
        feature_matrix_lbp = np.vstack((feature_matrix_lbp,one_patient1_lbp))
        feature_matrix_pftas = np.vstack((feature_matrix_pftas, one_patient1_pftas))

import pandas as pd
output1 = pd.DataFrame(feature_matrix_lbp)
output2 = pd.DataFrame(feature_matrix_pftas)
output1.to_csv('D:\\Data\\LUSC_Data\\LUSC_lbp.csv')   # File to save lbp features
output2.to_csv('D:\\Data\\LUSC_Data\\LUSC_pftas.csv') # File to save pftas features
name = pd.DataFrame(patient_list2)
name.to_csv('D:\\Data\\LUSC_Data\\LUSC_name.csv')      # File to Save the patient number corresponding to the pftas features file
time_end=time.time()
