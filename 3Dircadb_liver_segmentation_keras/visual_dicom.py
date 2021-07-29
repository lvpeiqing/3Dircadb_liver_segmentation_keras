import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2


i = 1
data_path = '..../data/3Dircadb/3Dircadb1.%d/PATIENT_DICOM/PATIENT_DICOM' % i
label_path = '..../data/3Dircadb/3Dircadb1.%d/MASKS_DICOM/MASKS_DICOM/liver' % i

image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
image_slices.sort(key = lambda x: int(x.InstanceNumber))

print(len(image_slices)) #129
print(image_slices[0].pixel_array.shape) #(512, 512)

image = np.stack([s.pixel_array for s in image_slices]) #np.stack()将单Dicom拼接成一个ct图像
print(image.shape) #(129, 512, 512)


plt.imshow(image_slices[0].pixel_arra) #或者plt.imshow(image[o],cmap='gray)
plt.show()
