import numpy as np
import pydicom
import os
from skimage import io
from model.unet import *
import tensorflow as tf


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# partB 接partA
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TOTAL = 2782  # 总共的训练数据
TOTAL_VAL = 152  # 总共的validation数据
# part1部分储存的数据文件
outputPath = '.../data/h5/train_liver.h5'  # 训练文件
val_outputPath = '.../data/h5/val_liver.h5'
# checkpoint_path = 'model.ckpt'
BATCH_SIZE = 4  # 根据服务器的GPU显存进行调整

print('-' * 30)
print('Loading and preprocessing test data...')
test_reader = HDF5DatasetGenerator(dbPath=val_outputPath, batchSize=BATCH_SIZE)
test_iter = test_reader.generator()
fixed_test_images, fixed_test_masks = test_iter.__next__()
print('-' * 30)

print('-' * 30)
model = get_unet()
print('Loading saved weights...')
print('-' * 30)
model.load_weights('.../checkpoints/weights_unet-28--0.92.h5')

print('-' * 30)
print('Predicting masks on test data...')
imgs_mask_test = model.predict(fixed_test_images, verbose=1)
print('-' * 30)

print('-' * 30)
print('Saving predicted masks to files...')
np.save('imgs_mask_test.npy', imgs_mask_test)
print('-' * 30)

pred_dir = '...\data\preds'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

i = 0
for image in imgs_mask_test:
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    gt = (fixed_test_masks[i, :, :, 0] * 255.).astype(np.uint8)
    ini = (fixed_test_images[i, :, :, 0] * 255.).astype(np.uint8)
    io.imsave(os.path.join(pred_dir, str(i) + '_ini.png'), ini)
    io.imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
    io.imsave(os.path.join(pred_dir, str(i) + '_gt.png'), gt)
    i += 1

print("total images in test ", str(i))