from model.unet import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from dataset.HDF5DatasetGenerator import HDF5DatasetGenerator
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

reader = HDF5DatasetGenerator(dbPath=outputPath,batchSize=BATCH_SIZE)
train_iter = reader.generator()

test_reader = HDF5DatasetGenerator(dbPath=val_outputPath,batchSize=BATCH_SIZE)
test_iter = test_reader.generator()
# fixed_test_images, fixed_test_masks = test_iter.__next__()
#
# def lr_schedule(epoch): return 0.0005 * 0.4**epoch
model = get_unet()
model_checkpoint = ModelCheckpoint(
    filepath='....\checkpoints/weights_unet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False,
    save_weights_only=False)
# learning_rate = np.array([lr_schedule(i) for i in range(10)])
# reduce_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
callbacks = [tensor_board, model_checkpoint, early_stop]  # , reduce_lr]
# 注：感觉validation的方式写的不对，应该不是这样弄的
model.fit_generator(train_iter,
                    steps_per_epoch=int(TOTAL / BATCH_SIZE),
                    verbose=1,
                    epochs=500,
                    shuffle=True,
                    validation_data=test_iter,
                    validation_steps=int(TOTAL_VAL / BATCH_SIZE),
                    callbacks=callbacks)

reader.close()
test_reader.close()



