import h5py
import cv2


f=h5py.File("E:/keras/liver_tumor_segmentation_keras/data/h5/train_liver.h5","r")
images = f['images']
masks = f['masks']
lenght=len(f)
print(images.shape[0])
# exit()

for i in range(lenght):
    image = images[i]
    label = masks[i]
    label[label > 0] = 255
    cv2.imwrite('.../visual_h5/image/'+str(i)+'.png',(image*255).astype('uint8'))
    cv2.imwrite('..../visual_h5/label/'+str(i)+'.png', label)

