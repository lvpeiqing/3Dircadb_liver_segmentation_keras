# 3Dircadb_liver_segmentation_keras

数据集3Dircadb， http://ircad.fr/research/3d-ircadb-01
                百度网盘链接：链接：https://pan.baidu.com/s/1wePSYU2icEyxl7pASTE4ow 
                                   提取码：wwmu 

How to use it？

step1：

       python preprocessing_train_data.py 
       python preprocessing_val_data.py 
       
       data
          3Dircadb
             3Dircadb1.1
                 MASKS_DICOM
                    MASKS_DICOM
                      liver
                       image_0
                       image_1
                       .......
                        
                 PATIENT_DICOM
                    PATIENT_DICOM
                      image_0
                      image_1
                      .......
             3Dircadb1.1
             3Dircadb1.1
             3Dircadb1.1
             ...........
             3Dircadb1.20
             
          h5
            train_liver.h5
            val_liver.h5
           
          preds
             
       
step2:
      python train.py
      
step3:
      python test.py

note: python visual_dicom.py和python visual_h5.py这两个可视化代码用于检测数据预处理之后的正确性 
