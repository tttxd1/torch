import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import tensorflow as tf
import FarSeg
import datetime
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
model_name_list = ['FarSeg', 'DeepUNet', 'SegNet', 'DeepLabv3+', 'SANet']
# ====================0=========1==========2===========3===========4==============
# ==================================超参数=======================================
model_name = model_name_list[0]
TRAIN_IMAGE_SIZE = 256
TEST_IMAGE_SIZE = 1024
# epochs指的就是训练过程中数据将被“轮”多少次
epochs = 300
batch_size = 8
# =============================================================================

model = FarSeg.FarSeg(input_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 4), num_class = 7, Falg_summary=True)

# =============================================================================
def mkSaveDir(mynet):
    TheTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    savePath = './logs/' + str(mynet) + '-' + str(TheTime)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return savePath


savePath = mkSaveDir(model_name)
checkpointPath= savePath + "/"+model_name+"-{epoch:03d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_loss', verbose=1,
                             save_best_only=False, save_weights_only=True, mode='auto', period=1)

EarlyStopping = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
tensorboard = TensorBoard(log_dir=savePath, histogram_freq=0)
callback_lists = [tensorboard, EarlyStopping, checkpoint]
# ===================================训练=========================================
train_image = np.load(r"E:\coastline_data\images\images_Train.npy")
train_Classify_GT    = np.load(r"E:\coastline_data\classify\class_labels_Train.npy")
train_SeaLand_GT = np.load(r"E:\coastline_data\coastline_two\TwoClass_Train.npy")
valid_image = np.load(r"E:\coastline_data\images\images_Vaild.npy")
valid_Classify_GT = np.load(r"E:\coastline_data\classify\class_labels_Vaild.npy")
valid_SeaLand_GT = np.load(r"E:\coastline_data\coastline_two\TwoClass_Vaild.npy")

# validation_data用来在每个epoch之后，或者每几个epoch，验证一次验证集，用来及早发现问题，比如过拟合等。
# ================= 原图 ========= <分类标签图 ，岸线标签图>；=== batch_size ：一次训练所选取的样本数。
History = model.fit(train_image, [train_Classify_GT, train_SeaLand_GT], batch_size=batch_size,
            # ===== 验证集的输入 ==== <验证集原图>==== [分类的标签，岸线标签]
                    validation_data=(valid_image, [valid_Classify_GT, valid_SeaLand_GT]),
                            epochs=epochs, verbose=1, shuffle=True,  callbacks=callback_lists)
with open(savePath + '/log_256.txt','w') as f:
    f.write(str(History.history))