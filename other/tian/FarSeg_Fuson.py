# -*- coding: utf-8 -*-
from keras.activations import sigmoid
from keras.backend import categorical_crossentropy
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import numpy as np

def lovasz_softmax(labels, probas, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """

    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels


def _SE(inputs, reduction=16):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    out_dim = K.int_shape(inputs)[channel_axis]  # 计算输入特征图的通道数
    temp_dim = max(out_dim // reduction, reduction)

    squeeze = GlobalAvgPool2D()(inputs)
    if channel_axis == -1:
        excitation = Reshape((1, 1, out_dim))(squeeze)
    else:
        excitation = Reshape((out_dim, 1, 1))(squeeze)
    excitation = Conv2D(temp_dim, 1, 1, activation='relu')(excitation)
    excitation = Conv2D(out_dim, 1, 1, activation='sigmoid')(excitation)

    return excitation
# kernel_size:3; groups:1
# kernel_size:5; groups:4
# kernel_size:7; groups:8
# kernel_size:9; groups:16
def group_conv2(x, filters, kernel_size, stride, groups, padding='same'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]  # 计算输入特征图的通道数
    nb_ig = in_channels // groups  # 对输入特征图通道进行分组
    nb_og = filters // groups  # 对输出特征图通道进行分组
    assert in_channels % groups == 0
    assert filters % groups == 0
    # assert filters > groups
    gc_list = []
    for i in range(groups):
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        x_group = ZeroPadding2D(padding=padding, data_format=None)(x_group)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel_size, strides=stride, use_bias=False)(x_group))
    return Concatenate(axis=channel_axis)(gc_list) if groups != 1 else gc_list[0]


def EPSA2(inputs, out_channel, conv_kernels, stride = 1, conv_groups=[1, 4, 8, 16]):
    in_dim = K.int_shape(inputs)
    split_num = len(conv_kernels)
    split_channel = out_channel // len(conv_kernels)
    conv_1 = group_conv2(inputs, out_channel//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
    conv_2 = group_conv2(inputs, out_channel // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                  stride=stride, groups=conv_groups[1])
    conv_3 = group_conv2(inputs, out_channel // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                  stride=stride, groups=conv_groups[2])
    conv_4 = group_conv2(inputs, out_channel // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                  stride=stride, groups=conv_groups[3])
    # 多尺度拼接
    feats = Concatenate()([conv_1, conv_2, conv_3, conv_4])
    feats = Reshape((in_dim[1], in_dim[2], split_channel, split_num))(feats)

    x1_se = _SE(conv_1)
    x2_se = _SE(conv_2)
    x3_se = _SE(conv_3)
    x4_se = _SE(conv_4)

    merge_se = Concatenate()([x1_se, x2_se, x3_se, x4_se])
    merge_se = Reshape((1, 1, split_channel, split_num))(merge_se)

    merge_se = Activation('softmax')(merge_se)
    feats_weight = Multiply()([merge_se, feats])

    feats_weight = Reshape((in_dim[1], in_dim[2], out_channel))(feats_weight)

    return feats_weight
def SWF_Conv(inputs, in_cannel ,stride, kernel_size):
    result = inputs
    # 多尺度
    r = kernel_size
    filters = in_cannel
    # (r:第一行, r：最后一行), (r：第一列, 0：最后一列))
    P_L = ZeroPadding2D(((r, r), (r, 0)))(result)
    P_R = ZeroPadding2D(((r, r), (0, r)))(result)
    P_U = ZeroPadding2D(((r, 0), (r, r)))(result)
    P_D = ZeroPadding2D(((0, r), (r, r)))(result)


    P_NW = ZeroPadding2D(((r, 0), (r, 0)))(result)
    P_NE = ZeroPadding2D(((r, 0), (0, r)))(result)
    P_SW = ZeroPadding2D(((0, r), (r, 0)))(result)
    P_SE = ZeroPadding2D(((0, r), (0, r)))(result)

    # 八种方式卷积 + relu
    C_L = Conv2D(filters, (2 * r + 1, r + 1), strides=stride, activation='relu')(P_L)
    C_R = Conv2D(filters, (2 * r + 1, r + 1), strides=stride, activation='relu')(P_R)
    C_U = Conv2D(filters, (r + 1, 2 * r + 1), strides=stride, activation='relu')(P_U)
    C_D = Conv2D(filters, (r + 1, 2 * r + 1), strides=stride, activation='relu')(P_D)
    C_NW = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_NW)
    C_NE = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_NE)
    C_SW = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_SW)
    C_SE = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_SE)

    # 拼接
    x = Concatenate()([C_L, C_R, C_U, C_D, C_NW, C_NE, C_SW, C_SE])




    # #         # 8 * w -> 4 * 8 * w -> 8 * w
    # d = 4 * 8 * w
    # expan = Conv2D(d, 1, activation='relu')(x)
    # dotsum = Lambda(tf.reduce_sum, arguments={'axis': (3), 'keepdims': True})(expan)
    # return dotsum

    # x = Conv2D(in_cannel, 1, strides=1, activation='relu')(x)
    return x

# x, 3, [64, 64, 256], stage=2, block='b'
def identity_block(input_tensor, kernel_size, filters, stage, block, flag):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 换的就是你
    # x = EPSA2(x, filters2, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# EPSA(in_channel, out_channel, conv_kernels=[3, 5, 7, 9], stride = 1, conv_groups=[1, 4, 8, 16]):
def conv_block(input_tensor, kernel_size, filters, flag, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # x = EPSA2(x, filters2, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def SceneEmbedding(x):
    squeeze = GlobalAveragePooling2D()(x)
    squeeze = Reshape((1, 1, 256))(squeeze)
    squeeze = Conv2D(256, 1, activation='relu', padding = 'same')(squeeze)
    # squeeze = K.reshape(squeeze, (-1, 1, 256))
    squeeze = Activation('sigmoid')(squeeze)
    Emb = Multiply()([squeeze, x])


    max = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(Emb)
    avg = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(Emb)
    att = Concatenate()([max, avg])
    att = Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation=None)(att)
    # att = BatchNormalization()(att)
    # att = Activation('relu')(att)

    out = Multiply()([x, att])
    # spo = att.get_shape().as_list()
    # _, h, w, filters = spo
    # spo = tf.transpose(K.reshape(att, (-1, h * w, 1)), (0, 2, 1))
    # bcT = K.batch_dot(squeeze, spo)
    # softmax_bcT = Activation('softmax')(bcT)
    # bcTd = Lambda(lambda x: K.reshape(softmax_bcT, (-1, h, w, filters)))(softmax_bcT)
    # vec_d = K.reshape(x, (-1, h * w, filters))
    # bcTd = K.batch_dot(softmax_bcT, vec_d)
    # bcTd = Lambda(lambda x: K.reshape(bcTd, (-1, h, w, filters)))(x)
    return out


def TransitionLayer(x, nb_filter, alpha=0.1):
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, 3, padding='same')(x)
    return x

def my_upsampling(x, img_w, img_h, x2):
    """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
    return tf.image.resize_images(x, (img_w * x2, img_h * x2), 0)

def FarSeg(input_size, num_class, pretrained_weights = False, Falg_summary=True , model_summary=False):
    inputs = Input(input_size)

    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 原图的1/4
    # c2 = SWF_Conv(x, 32, stride = 4, kernel_size = 1)
    c2 = MaxPooling2D(pool_size=(4, 4))(x)
    # c2：64,64,256
    c2 = conv_block(c2, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), flag=0)
    c2 = identity_block(c2, 3, [64, 64, 256], stage=2, block='b', flag=0)
    c2 = identity_block(c2, 3, [64, 64, 256], stage=2, block='c', flag=0)

    # c3：32,32,512
    # c3 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = conv_block(c2, 3, [128, 128, 512], stage=3, block='a', flag=0)
    c3 = identity_block(c3, 3, [128, 128, 512], stage=3, block='b', flag=0)
    c3 = identity_block(c3, 3, [128, 128, 512], stage=3, block='c', flag=0)
    c3 = identity_block(c3, 3, [128, 128, 512], stage=3, block='d', flag=0)

    # c4：16,16,1024
    # c4 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = conv_block(c3, 3, [256, 256, 1024], stage=4, block='a', flag=0)
    c4 = identity_block(c4, 3, [256, 256, 1024], stage=4, block='b', flag=0)
    c4 = identity_block(c4, 3, [256, 256, 1024], stage=4, block='c', flag=0)
    c4 = identity_block(c4, 3, [256, 256, 1024], stage=4, block='d', flag=0)
    c4 = identity_block(c4, 3, [256, 256, 1024], stage=4, block='e', flag=0)
    c4 = identity_block(c4, 3, [256, 256, 1024], stage=4, block='f', flag=0)

    # c5：8,8,2048
    # c5 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = conv_block(c4, 3, [512, 512, 2048], stage=5, block='a', flag=0)
    c5 = identity_block(c5, 3, [512, 512, 2048], stage=5, block='b', flag=0)
    c5 = identity_block(c5, 3, [512, 512, 2048], stage=5, block='c', flag=0)
    # ======================================解码.4=============================================
    # 1,1,n向量是256维的
    # u = Conv2D(256, 1, padding='same', name='pca1', kernel_initializer='he_normal')(c5)
    # u = SceneEmbedding(u)
    # # 8,8,256
    p5_two = Conv2D(2048, 1, padding='same', kernel_initializer='he_normal')(c5)
    p5_mul = Conv2D(2048, 1, padding='same', kernel_initializer='he_normal')(c5)
    # 2048,256,256
    # c5是低级，p5是高级
    # ======================================解码.3=============================================
    p4_two = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p5_two))
    # p4_two = TransitionLayer(p4_two, 1024)
    p4_two = concatenate([c4, p4_two])

    p4_mul = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p5_mul))
    # p4_mul = TransitionLayer(p4_mul, 1024)
    p4_mul = concatenate([c4, p4_mul])
    # ======================================解码.2=============================================
    p3_two = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p4_two))
    # p3_two = TransitionLayer(p3_two, 512)
    p3_two = concatenate([c3, p3_two])

    p3_mul = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p4_mul))
    # p3_mul = TransitionLayer(p3_mul, 512)
    p3_mul = concatenate([c3, p3_mul])
    # p3_1 = TransitionLayer(p3, 256)
    # ======================================解码.1=============================================
    p2_two = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p3_two))
    # p2_two = TransitionLayer(p2_two, 256)
    p2_two = concatenate([c2, p2_two])

    p2_mul = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p3_mul))
    # p2_mul = TransitionLayer(p2_mul, 256)
    p2_mul = concatenate([c2, p2_mul])

    # Z1 = Conv2D(256, 1, strides=1, padding='same')(p5)
    # Z1 = BatchNormalization(axis=3)(Z1)
    # Z1 = Activation('relu')(Z1)
    # R1 = Activation('sigmoid')(p5)
    # R1 = SceneEmbedding(Z1)
    # Z1 = multiply([Z1, R1])

    # Z2 = Conv2D(256, 1, strides=1, padding='same')(p4)
    # Z2 = BatchNormalization(axis=3)(Z2)
    # Z2 = Activation('relu')(Z2)
    # R2 = Activation('sigmoid')(p4)
    # R2 = SceneEmbedding(Z2)
    # Z2 = multiply([Z2, R2])
    #
    # Z3 = Conv2D(256, 1, strides=1, padding='same')(p3)
    # Z3 = BatchNormalization(axis=3)(Z3)
    # Z3 = Activation('relu')(Z3)
    # R3 = Activation('sigmoid')(p3)
    # R3 = SceneEmbedding(Z3)
    # Z3 = multiply([Z3, R3])
    #
    # Z4 = Conv2D(256, 1, strides=1, padding='same')(p2)
    # Z4 = BatchNormalization(axis=3)(Z4)
    # Z4 = Activation('relu')(Z4)
    # R4 = SceneEmbedding(Z4)
    # R4 = Activation('sigmoid')(p2)
    # Z4 = multiply([Z4, R4])

    # f_p4 = Conv2DTranspose(32, 3, strides=(8, 8), activation='relu', padding='same', kernel_initializer='he_normal')(Z1)

    # f_p3 = Conv2DTranspose(32, 3, strides=(4, 4), activation='relu', padding='same', kernel_initializer='he_normal')(Z2)

    # f_p2 = Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(Z3)

    # f_p1 = Conv2DTranspose(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Z4)

    # x1 = Concatenate()([f_p1, f_p2, f_p3, f_p4])
    x1_two = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(4, 4))(p2_two))
    x1_mul = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(4, 4))(p2_mul))
    # x = Lambda(my_upsampling, arguments={'img_w': K.int_shape(x)[1], 'img_h': K.int_shape(x)[2]})(x1)
    out_two = Conv2D(1, 1, activation='sigmoid', name = 'out_two')(x1_two)
    out_mul = Conv2D(num_class, 1, activation='softmax', name = 'out_mul')(x1_mul)
    model = Model(input=inputs, output=[out_mul, out_two])

    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'out_mul': lovasz_softmax,
                        'out_two': 'binary_crossentropy'},
                  metrics={'out_mul': ['accuracy'],
                           'out_two': ['accuracy'],
                           })

    if Falg_summary:
        model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = FarSeg(num_class=7, input_size=(256, 256, 4), Falg_summary=True)
