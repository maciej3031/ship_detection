import gc

import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy

from config import DEFAULT_THRESHOLDS

gc.enable()  # memory is tight


def kaggle_IoU(y_true, y_pred):
    gc.collect()
    score_list = []
    for thresh in DEFAULT_THRESHOLDS:
        iou = IoU(y_true, y_pred, thresh=thresh)
        score_list.append(iou)

    return K.mean(tf.convert_to_tensor(score_list))


def IoU(y_true, y_pred, thresh=0.5, eps=1e-6):
    y_pred = tf.to_float(tf.to_int32(y_pred > thresh))
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + eps) / (union + eps)


def dice_loss(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return 1 - (2.0 * intersection + smooth) / (union + smooth)


def bin_cross_and_IoU(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - 10 * K.log(1 - dice_loss(y_true, y_pred))
