import gc

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from prepare_datasets import get_train_val_datasets, drop_empty_images, get_unique_img_ids, split_validation_dataset,\
    load_dataset
from config import MAX_TRAIN_EPOCHS, MAX_TRAIN_STEPS, BATCH_SIZE, LOAD_WEIGHTS
from models import UNet
from utils.data_utils import create_aug_gen, make_image_gen
from utils.keras_utils import IoU, kaggle_IoU, bin_cross_and_IoU

gc.enable()  # memory is tight


def get_callbacks(seg_model):
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint(seg_model.WEIGHT_PATH,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=1,
                                       verbose=1,
                                       mode='min',
                                       min_delta=0.00001,
                                       cooldown=0,
                                       min_lr=1e-8)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=10)
    return [checkpoint, early, reduceLROnPlat, csv_logger]


def load_weight_if_possible(seg_model):
    try:
        seg_model.load_weights(seg_model.WEIGHT_PATH)
    except OSError:
        print('No file with weights available! Starting from scratch...')


def fit():
    df = load_dataset()
    unique_img_ids = get_unique_img_ids(df)
    balanced_train_df = drop_empty_images(unique_img_ids)
    train_df, valid_df = get_train_val_datasets(df, balanced_train_df)
    valid_x, valid_y = split_validation_dataset(valid_df)

    seg_model = UNet().get_model()

    callbacks_list = get_callbacks(seg_model)

    if LOAD_WEIGHTS:
        load_weight_if_possible(seg_model)

    seg_model.compile(optimizer=Adam(0.001, decay=0.000001), loss=bin_cross_and_IoU,
                      metrics=['binary_accuracy', IoU, kaggle_IoU])

    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0] // BATCH_SIZE)
    aug_gen = create_aug_gen(make_image_gen(train_df))
    loss_hist = [seg_model.fit_generator(aug_gen,
                                         steps_per_epoch=step_count,
                                         epochs=MAX_TRAIN_EPOCHS,
                                         validation_data=(valid_x, valid_y),
                                         callbacks=callbacks_list,
                                         workers=1)]
    return loss_hist


if __name__ == '__main__':
    while True:
        loss_history = fit()
        if np.min([mh.history['val_loss'] for mh in loss_history]) < 4.0:
            break
