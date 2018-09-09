import argparse
import gc
import warnings

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from config import MAX_TRAIN_EPOCHS, MAX_TRAIN_STEPS, BATCH_SIZE, LEARNING_RATE, INITIAL_TRAINING, LOAD_WEIGHTS, \
    DROP_EMPTY_IMAGES, CLASSIFY_MODE, INITIAL_TRAINING_EPOCHS
from models import UNet, UNet2, TernausNetV1, TernausNetV2, VGG19UNetV1, VGG19UNetV2, ResNet50UnetV1, ResNet152Unet, \
    ResNet152UnetV2, ResNet34UnetV1
from segmentation_models.segmentation_models.utils import set_trainable
from utils.data_utils import create_aug_gen, make_image_gen, get_train_val_datasets, drop_empty_images, \
    get_unique_img_ids, split_validation_dataset, load_dataset, get_balanced_dataset
from utils.keras_utils import IoU, kaggle_IoU, focal_loss_and_dice_loss


def warn(*args, **kwargs):
    pass


warnings.warn = warn
gc.enable()  # memory is tight

AVAILABLE_MODELS = {model.MODEL_NAME: model for model in [UNet2(), UNet(), TernausNetV1(), TernausNetV2(),
                                                          VGG19UNetV1(), VGG19UNetV2(), ResNet50UnetV1(),
                                                          ResNet152Unet(), ResNet152UnetV2(), ResNet34UnetV1()]}


def get_callbacks(seg_model):
    csv_logger = CSVLogger(seg_model.FIT_HISTORY_PATH, append=False, separator=';')
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
                                       min_delta=0.0001,
                                       cooldown=0,
                                       min_lr=1e-10)

    early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)
    return [checkpoint, early, reduceLROnPlat, csv_logger]


def load_weight_if_possible(seg_model, keras_model):
    try:
        keras_model.load_weights(seg_model.WEIGHT_PATH)
        print('Weights loaded!')
    except OSError:
        print('No file with weights available! Starting from scratch...')


def load_data(args):
    df = load_dataset()
    unique_img_ids = get_unique_img_ids(df)
    balanced_dataset = get_balanced_dataset(unique_img_ids)

    if args['drop_empty_images']:
        balanced_dataset = drop_empty_images(balanced_dataset)

    train_df, valid_df = get_train_val_datasets(df, balanced_dataset)
    valid_x, valid_y = split_validation_dataset(valid_df, classify=args['classify_mode'])

    return train_df, valid_x, valid_y


def fit(train_df, valid_x, valid_y, args):
    seg_model = AVAILABLE_MODELS.get(args['model_name'])
    classify = args['classify_mode']
    if classify:
        keras_model = seg_model.get_classifier_model(args['initial_training'])
    else:
        keras_model = seg_model.get_model(args['initial_training'])

    print(keras_model.summary())

    callbacks_list = get_callbacks(seg_model)

    if args['load_weights']:
        load_weight_if_possible(seg_model, keras_model)

    keras_model.compile(optimizer=Adam(args['learning_rate'], decay=0.000001),
                        loss=focal_loss_and_dice_loss,
                        metrics=['accuracy', IoU, kaggle_IoU])

    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0] // BATCH_SIZE)
    aug_gen = create_aug_gen(make_image_gen(train_df, classify=classify), classify=classify)

    if args['initial_training']:
        # pretrain model decoder
        keras_model.fit_generator(aug_gen,
                                  steps_per_epoch=step_count,
                                  epochs=INITIAL_TRAINING_EPOCHS,
                                  validation_data=(valid_x, valid_y),
                                  callbacks=callbacks_list,
                                  workers=1,
                                  max_queue_size=1)

    # release all layers for training
    set_trainable(keras_model)  # set all layers trainable and recompile model

    # continue training
    loss_hist = [keras_model.fit_generator(aug_gen,
                                           steps_per_epoch=step_count,
                                           epochs=MAX_TRAIN_EPOCHS,
                                           validation_data=(valid_x, valid_y),
                                           callbacks=callbacks_list,
                                           workers=1,
                                           max_queue_size=1)]
    return loss_hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-mn', '--model_name', default='simple_unet')
    ap.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE)
    ap.add_argument('-cl', '--classify_mode', action='store_true', default=CLASSIFY_MODE)
    ap.add_argument('-lw', '--load_weights', action='store_true', default=LOAD_WEIGHTS)
    ap.add_argument('-it', '--initial_training', action='store_true', default=INITIAL_TRAINING)
    ap.add_argument('-de', '--drop_empty_images', action='store_true', default=DROP_EMPTY_IMAGES)

    args = vars(ap.parse_args())

    train_df, valid_x, valid_y = load_data(args)

    while True:
        loss_history = fit(train_df, valid_x, valid_y, args)
        gc.collect()
        if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.2:
            break
