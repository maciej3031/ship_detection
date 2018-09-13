import gc
import os

import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, IMG_SCALING, AUGMENTATION_DETAILS
from config import VALID_IMG_COUNT, TRAIN_DIR, INPUT_DIR, SAMPLES_PER_GROUP
from utils.rle_utils import masks_as_image

gc.enable()  # memory is tight


def make_image_gen(in_df, batch_size=BATCH_SIZE, img_scaling=IMG_SCALING, classify=False):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb, out_mask, labels = [], [], []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_DIR, c_img_id)
            c_img = cv2.imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            label = [in_df[in_df.ImageId == c_img_id]['has_ship'].iloc[0]]
            if IMG_SCALING is not None:
                c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]
            out_rgb.append(c_img)
            out_mask.append(c_mask)
            labels.append(label)
            if len(out_rgb) >= batch_size:
                if not classify:
                    yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                else:
                    yield np.stack(out_rgb, 0) / 255.0, np.array(labels)
                out_rgb, out_mask, labels = [], [], []
                gc.collect()


def create_aug_gen(in_gen, seed=None, classify=False):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    data_gen = ImageDataGenerator(**AUGMENTATION_DETAILS)
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = data_gen.flow(255 * in_x,
                            batch_size=in_x.shape[0],
                            seed=seed,
                            shuffle=True)
        if not classify:
            g_y = data_gen.flow(in_y,
                                batch_size=in_x.shape[0],
                                seed=seed,
                                shuffle=True)

        yield next(g_x) / 255.0, next(g_y) if not classify else in_y
        gc.collect()


def load_dataset():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train_ship_segmentations.csv"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test_ship_segmentations.csv"))
    return pd.concat([train, test])


def get_unique_img_ids(df):
    df['ships'] = df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)

    # some files are too small/corrupt
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
        lambda c_img_id: os.stat(os.path.join(TRAIN_DIR, c_img_id)).st_size / 1024)
    return unique_img_ids[unique_img_ids['file_size_kb'] > 50]  # keep only +50kb files


def drop_empty_images(unique_img_ids):
    return unique_img_ids[unique_img_ids.ships != 0]


def get_balanced_dataset(unique_img_ids):
    return unique_img_ids.groupby('ships').apply(
        lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)


def get_train_val_datasets(df, balanced_train_df):
    train_ids, valid_ids = train_test_split(balanced_train_df,
                                            test_size=VALID_IMG_COUNT,
                                            stratify=balanced_train_df['ships'])

    train_df = pd.merge(df, train_ids, on='ImageId')
    valid_df = pd.merge(df, valid_ids, on='ImageId')

    return train_df, valid_df


def split_validation_dataset(valid_df, classify):
    valid_gen = make_image_gen(valid_df, VALID_IMG_COUNT, classify=classify)
    return next(valid_gen)
