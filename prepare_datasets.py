import gc
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import VALID_IMG_COUNT, TRAIN_DIR, INPUT_DIR
from utils.data_utils import make_image_gen

gc.enable()


def load_dataset():
    return pd.read_csv(os.path.join(INPUT_DIR, "train_ship_segmentations.csv"))


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


def get_train_val_datasets(df, balanced_train_df):
    train_ids, valid_ids = train_test_split(balanced_train_df,
                                            test_size=VALID_IMG_COUNT,
                                            stratify=balanced_train_df['ships'])
    train_df = pd.merge(df, train_ids)
    valid_df = pd.merge(df, valid_ids)
    return train_df, valid_df


def split_validation_dataset(valid_df):
    valid_gen = make_image_gen(valid_df, VALID_IMG_COUNT)
    return next(valid_gen)
