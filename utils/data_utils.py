import gc
import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from config import BATCH_SIZE, IMG_SCALING, AUGMENT_BRIGHTNESS, TRAIN_DIR, AUGMENTATION_DETAILS
from utils.rle_utils import masks_as_image

gc.enable()  # memory is tight


def make_image_gen(in_df, batch_size=BATCH_SIZE, img_scaling=IMG_SCALING):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_DIR, c_img_id)
            c_img = cv2.imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            if IMG_SCALING is not None:
                c_img = c_img[::img_scaling[0], ::img_scaling[1]]
                c_mask = c_mask[::img_scaling[0], ::img_scaling[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []
                gc.collect()


def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    # brightness can be problematic since it seems to change the labels differently from the images
    if AUGMENT_BRIGHTNESS:
        AUGMENTATION_DETAILS['brightness_range'] = [0.5, 1.5]
    image_gen = ImageDataGenerator(**AUGMENTATION_DETAILS)

    if AUGMENT_BRIGHTNESS:
        AUGMENTATION_DETAILS.pop('brightness_range')
    label_gen = ImageDataGenerator(**AUGMENTATION_DETAILS)
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)
        gc.collect()
