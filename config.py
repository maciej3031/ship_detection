import os

INPUT_DIR = "input"
CHECKPOINTS_DIR = 'checkpoints'
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

AUGMENT_BRIGHTNESS = False
AUGMENTATION_DETAILS = dict(rotation_range=45,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.01,
                            zoom_range=[0.9, 1.25],
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='reflect',
                            data_format='channels_last')

LOAD_WEIGHTS = True
GAUSSIAN_NOISE = 0.1
INPUT_SHAPE = (384, 384, 3)
DEFAULT_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
VALID_IMG_COUNT = 1000
IMG_SCALING = (2, 2)
NET_SCALING = (1, 1)
BATCH_SIZE = 50

MAX_TRAIN_STEPS = 20
MAX_TRAIN_EPOCHS = 99
