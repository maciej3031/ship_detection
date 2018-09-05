import os

INPUT_DIR = "input"
CHECKPOINTS_DIR = 'checkpoints'
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

AUGMENTATION_DETAILS = dict(rotation_range=90,
                            shear_range=0.01,
                            zoom_range=[0.9, 1.1],
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='reflect',
                            data_format='channels_last')

SAMPLES_PER_GROUP = 20000
GAUSSIAN_NOISE = 0.1
INPUT_SHAPE = (256, 256, 3)
DEFAULT_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
VALID_IMG_COUNT = 3000
IMG_SCALING = (3, 3)
NET_SCALING = (1, 1)
BATCH_SIZE = 10

MAX_TRAIN_STEPS = 200
MAX_TRAIN_EPOCHS = 99

FOCAL_LOSS_ALPHA = 0.5
FOCAL_LOSS_GAMMA = 4.0