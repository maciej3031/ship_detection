import os

from keras import layers, models

from config import CHECKPOINTS_DIR, NET_SCALING, GAUSSIAN_NOISE, INPUT_SHAPE


class UNet:
    MODEL_NAME = 'simple_unet'
    FULL_RES_MODEL_NAME = 'simple_unet_full_res'
    UPSAMPLE_MODE = 'SIMPLE'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "_history.csv".format(MODEL_NAME))

    @staticmethod
    def upsample_conv(filters, kernel_size, strides, padding):
        return layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

    @staticmethod
    def upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)

    def upsample(self, **kwargs):
        if self.UPSAMPLE_MODE == 'DECONV':
            return self.upsample_conv(**kwargs)
        else:
            return self.upsample_simple(**kwargs)

    def get_model(self):
        input_img = layers.Input(INPUT_SHAPE, name='RGB_Input')
        pp_in_layer = input_img

        if NET_SCALING is not None:
            pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

        pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
        pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = self.upsample(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = self.upsample(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = self.upsample(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = self.upsample(filters=8, kernel_size=(2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
        # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
        if NET_SCALING is not None:
            d = layers.UpSampling2D(NET_SCALING)(d)

        return models.Model(inputs=[input_img], outputs=[d])


class UNet2:
    MODEL_NAME = 'simple_unet2'
    FULL_RES_MODEL_NAME = 'simple_unet2_full_res'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "_history.csv".format(MODEL_NAME))

    def get_model(self):

        inp = layers.Input(INPUT_SHAPE, name='RGB_Input')

        c1 = layers.Conv2D(8, (3, 3), padding='same')(inp)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('relu')(c1)
        c2 = layers.Conv2D(8, (3, 3), padding='same')(c1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Activation('relu')(c2)
        d1 = layers.Dropout(0.1)(c2)

        c3 = layers.Conv2D(16, (3, 3), padding='same')(d1)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Activation('relu')(c3)
        c4 = layers.Conv2D(16, (3, 3), padding='same')(c3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Activation('relu')(c4)
        d2 = layers.Dropout(0.1)(c4)

        p1 = layers.MaxPooling2D((3, 3), 2)(d2)

        c5 = layers.Conv2D(32, (3, 3), padding='same')(p1)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Activation('relu')(c5)
        c6 = layers.Conv2D(32, (3, 3), padding='same')(c5)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Activation('relu')(c6)
        d3 = layers.Dropout(0.1)(c6)

        p2 = layers.MaxPooling2D((3, 3), 2)(d3)

        c7 = layers.Conv2D(64, (3, 3), padding='same')(p2)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Activation('relu')(c7)
        c8 = layers.Conv2D(64, (3, 3), padding='same')(c7)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Activation('relu')(c8)
        d4 = layers.Dropout(0.1)(c8)

        u1 = layers.Conv2DTranspose(32, (6, 6), strides=(2, 2), padding='same', use_bias=True)(d4)
        u1 = layers.concatenate([u1, d3])

        c9 = layers.Conv2D(32, (3, 3), padding='same')(u1)
        c9 = layers.BatchNormalization()(c9)
        c9 = layers.Activation('relu')(c9)
        c10 = layers.Conv2D(32, (3, 3), padding='same')(c9)
        c10 = layers.BatchNormalization()(c10)
        c10 = layers.Activation('relu')(c10)
        d5 = layers.Dropout(0.1)(c10)

        u2 = layers.Conv2DTranspose(16, (6, 6), strides=(2, 2), padding='same', use_bias=True)(d5)
        u2 = layers.concatenate([u2, d2])

        c11 = layers.Conv2D(16, (3, 3), padding='same')(u2)
        c11 = layers.BatchNormalization()(c11)
        c11 = layers.Activation('relu')(c11)
        c12 = layers.Conv2D(16, (3, 3), padding='same')(c11)
        c12 = layers.BatchNormalization()(c12)
        c12 = layers.Activation('relu')(c12)
        d6 = layers.Dropout(0.1)(c12)

        c13 = layers.Conv2D(8, (3, 3), padding='same')(d6)
        c13 = layers.BatchNormalization()(c13)
        c13 = layers.Activation('relu')(c13)
        d7 = layers.Dropout(0.1)(c13)
        c14 = layers.Conv2D(8, (3, 3), padding='same')(d7)
        c14 = layers.BatchNormalization()(c14)
        c14 = layers.Activation('relu')(c14)

        output = layers.Conv2D(1, (1, 1), activation='sigmoid')(c14)

        return models.Model(inputs=[inp], outputs=[output])