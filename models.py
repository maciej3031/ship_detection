import os

from keras import layers, models
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19

from config import CHECKPOINTS_DIR, NET_SCALING, GAUSSIAN_NOISE, INPUT_SHAPE
from resnet_152 import resnet152_model
from segmentation_models.segmentation_models import Unet


class BaseUNet:
    UPSAMPLE_MODE = 'SIMPLE'

    @staticmethod
    def upsample_conv(filters, kernel_size, strides, padding, **kwargs):
        return layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      **kwargs)

    @staticmethod
    def upsample_simple(filters, kernel_size, strides, padding, **kwargs):
        return layers.UpSampling2D(strides)

    def upsample(self, **kwargs):
        if self.UPSAMPLE_MODE == 'DECONV':
            return self.upsample_conv(**kwargs)
        else:
            return self.upsample_simple(**kwargs)


class ResNet34UnetV1(BaseUNet):
    MODEL_NAME = 'resnet34_unet_v1'
    FULL_RES_MODEL_NAME = 'resnet34_unet_full_res_v1'
    UPSAMPLE_MODE = 'DECONV'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        return Unet(backbone_name='resnet34', encoder_weights='imagenet', freeze_encoder=train_only_top)


class ResNet152UnetV2(BaseUNet):
    MODEL_NAME = 'resnet152_unet_v2'
    FULL_RES_MODEL_NAME = 'resnet152_unet_full_res_v2'
    UPSAMPLE_MODE = 'DECONV'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        return Unet(backbone_name='resnet152', encoder_weights='imagenet', freeze_encoder=train_only_top)


class ResNet152Unet(BaseUNet):
    MODEL_NAME = 'resnet152_unet'
    FULL_RES_MODEL_NAME = 'resnet152_unet_full_res'
    UPSAMPLE_MODE = 'DECONV'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        weights_path = os.path.join('checkpoints', 'resnet152_weights_tf.h5')
        base_model = resnet152_model(weights_path)

        if train_only_top:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = True

        resnet_1 = base_model.get_layer('conv1_relu').output
        resnet_2 = base_model.get_layer('res2c_relu').output
        resnet_3 = base_model.get_layer('res3b7_relu').output
        resnet_4 = base_model.get_layer('res4b35_relu').output
        resnet_5 = base_model.get_layer('res5c_relu').output

        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(resnet_5)

        u6 = self.upsample(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
        u6 = layers.concatenate([u6, resnet_4])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = self.upsample(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, resnet_3])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = self.upsample(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(c7)
        u8 = layers.Conv2D(128, (2, 2), strides=(1, 1), activation='relu', padding='valid')(u8)
        u8 = layers.concatenate([u8, resnet_2])
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = self.upsample(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', output_padding=(1, 1))(c8)
        u9 = layers.concatenate([u9, resnet_1], axis=3)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        u10 = self.upsample(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(c9)
        output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u10)

        return models.Model(inputs=base_model.input, outputs=[output])


class ResNet50UnetV1(BaseUNet):
    MODEL_NAME = 'resnet50_unet_v1'
    FULL_RES_MODEL_NAME = 'resnet50_unet_full_res_v1'
    UPSAMPLE_MODE = 'DECONV'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

        if train_only_top:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = True

        activation_1 = base_model.get_layer('activation_1').output
        activation_10 = base_model.get_layer('activation_10').output
        activation_22 = base_model.get_layer('activation_22').output
        activation_40 = base_model.get_layer('activation_40').output
        activation_49 = base_model.get_layer('activation_49').output

        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(base_model.output)

        u6 = layers.concatenate([c6, activation_49])
        u6 = self.upsample(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(u6)
        c6 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(u6)

        u7 = layers.concatenate([c6, activation_40])
        u7 = self.upsample(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(u7)
        c7 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(u7)

        u8 = layers.concatenate([c7, activation_22])
        u8 = self.upsample(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(u8)
        u8 = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='valid')(u8)
        c8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u8)

        u9 = layers.concatenate([c8, activation_10])
        u9 = self.upsample(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', output_padding=(1, 1))(u9)
        c9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u9)

        u10 = layers.concatenate([c9, activation_1], axis=3)
        u10 = self.upsample(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(u10)
        c10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u10)

        output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(c10)

        return models.Model(inputs=base_model.input, outputs=[output])


class VGG19UNetV1(BaseUNet):
    MODEL_NAME = 'vgg19_unet_v1'
    FULL_RES_MODEL_NAME = 'vgg19_unet_full_res_v1'
    UPSAMPLE_MODE = 'DECONV'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

    def get_classifier_model(self, train_only_top=False):
        if train_only_top:
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            for layer in self.base_model.layers:
                layer.trainable = True

        vgg = self.base_model.output
        flat = layers.Flatten()(vgg)
        output = layers.Dense(1, activation='sigmoid')(flat)

        return models.Model(inputs=self.base_model.input, outputs=[output])

    def get_model(self, train_only_top=False):

        if train_only_top:
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            for layer in self.base_model.layers:
                layer.trainable = True

        block1_conv4 = self.base_model.get_layer('block1_conv2').output
        block2_conv4 = self.base_model.get_layer('block2_conv2').output
        block3_conv4 = self.base_model.get_layer('block3_conv4').output
        block4_conv4 = self.base_model.get_layer('block4_conv4').output
        block5_conv4 = self.base_model.get_layer('block5_conv4').output
        block5_pool = self.base_model.get_layer('block5_pool').output

        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block5_pool)

        u6 = self.upsample(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
        u6 = layers.concatenate([u6, block5_conv4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)

        u7 = self.upsample(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, block4_conv4])
        c7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u7)

        u8 = self.upsample(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, block3_conv4])
        c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u8)

        u9 = self.upsample(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, block2_conv4])
        c9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u9)

        u10 = self.upsample(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(c9)
        u10 = layers.concatenate([u10, block1_conv4], axis=3)
        output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u10)

        return models.Model(inputs=self.base_model.input, outputs=[output])


class VGG19UNetV2(BaseUNet):
    MODEL_NAME = 'vgg19_unet_v2'
    FULL_RES_MODEL_NAME = 'vgg19_unet_full_res_v2'
    UPSAMPLE_MODE = 'SIMPLE'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

        if train_only_top:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = True

        block1_conv4 = base_model.get_layer('block1_conv2').output
        block2_conv4 = base_model.get_layer('block2_conv2').output
        block3_conv4 = base_model.get_layer('block3_conv4').output
        block4_conv4 = base_model.get_layer('block4_conv4').output
        block5_conv4 = base_model.get_layer('block5_conv4').output
        block5_pool = base_model.get_layer('block5_pool').output

        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(block5_pool)

        u6 = self.upsample(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
        c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
        cc1 = layers.concatenate([c6, block5_conv4])

        u7 = self.upsample(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(cc1)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        cc2 = layers.concatenate([c7, block4_conv4])

        u8 = self.upsample(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(cc2)
        c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c8)
        cc3 = layers.concatenate([c8, block3_conv4])

        u9 = self.upsample(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(cc3)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        cc4 = layers.concatenate([c9, block2_conv4])

        u10 = self.upsample(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(cc4)
        c10 = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(u10)
        c10 = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(c10)
        cc5 = layers.concatenate([c10, block1_conv4], axis=3)

        output = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(cc5)
        output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(output)

        return models.Model(inputs=base_model.input, outputs=[output])


class TernausNetV1(BaseUNet):
    MODEL_NAME = 'ternaus_net_v1'
    FULL_RES_MODEL_NAME = 'ternaus_net_v1_full_res'
    UPSAMPLE_MODE = 'DECONV'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        inp = layers.Input(INPUT_SHAPE, name='RGB_Input')

        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inp)
        p1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        p2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c4)

        c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
        p5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c5)

        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p5)

        u6 = self.upsample(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
        u6 = layers.concatenate([u6, c5])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)

        u7 = self.upsample(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c4])
        c7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u7)

        u8 = self.upsample(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c3])
        c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u8)

        u9 = self.upsample(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c2])
        c9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u9)

        u10 = self.upsample(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(c9)
        u10 = layers.concatenate([u10, c1], axis=3)
        output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u10)

        return models.Model(inputs=[inp], outputs=[output])


class TernausNetV2(BaseUNet):
    MODEL_NAME = 'ternaus_net_v2'
    FULL_RES_MODEL_NAME = 'ternaus_net_v2_full_res'
    UPSAMPLE_MODE = 'SIMPLE'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        inp = layers.Input(INPUT_SHAPE, name='RGB_Input')

        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inp)
        p1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        p2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c4)

        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        p5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c5)

        c6 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p5)

        u6 = self.upsample(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
        c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
        cc1 = layers.concatenate([c6, c5])

        u7 = self.upsample(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(cc1)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        cc2 = layers.concatenate([c7, c4])

        u8 = self.upsample(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(cc2)
        c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c8)
        cc3 = layers.concatenate([c8, c3])

        u9 = self.upsample(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(cc3)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        cc4 = layers.concatenate([c9, c2])

        u10 = self.upsample(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(cc4)
        c10 = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(u10)
        c10 = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(c10)
        cc5 = layers.concatenate([c10, c1], axis=3)

        output = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(cc5)
        output = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(output)

        return models.Model(inputs=[inp], outputs=[output])


class UNet(BaseUNet):
    MODEL_NAME = 'simple_unet'
    FULL_RES_MODEL_NAME = 'simple_unet_full_res'
    UPSAMPLE_MODE = 'SIMPLE'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
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


class UNet2(BaseUNet):
    MODEL_NAME = 'simple_unet2'
    FULL_RES_MODEL_NAME = 'simple_unet2_full_res'
    WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "{}_weights.best.hdf5".format(MODEL_NAME))
    FIT_HISTORY_PATH = os.path.join(CHECKPOINTS_DIR, "{}_history.csv".format(MODEL_NAME))

    def get_model(self, train_only_top=False):
        inp = layers.Input(INPUT_SHAPE, name='RGB_Input')

        c1 = layers.Conv2D(8, (3, 3), padding='same')(inp)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('relu')(c1)
        c2 = layers.Conv2D(8, (3, 3), padding='same')(c1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Activation('relu')(c2)
        d1 = layers.Dropout(0.1)(c2)

        p1 = layers.MaxPooling2D((3, 3), 2, padding='same')(d1)

        c3 = layers.Conv2D(16, (3, 3), padding='same')(p1)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Activation('relu')(c3)
        c4 = layers.Conv2D(16, (3, 3), padding='same')(c3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Activation('relu')(c4)
        d2 = layers.Dropout(0.1)(c4)

        p2 = layers.MaxPooling2D((3, 3), 2, padding='same')(d2)

        c5 = layers.Conv2D(32, (3, 3), padding='same')(p2)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Activation('relu')(c5)
        c6 = layers.Conv2D(32, (3, 3), padding='same')(c5)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Activation('relu')(c6)
        d3 = layers.Dropout(0.1)(c6)

        p3 = layers.MaxPooling2D((3, 3), 2, padding='same')(d3)

        c7 = layers.Conv2D(64, (3, 3), padding='same')(p3)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Activation('relu')(c7)
        c8 = layers.Conv2D(64, (3, 3), padding='same')(c7)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Activation('relu')(c8)
        d4 = layers.Dropout(0.1)(c8)

        u1 = layers.Conv2DTranspose(32, (6, 6), strides=(2, 2), padding='same', use_bias=False)(d4)
        u1 = layers.concatenate([u1, d3])

        c9 = layers.Conv2D(32, (3, 3), padding='same')(u1)
        c9 = layers.BatchNormalization()(c9)
        c9 = layers.Activation('relu')(c9)
        c10 = layers.Conv2D(32, (3, 3), padding='same')(c9)
        c10 = layers.BatchNormalization()(c10)
        c10 = layers.Activation('relu')(c10)
        d5 = layers.Dropout(0.1)(c10)

        u2 = layers.Conv2DTranspose(16, (6, 6), strides=(2, 2), padding='same', use_bias=False)(d5)
        u2 = layers.concatenate([u2, d2])

        c11 = layers.Conv2D(16, (3, 3), padding='same')(u2)
        c11 = layers.BatchNormalization()(c11)
        c11 = layers.Activation('relu')(c11)
        c12 = layers.Conv2D(16, (3, 3), padding='same')(c11)
        c12 = layers.BatchNormalization()(c12)
        c12 = layers.Activation('relu')(c12)
        d6 = layers.Dropout(0.1)(c12)

        u3 = layers.Conv2DTranspose(8, (6, 6), strides=(2, 2), padding='same', use_bias=False)(d6)
        u3 = layers.concatenate([u3, d1])

        c13 = layers.Conv2D(8, (3, 3), padding='same')(u3)
        c13 = layers.BatchNormalization()(c13)
        c13 = layers.Activation('relu')(c13)
        d7 = layers.Dropout(0.1)(c13)
        c14 = layers.Conv2D(8, (3, 3), padding='same')(d7)
        c14 = layers.BatchNormalization()(c14)
        c14 = layers.Activation('relu')(c14)

        output = layers.Conv2D(1, (1, 1), activation='sigmoid')(c14)

        return models.Model(inputs=[inp], outputs=[output])
