#imports
from tensorflow.keras.layers import BatchNormalization, GlobalAvgPool2D, LeakyReLU
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, PReLU, Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model, Input

class ESRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks):

        inputs = Input((None, None, 3))
        xIn = Rescaling(scale=(1.0 / 255.0), offset=0.0) (inputs)

        x = Conv2D(featureMaps, 3, padding="same") (xIn)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1,2])(x)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        xSkip = Add() ([xIn, x])

        for _ in range(residualBlocks - 1):
            x = Conv2D(featureMaps, 3, padding="same")(xSkip)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1,2])(x)
            x = Conv2D(featureMaps, 3, padding="same")(x)
            x = BatchNormalization()(x)
            xSkip = Add() ([xSkip, x])

        x = Conv2D(featureMaps, 3, padding="same")(xSkip)
        x = BatchNormalization()(x)
        x = Add()([xIn, x])

        x = Conv2D(featureMaps * (scalingFactor // 2), 3, padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes = [1,2])(x)

        x = Conv2D(featureMaps * scalingFactor, 3, padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2]) (x)

        x = Conv2D(3, 9, padding="same", activation="tanh")(x)
        x = Rescaling(scale=127.5, offset=127.5)(x)

        generator = Model(inputs, x)

        return generator
    
    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):

        inputs = Input((None, None, 3))
        x = Rescaling(scale=(1.0 / 127.5), offset=-1.0)(inputs)
        x = Conv2D(featureMaps, 3, padding="same")(x)

        x = LeakyReLU(leakyAlpha)(x)

        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leakyAlpha)(x)

        for i in range(1, discBlocks):
            
            x = Conv2D(featureMaps * (2 ** i), 3, strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

            x = Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

            x = Dense(1, activation="sigmoid")(x)

            discriminator = Model(inputs, x)

            return discriminator
        
    
