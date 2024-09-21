#imports
import tensorflow as tf

class ESRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks):

        inputs = tf.keras.Input((None, None, 3))
        xIn = tf.keras.layers.Rescaling(scale=(1.0 / 255.0), offset=0.0) (inputs)
        
        xIn = tf.keras.layers.Conv2D(featureMaps, 9, padding="same")(xIn)
        xIn = tf.keras.layers.PReLU(shared_axes=[1, 2])(xIn)

        x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same") (xIn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        xSkip = tf.keras.layers.Add() ([xIn, x])

        for _ in range(residualBlocks - 1):
            x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same")(xSkip)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
            x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            xSkip = tf.keras.layers.Add() ([xSkip, x])

        x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same")(xSkip)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([xIn, x])

        x = tf.keras.layers.Conv2D(featureMaps * (scalingFactor // 2), 3, padding="same")(x)
        x = DepthToSpace(2)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

        x = tf.keras.layers.Conv2D(featureMaps * scalingFactor, 3, padding="same")(x) 
        #x = DepthToSpace(4)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2]) (x)

        x = tf.keras.layers.Conv2D(3, 9, padding="same", activation="tanh")(x)
        x = tf.keras.layers.Rescaling(scale=127.5, offset=127.5)(x)

        generator = tf.keras.Model(inputs, x)

        return generator
    
    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):

        inputs = tf.keras.Input((None, None, 3))
        x = tf.keras.layers.Rescaling(scale=(1.0 / 127.5), offset=-1.0)(inputs)
        x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same")(x)

        x = tf.keras.layers.LeakyReLU(leakyAlpha)(x)

        x = tf.keras.layers.Conv2D(featureMaps, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(leakyAlpha)(x)

        for i in range(1, discBlocks):
            
            x = tf.keras.layers.Conv2D(featureMaps * (2 ** i), 3, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(leakyAlpha)(x)

            x = tf.keras.layers.Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(leakyAlpha)(x)

        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.LeakyReLU(leakyAlpha)(x)

        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        discriminator = tf.keras.Model(inputs, x)

        return discriminator



class DepthToSpace(tf.keras.layers.Layer):
    def __init__(self, block_size, **kwargs):
        super(DepthToSpace, self).__init__(**kwargs)
        self.block_size = block_size


    def call(self, inputs):       
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})

        return config