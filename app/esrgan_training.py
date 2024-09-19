from tensorflow.keras import Model
from tensorflow import GradientTape, concat, zeros, ones
import tensorflow as tf

class ESRGANTraining(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batchSize = batchSize

    def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
        super().compile()

        self.gOptimizer = gOptimizer
        self.dOptimizer = dOptimizer

        self.bceLoss = bceLoss
        self.mseLoss = mseLoss

    def train_step(self, images):

        (lrImages, hrImages) = images
        lrImages = tf.cast(lrImages, tf.float32)
        hrImages = tf.cast(hrImages, tf.float32)

        srImages = self.generator(lrImages)

        combinedImages = concat([srImages, hrImages], axis=0)

        labels = concat([zeros((self.batchSize, 1)), ones((self.batchSize, 1))], axis=0)

        with GradientTape() as tape:
            predictions = self.discriminator(combinedImages)

            dLoss = self.bceLoss(labels, prediction)

        grads = tape.gradient(dLoss, self.discriminator.trainable_variables)

        self.dOptimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )

        misleadingLabels = ones((self.batchSize, 1))

        with GradientTape() as tape:
            fakeImages = self.generator(lrImages)

            predictions = self.discriminator(fakeImages)

            gLoss = 1e-3 * self.bceloss(misleadingLabels, predictions)

            srVgg = tf.keras.applications.vgg19.preprocess_input(fakeImages)
            srVgg = self.vgg(srVgg) / 12.75
            hrVgg = tf.keras.applications.vgg19.preprocess_input(hrImages)
            hrVgg = self.vgg(hrVgg) / 12.75

            percLoss = self.mseloss(hrVgg, srVgg)
            
            gTotalLoss = gLoss + percLoss

        grads = tape.gradient(gTotalLoss, self.generator.trainable_variables)

        self.gOptimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return { "dLoss": dLoss, "gTotalLoss": gTotalLoss, "gLoss": gLoss, "percLoss": percLoss }