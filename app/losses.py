import tensorflow as tf
from tensorflow import reduce_mean

class Losses:
    def __init__(self, numReplicas):
        self.numReplicas = numReplicas
    
    def bce_loss(self, real, pred):
        BCE = tf.keras.losses.BinaryCrossEntropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = BCE(real, pred)

        loss = reduce_mean(loss) * (1. / self.numReplicas)

        return loss
    
    def mse_loss(self, real, pred):
        MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss = MSE(real, pred)

        loss = reduce_mean(loss) * (1. / self.numReplicas)

        return loss
    
