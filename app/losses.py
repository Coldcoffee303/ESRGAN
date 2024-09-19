from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossEntropy
from tensorflow.kersa.losses import Reduction
from tensorflow import reduce_mean

class Losses:
    def __init__(self, numReplicas):
        self.numReplicas = numReplicas
    
    def bce_loss(self, real, pred):
        BCE = BinaryCrossEntropy(reduction=Reduction.NONE)
        loss = BCE(real, pred)

        loss = reduce_mean(loss) * (1. / self.numReplicas)

        return loss
    
    def mse_loss(self, real, pred):
        MSE = MeanSquaredError(reduction=Reduction.NONE)
        loss = MSE(real, pred)

        loss = reduce_mean(loss) * (1. / self.numReplicas)

        return loss
    
