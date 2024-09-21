# python train_srgan.py --device tpu
# python train_srgan.py --device gpu

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))


tf.random.set_seed(42)

from app.data_preprocess import load_dataset
from app.esrgan import ESRGAN
from app.vgg import VGG
from app.esrgan_training import ESRGANTraining
from app import config
from app.losses import Losses
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.optimizers import Adam
from tensorflow.io.gfile import glob
import argparse
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="gpu",
    choices=["gpu", "tpu"], type=str,
    help="device to use for training (gpu or tpu)")
args = vars(ap.parse_args())


# check if we are using TPU
if args["device"] == "tpu":

    tpu = distribute.cluster_resolver.TPUClusterResolver() 
    experimental_connect_to_cluster(tpu)
    initialize_tpu_system(tpu)
    strategy = distribute.TPUStrategy(tpu)

    if config.TPU_BASE_TFR_PATH == "gs://<PATH_TO_GCS_BUCKET>/tfrecord":
        print("[INFO] not a valid GCS Bucket path...")
        sys.exit(0)
    

    tfrTrainPath = config.TPU_DIV2K_TFR_TRAIN_PATH
    pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.TPU_GENERATOR_MODEL

# otherwise
elif args["device"] == "gpu":

    strategy = distribute.MirroredStrategy()

    tfrTrainPath = config.GPU_DIV2K_TFR_TRAIN_PATH
    pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.GPU_GENERATOR_MODEL

else:

    print("[INFO] please enter a valid device argument...")
    sys.exit(0)



print("[INFO] number of accelerators: {}..."
    .format(strategy.num_replicas_in_sync))

print("[INFO] grabbing the train TFRecords...")
trainTfr = glob(tfrTrainPath +".tfrec")
print(tfrTrainPath +".tfrec")
print(trainTfr)

print("[INFO] creating train and test dataset...")
trainDs = load_dataset(filenames=trainTfr, train=True,
    batchSize=config.TRAIN_BATCH_SIZE * strategy.num_replicas_in_sync)


with strategy.scope():
    try:
        losses = Losses(numReplicas=strategy.num_replicas_in_sync)
        # initialize the generator, and compile it with Adam optimizer and
        # MSE loss
        generator = ESRGAN.generator(
            scalingFactor=config.SCALING_FACTOR,
            featureMaps=config.FEATURE_MAPS,
            residualBlocks=config.RESIDUAL_BLOCKS)
    except Exception as e:
        print("error 1")
        print(e)
    try:
        generator.compile(
            optimizer=Adam(learning_rate=config.PRETRAIN_LR),
            loss=losses.mse_loss)
    except Exception as e:
        print("error 2")
        print(e)


    print("[INFO] pretraining SRGAN generator...")
    try: 
        print(trainDs)
        generator.fit(trainDs, epochs=config.PRETRAIN_EPOCHS, steps_per_epoch=config.STEPS_PER_EPOCH)
    except Exception as e:
        print("-------------")
        print(e)