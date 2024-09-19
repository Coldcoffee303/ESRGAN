# python inference.py --device gpu
# python inference.py --device tpu

from app.data_preprocess import load_dataset
from app.utils import zoom_into_images
from app import config
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.io.gfile import glob
from matplotlib.pyplot import subplots
import argparse
import sys
import os

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

    tfrTestPath = config.TPU_DIV2K_TFR_TEST_PATH
    pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.TPU_GENERATOR_MODEL

# otherwise
elif args["device"] == "gpu":

    strategy = distribute.MirroredStrategy()

    tfrTestPath = config.GPU_DIV2K_TFR_TEST_PATH
    pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.GPU_GENERATOR_MODEL

# else
else:
    print("[INFO] please enter a valid device argument...")
    sys.exit(0)


# get the dataset
print("[INFO] loading the test dataset...")
testTfr = glob(tfrTestPath + "/*.tfrec")
testDs = load_dataset(testTfr, config.INFER_BATCH_SIZE, train=False)

(lrImage, hrImage) = next(iter(testDs))

with strategy.scope(): 
 
    print("[INFO] loading the pre-trained and fully trained esrgan model...")
    esrganPreGen = load_model(pretrainedGenPath, compile=False)
    esrganGen = load_model(genPath, compile=False)
    

    print("[INFO] making predictions with pre-trained and fully trained esrgan model...")
    esrganPreGenPred = esrganPreGen.predict(lrImage)
    esrganGenPred = esrganGen.predict(lrImage)


print("[INFO] plotting the esrgan predictions...")
(fig, axes) = subplots(nrows=config.INFER_BATCH_SIZE, ncols=4,
    figsize=(50, 50))

for (ax, lowRes, srPreIm, esrganIm, highRes) in zip(axes, lrImage,
        esrganPreGenPred, esrganGenPred, hrImage):

    ax[0].imshow(array_to_img(lowRes))
    ax[0].set_title("Low Resolution Image")

    ax[1].imshow(array_to_img(srPreIm))
    ax[1].set_title("esrgan Pretrained")

    ax[2].imshow(array_to_img(esrganIm))
    ax[2].set_title("esrgan")

    ax[3].imshow(array_to_img(highRes))
    ax[3].set_title("High Resolution Image")


if not os.path.exists(config.BASE_IMAGE_PATH):
    os.makedirs(config.BASE_IMAGE_PATH)

print("[INFO] saving the esrgan predictions to disk...")
fig.savefig(config.GRID_IMAGE_PATH)

zoom_into_images(esrganPreGenPred[0], "esrgan Pretrained")
zoom_into_images(esrganGenPred[0], "esrgan")