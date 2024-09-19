import os

DATASET = "paste dataset path here"

#shard & batch size configs
SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 8

#dataset configs
HR_SHAPE = [128, 128, 3]
LR_SHAPE = [32, 32, 3]
SCALING_FACTOR = 4

#GAN Model configs
FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 16
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4

#training configs
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-5
PRETRAIN_EPOCHS = 2000
FINETUNE_EPOCHS = 2000
STEPS_PER_EPOCH = 10

#dataset path
BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")

#tfrecords for GPU training
GPU_BASE_TFR_PATH = "tfrecord"
GPU_DIV2K_TFR_TRAIN_PATH = os.path.join(GPU_BASE_TFR_PATH, "train")
GPU_DIV2K_TFR_TEST_PATH = os.path.join(GPU_BASE_TFR_PATH, "test")

#tfrecords for TPU training
TPU_BASE_TFR_PATH = "gs://<PATH_TO_GCS_BUCKET>/tfrecord"
TPU_DIV2K_TFR_TRAIN_PATH = os.path.join(TPU_BASE_TFR_PATH, "train")
TPU_DIV2K_TFR_TEST_PATH = os.path.join(TPU_BASE_TFR_PATH, "test")

#output directory path
BASE_OUTPUT_PATH = "outputs"

#GPU training ESRGAN model paths
GPU_PRETRAINED_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models", "pretrained_generator")
GPU_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models", "generator")

#TPU training ESRGAN model paths
TPU_OUTPUT_PATH = "gs://<PATH_TO_GCS_BUCKET>/outputs"
TPU_PRETRAINED_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH, "models", "pretrained_generator")
TPU_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH, "models", "generator")

#path to inferered images & grid images
BASE_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_IMAGE_PATH, "grid.png")
