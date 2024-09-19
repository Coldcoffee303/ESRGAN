from tensorflow.io import FixedLenFeature
from tensorflow.io import parse_single_example
from tensorflow.io import parse_tensor
from tensorflow.image import flip_left_right
from tensorflow.image import rot90
import tensorflow as tf

#autotune object
AUTO = tf.data.AUTOTUNE

def random_crop(lrImage, hrImage, hrCropSize=128, scale=4):
    lrCropSize = hrCropSize // 2
    lrImageShape = tf.shape(lrImage)[:2]

    lrW = tf.random.uniform(shape=(),
        maxval = lrImageShape[1] - lrCropSize + 1, dtype=tf.int32)
    
    lrH = tf.random.uniform(shape=(),
        maxval = lrImageShape[0] - lrCropSize + 1, dtype=tf.int32)
    
    hrW = lrW * scale
    hrH = lrH * scale

    lrImageCropped = tf.slice(lrImage, [lrH, lrW, 0],
        [(lrCropSize), (lrCropSize), 3])
    hrImageCropped = tf.slice(hrImage, [hrH, hrW, 0],
        [(hrCropSize), (hrCropSize), 3])
    
    return (lrImageCropped, hrImageCropped)

def get_center_crop(lrImage, hrImage, hrCropSize=128, scale=4):
    
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]

    lrW = lrImageShape[1] // 2
    lrH = lrImageShape[0] // 2

    hrW = lrW * scale
    hrH = lrH * scale

    lrImageCropped = tf.slice(lrImage, [lrH - (lrCropSize // 2),
        lrW - (lrCropSize // 2), 0], [lrCropSize, lrCropSize, 3])
    
    hrImageCropped = tf.slice(hrImage, [hrH - (hrCropSize // 2),
        hrW - (hrCropSize // 2), 0], [hrCropSize, hrCropSize, 3])
    
    return (lrImageCropped, hrImageCropped)

def random_flip(lrImage, hrImage):
    flipProb = tf.random.uniform(shape=(), maxval=1)
    (lrImage, hrImage) = tf.cond(flipProb < 0.5,
        lambda: (lrImage, hrImage),
        lambda: (flip_left_right(lrImage), flip_left_right(hrImage)))
    
    return (lrImage, hrImage)


def random_rotate(lrImage, hrImage):
    n = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)

    lrImage = rot90(lrImage, n)
    hrImage = rot90(hrImage, n)

    return (lrImage, hrImage)

def read_train_example(example):

    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }

    example = parse_single_example(example, feature)

    lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
    hrImage = parse_tensor(example["hr"], out_type=tf.uint8)

    (lrImage, hrImage) = random_crop(lrImage, hrImage)
    (lrImage, hrImage) = random_flip(lrImage, hrImage)
    (lrImage, hrImage) = random_rotate(lrImage, hrImage)

    lrImage = tf.reshape(lrImage, (32, 32, 3))
    hrImage = tf.reshape(hrImage, (128, 128, 3))
    
    return (lrImage, hrImage)

def read_test_example(example):

    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string)
    }

    example = parse_single_example(example, feature)

    lrImage = parse_tensor(example["lr"], out_type= tf.uint8)
    hrImage = parse_tensor(example["hr"], out_type= tf.uint8)
    
    (lrImage, hrImage) = get_center_crop(lrImage, hrImage)

    lrImage = tf.reshape(lrImage, (32, 32, 3))
    hrImage = tf.reshape(hrImage, (128, 128, 3))

    return (lrImage, hrImage)

def load_dataset(filenames, batchSize, train=False):

    dataset = tf.data.TFRecordDataset(filenames,
        num_parallel_reads=AUTO)
    
    if train:
        dataset = dataset.map(read_train_example,
            num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(read_test_example,
            num_parallel_calls=AUTO)
        
    dataset = ( dataset
        .shuffle(batchSize)
        .batch(batchSize)
        .repeat()
        .prefetch(AUTO)
    )

    return dataset
