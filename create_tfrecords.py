import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# HELPER FUNCTIONS TO CONVERT DATA

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# FUNCTION TO SERIALIZE EACH EXAMPLE AND RETURN IT
def serialize_example(lr_image, hr_image):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'lr': _bytes_feature(lr_image),  # Low-resolution image
        'hr': _bytes_feature(hr_image),  # High-resolution image
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# FUNCTION THAT WRITES IMAGES TO TFRECORD FILE
def write_tfrecords(filenames, tfrecord_path):
    """
    Given image file paths, write low-res and high-res versions to TFRecord.
    """
    writer = tf.io.TFRecordWriter(tfrecord_path)
    
    for image_path in filenames:
        # Read and process the high-resolution image
        hr_image = Image.open(image_path).convert("RGB")
        hr_image = hr_image.resize((64, 64))  # Assuming 64x64 is the desired high-res size
        hr_image = np.array(hr_image)
        hr_image = tf.io.serialize_tensor(hr_image).numpy()  # Convert to bytes

        # Create low-resolution version
        lr_image = Image.open(image_path).convert("RGB")
        lr_image = lr_image.resize((32, 32))  # Assuming 32x32 is the desired low-res size
        lr_image = np.array(lr_image)
        lr_image = tf.io.serialize_tensor(lr_image).numpy()  # Convert to bytes

        # Serialize the example and write it to the TFRecord file
        example = serialize_example(lr_image, hr_image)
        writer.write(example)

    writer.close()
    print(f"TFRecord saved at {tfrecord_path}")


# SPLIT IMAGES INTO TRAIN AND TEST, THEN WRITE TO TFRECORDS
def process_images(image_folder):
    # Get all image paths from the folder
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
    
    # Split the dataset into training and testing
    train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    # Write to train.tfrec and test.tfrec
    write_tfrecords(train_paths, "train.tfrec")
    write_tfrecords(test_paths, "test.tfrec")


# FUNCTION CALL
image_folder = "Data\Residential"  # Update this to your folder containing JPG images
process_images(image_folder)
