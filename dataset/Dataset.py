import tensorflow as tf
import numpy as np
import logging
import os
import random
import matplotlib.pyplot as plt
import cv2

class Dataset:
    def __init__(self, data_dir, tfrecord, image_size):
        self.image_size = image_size
        self.data_dir = data_dir
        self.tfrecord_file = tfrecord
        filenames = [
            os.path.join(self.data_dir, filename)
            for filename in os.listdir(self.data_dir) if filename.endswith('.jpg')
        ]
        random.shuffle(filenames)
        labels = [
            0 if filename.split('\\')[-1][:3] == 'cat' else 1
            for filename in filenames
        ]

        self._to_tfrecord(filenames, labels)

    def _to_tfrecord(self, filenames, labels=None):
        '''
            convert data to TFRecord format
        '''
        # if os.path.isfile(self.tfrecord_file):
        #     logging.info(
        #         'TFRecord: {} has been generated!'.format(self.tfrecord_file))
        #     return
        with tf.io.TFRecordWriter(self.tfrecord_file) as writer:
            for filename, label in zip(filenames, labels):
                image = open(filename, 'rb').read()
                feature = {
                    'image':
                        tf.train.Feature(bytes_list=tf.train.BytesList(
                            value=[image])),
                    'label':
                        tf.train.Feature(int64_list=tf.train.Int64List(
                            value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(
                    feature=feature))
                writer.write(
                    example.SerializeToString())  # serialize to TFRecord File
        # serialize to TFRecord File

    def _parse_example(self, example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        feature_dict = tf.io.parse_single_example(example, feature_description)
        image_array = tf.io.decode_jpeg(feature_dict['image'])
        # image_array = image_array[tf.newaxis, ...]
        image_array = tf.image.resize(image_array, [224, 224], method='bilinear')
        image_array = tf.cast(image_array, tf.float32)

        feature_dict['image'] = image_array / 255.0

        return feature_dict['image'], feature_dict['label']

    def _from_tfrecord(self):
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_file)

        self.dataset = raw_dataset.map(self._parse_example)

        return self.dataset

# data_root = 'D:\\Data\\cat_vs_dog\\train'
# dataset = Dataset(data_root)
# dataset._to_tfrecord(os.path.join(data_root, 'train.tfrecord'))
# ds = dataset._from_tfrecord(os.path.join(data_root, 'train.tfrecord'))

# for image, label in ds:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy())
#     plt.show()
#     break
