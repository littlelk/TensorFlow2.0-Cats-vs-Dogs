import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import imageio
import re


from net.VGG16 import VGG16
from net.AlexNet import AlexNet
from dataset.Dataset import Dataset


parser = argparse.ArgumentParser(description='Process some integers setting')
parser.add_argument('--phase',
                    default='train',
                    help='train or predict')
parser.add_argument('--train_data_dir',
                    default='D:\\Data\\cat_vs_dog\\train',
                    help='train data dir')
parser.add_argument('--test_data_dir',
                    default='D:\\Data\\cat_vs_dog\\test',
                    help='test data dir')
parser.add_argument('--train_tfrecord',
                    default='D:\\Data\\cat_vs_dog\\train.tfrecord',
                    help='the path of tfrecord')
parser.add_argument('--test_tfrecord',
                    default='D:\\Data\\cat_vs_dog\\test.tfrecord',
                    help='the path of tfrecord')
parser.add_argument('--logdir',
                    default='logdir',
                    help='the path for logs')
parser.add_argument('--checkpoint',
                    default='AlexNet_checkpoint.h5',
                    help='the path for checkpoint')
parser.add_argument('--image_size',
                    default=[224, 224],
                    help='the size of resized image')

parser.add_argument('--epochs', default=100, help='number of epochs to train')
parser.add_argument('--batch_size',
                    default=64,
                    help='the size of data to process one time')
parser.add_argument('--learning_rate',
                    default=0.001,
                    help='learning rate for bp')
args = parser.parse_args()

# (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
# train_data = np.expand_dims(train_data.astype(np.float32) / 255., axis=-1)
# train_label = train_label.astype(np.int32)
# train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))
# train_ds = train_ds.shuffle(buffer_size=100)
# train_ds = train_ds.batch(args.batch_size)
# train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def train():
    dataset = Dataset(args.train_data_dir, args.train_tfrecord, args.image_size)
    train_ds = dataset._from_tfrecord()
    train_ds = train_ds.shuffle(buffer_size=5000)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 定义网络
    model = AlexNet()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    # Windows上的bug，必须要使用os.path.join封装一下
    logdir = os.path.join(args.logdir)
    checkpoint = os.path.join(logdir, args.checkpoint)
    print(logdir)
    print(checkpoint)
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-5)
    ]

    model.fit(train_ds, epochs=args.epochs, callbacks=callbacks)

    # model.save_weights('vgg16_weight.h5')


def predict():
    model = AlexNet()

    model.build(input_shape=(None, 224, 224, 3))

    logdir = os.path.join(args.logdir)
    checkpoint = os.path.join(logdir, args.checkpoint)

    model.load_weights(checkpoint)
    # dataset = Dataset(args.test_data_dir, args.test_tfrecord, args.image_size, args)
    # train_ds = dataset._from_tfrecord()
    # train_ds = train_ds.batch(args.batch_size)
    filenames = [os.path.join(args.test_data_dir, filename) for filename in os.listdir(args.test_data_dir)]

    filenames = sorted(filenames, key=lambda x: int((x.split('\\')[-1]).split('.')[0]))
    # xx = cv2.resize(cv2.imread('D:\\Data\\cat_vs_dog\\test\\1.jpg'), (224, 224))

    lens = len(filenames)

    results = {'id': list(np.arange(lens)), 'label': []}

    for idx, filename in enumerate(filenames):
        image = cv2.imread(filename).astype(np.float32) / 255.
        image_resized = cv2.resize(image, tuple(args.image_size))
        image_expanded = image_resized[np.newaxis, ...]

        predict_dict = model.predict(image_expanded)

        results['label'].append(np.argmax(predict_dict, axis=-1)[0])

    result_frame = pd.DataFrame(results, index=None)
    result_frame.to_csv('submission.csv')


if __name__ == "__main__":
    if args.phase == 'train':
        train()
    elif args.phase == 'predict':
        predict()
