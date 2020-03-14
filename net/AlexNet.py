import tensorflow as tf


class AlexNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(96, 11, strides=4, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same')

        self.conv3 = tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc3 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = tf.nn.local_response_normalization(x, 2, 2, 1e-4, 0.75)

        x = self.conv2(x)
        x = self.pool2(x)
        x = tf.nn.local_response_normalization(x, 2, 2, 1e-4, 0.75)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = tf.nn.dropout(x, 0.5)
        x = self.fc2(x)
        x = tf.nn.dropout(x, 0.5)
        out = self.fc3(x)

        return out
