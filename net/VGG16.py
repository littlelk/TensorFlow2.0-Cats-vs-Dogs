import tensorflow as tf


class VGG16(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(64,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                      strides=2,
                                      padding='same'),
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(128,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                      strides=2,
                                      padding='same')
        ])
        self.conv_block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(256,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                      strides=2,
                                      padding='same')
        ])
        self.conv_block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(512,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(512,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        ])

        self.conv_block5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(512,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(512,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        ])
        self.fc_block = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        out = self.fc_block(x)
        return out
