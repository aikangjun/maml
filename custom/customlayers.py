from custom import *
import tensorflow as tf

class ConvBNLeaky(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str = 'SAME',
                 groups:int=2,
                 **kwargs):
        super(ConvBNLeaky, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  groups=groups)
        self.bn = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple = (3, 3),
                 strides: tuple = (1, 1),
                 padding: str = 'SAME',
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.conv_1 = ConvBNLeaky(filters=filters // 2,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding=padding)
        self.conv_2 = ConvBNLeaky(filters=filters // 4,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)
        self.conv_3 = ConvBNLeaky(filters=filters//2,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding=padding)
    def call(self, inputs, *args, **kwargs):
        inputs = self.conv_1(inputs)
        x = self.conv_2(inputs)
        x = self.conv_3(x)
        outputs = tf.concat([inputs,x],axis=-1)
        return outputs

if __name__ == '__main__':
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    out = tf.concat([t1,t2],axis=-1)
    print(out)