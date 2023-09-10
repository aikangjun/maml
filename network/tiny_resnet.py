from custom import *
from custom.customlayers import ConvBNLeaky,ResBlock
class Tiny_Resnet(models.Model):
    def __init__(self,
                 num_classes:int,
                 **kwargs):
        super(Tiny_Resnet, self).__init__(**kwargs)
        self.conv_1 = ConvBNLeaky(filters=32,
                                  kernel_size=(3,3),
                                  strides=(1,1),
                                  padding='SAME',
                                  groups=1)
        self.resblk_2 = ResBlock(filters=64)
        self.resblk_3 = ResBlock(filters=128)
        self.globalpooling_4 = layers.GlobalAveragePooling2D()
        self.dense_5 = layers.Dense(num_classes)
        self.softmax_6 = layers.Softmax(axis=-1)
    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.resblk_2(x)
        x = self.resblk_3(x)
        x = self.globalpooling_4(x)
        x = self.dense_5(x)
        x = self.softmax_6(x)
        return x
if __name__ == '__main__':
    import tensorflow as tf
    a = tf.random.normal(shape=(4,12,12,3))
    network = Tiny_Resnet(num_classes=5)
    out = network(a)