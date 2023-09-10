from network.tiny_resnet import Tiny_Resnet
from custom import *
import tensorflow as tf


class MAML:
    '''
    MAML是模型(一种network的训练方式)
    '''

    def __init__(self,
                 meta_train_lr: float,
                 meta_test_lr: float,
                 num_classes: int,
                 **kwargs):
        super(MAML, self).__init__(**kwargs)
        self.meta_train_lr = meta_train_lr
        self.meta_test_lr = meta_test_lr
        self.num_classes = num_classes

        self.network = Tiny_Resnet(num_classes=self.num_classes)
        self.optimizer = optimizers.Adam(learning_rate=meta_train_lr)
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)

        self.total_support_loss = []

        self.support_loss = metrics.Mean()
        self.query_loss = metrics.Mean()

        self.support_acc = metrics.SparseCategoricalAccuracy()
        self.query_acc = metrics.SparseCategoricalAccuracy()

        self.test_loss = metrics.Mean()
        self.test_acc = metrics.SparseCategoricalAccuracy()
        self.total_query_loss = []

    def train(self, support_sources, support_targets, query_sources, query_targets):

        with tf.GradientTape() as query_tape:
            for support_source, support_target, query_source, query_target \
                    in zip(support_sources, support_targets, query_sources, query_targets):
                # 从bacth_size中单独拿出每一个任务
                with tf.GradientTape() as support_tape:
                    support_softmax = self.network(support_source)
                    support_loss = self.loss_fn(support_target, support_softmax)

                sub_gradients = support_tape.gradient(support_loss, self.network.trainable_variables)

                sub_network = Tiny_Resnet(num_classes=self.num_classes)
                sub_network(layers.Input(shape=(None, None, 3)))
                sub_network.set_weights(self.network.get_weights())
                # 计算该task在meta模型的梯度
                '''
                手动更新sub_network网络参数，将Variable变为带meta模型的梯度的Constant
                sub_network网络全部为Constant,将不能计算梯度，反而附着meta模型的梯度
                sub_network网络输出的loss可以梯度下降作用于meta模型
                '''
                # updated_vars = [var - self.meta_train_lr * grad for var, grad in
                #                 zip(self.network.trainable_variables, sub_gradients)]
                # sub_network.trainable_variables.assign(updated_vars)
                # 以上方法不能实现梯度手动更新
                z = 0
                for k in range(len(sub_network.layers)):
                    if sub_network.layers[k].name.split('_')[0] not in ['conv', 'res', 'dense']:
                        continue
                    else:
                        if sub_network.layers[k].name.split('_')[0] == 'conv':
                            sub_network.layers[k].conv.kernel = tf.subtract(self.network.layers[k].conv.kernel,
                                                                            tf.multiply(self.meta_train_lr,
                                                                                        sub_gradients[z]))
                            sub_network.layers[k].conv.bias = tf.subtract(self.network.layers[k].conv.bias,
                                                                          tf.multiply(self.meta_train_lr,
                                                                                      sub_gradients[z + 1]))
                            sub_network.layers[k].bn.gamma = tf.subtract(self.network.layers[k].bn.gamma,
                                                                         tf.multiply(self.meta_train_lr,
                                                                                     sub_gradients[z + 2]))
                            sub_network.layers[k].bn.beta = tf.subtract(self.network.layers[k].bn.beta,
                                                                        tf.multiply(self.meta_train_lr,
                                                                                    sub_gradients[z + 3]))
                            z += 4
                        elif sub_network.layers[k].name.split('_')[0] == 'res':
                            sub_network.layers[k].conv_1.conv.kernel = tf.subtract(
                                self.network.layers[k].conv_1.conv.kernel,
                                tf.multiply(self.meta_train_lr, sub_gradients[z]))
                            sub_network.layers[k].conv_1.conv.bias = tf.subtract(
                                self.network.layers[k].conv_1.conv.bias,
                                tf.multiply(self.meta_train_lr, sub_gradients[z + 1]))
                            sub_network.layers[k].conv_1.bn.gamma = tf.subtract(self.network.layers[k].conv_1.bn.gamma,
                                                                                tf.multiply(self.meta_train_lr,
                                                                                            sub_gradients[z + 2]))
                            sub_network.layers[k].conv_1.bn.beta = tf.subtract(self.network.layers[k].conv_1.bn.beta,
                                                                               tf.multiply(self.meta_train_lr,
                                                                                           sub_gradients[z + 3]))
                            z += 4
                            sub_network.layers[k].conv_2.conv.kernel = tf.subtract(
                                self.network.layers[k].conv_2.conv.kernel,
                                tf.multiply(self.meta_train_lr, sub_gradients[z]))
                            sub_network.layers[k].conv_2.conv.bias = tf.subtract(
                                self.network.layers[k].conv_2.conv.bias,
                                tf.multiply(self.meta_train_lr, sub_gradients[z + 1]))
                            sub_network.layers[k].conv_2.bn.gamma = tf.subtract(self.network.layers[k].conv_2.bn.gamma,
                                                                                tf.multiply(self.meta_train_lr,
                                                                                            sub_gradients[z + 2]))
                            sub_network.layers[k].conv_2.bn.beta = tf.subtract(self.network.layers[k].conv_2.bn.beta,
                                                                               tf.multiply(self.meta_train_lr,
                                                                                           sub_gradients[z + 3]))
                            z += 4
                            sub_network.layers[k].conv_3.conv.kernel = tf.subtract(
                                self.network.layers[k].conv_3.conv.kernel,
                                tf.multiply(self.meta_train_lr, sub_gradients[z]))
                            sub_network.layers[k].conv_3.conv.bias = tf.subtract(
                                self.network.layers[k].conv_3.conv.bias,
                                tf.multiply(self.meta_train_lr, sub_gradients[z + 1]))
                            sub_network.layers[k].conv_3.bn.gamma = tf.subtract(self.network.layers[k].conv_3.bn.gamma,
                                                                                tf.multiply(self.meta_train_lr,
                                                                                            sub_gradients[z + 2]))
                            sub_network.layers[k].conv_3.bn.beta = tf.subtract(self.network.layers[k].conv_3.bn.beta,
                                                                               tf.multiply(self.meta_train_lr,
                                                                                           sub_gradients[z + 3]))
                            z += 4
                        elif sub_network.layers[k].name.split('_')[0] == 'dense':
                            sub_network.layers[k].kernel = tf.subtract(self.network.layers[k].kernel,
                                                                       tf.multiply(self.meta_train_lr,
                                                                                   sub_gradients[z]))
                            sub_network.layers[k].bias = tf.subtract(self.network.layers[k].bias,
                                                                     tf.multiply(self.meta_train_lr,
                                                                                 sub_gradients[z + 1]))
                            z += 2
                query_logits = sub_network(query_source)
                query_loss = self.loss_fn(query_target, query_logits)
                self.total_query_loss.append(query_loss)
                self.support_loss(support_loss)
                self.query_loss(query_loss)
                self.support_acc(support_target, support_softmax)
                self.query_acc(query_target, query_logits)
            meta_gradients = query_tape.gradient(tf.reduce_mean(self.total_query_loss),
                                                 self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(meta_gradients, self.network.trainable_variables))

        self.total_query_loss.clear()


if __name__ == '__main__':
    sub_network = Tiny_Resnet(num_classes=3)
    inputs = tf.random.normal(shape=(4, 13, 13, 1))
    outputs = sub_network(inputs)
    print(sub_network.layers)
    print(sub_network.trainable_variables)
