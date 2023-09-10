import tensorflow.keras as keras

# 用在customlayers.py中，继承layers.Conv2D,Layer等类
layers = keras.layers
activations = keras.activations
# 用在network文件的tiny_resnet.py中，继承models.Model类
models = keras.models
# 用在定义tiny_resnet_model.py中，optimizer.Adam()定义优化器、losses.SparseCategoricalCrossentropy()损失函数，
# metrics.Mean()评估损失 metrics.SparseCategoricalAccuracy()
optimizers = keras.optimizers
losses = keras.losses
metrics = keras.metrics
# 可用于卷积核的初始化
initializers = keras.initializers
