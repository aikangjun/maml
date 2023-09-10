# generate
data_path = r'D:\dataset\image\omniglot-master\python\images_background'
batch_size = 4
n_way = 5  # 5-way
k_shot = 5  # 5-shot
q_query = 5
image_szie = (28,28)
num_train_classes = 200   #
# network
num_classes = num_train_classes
# model
meta_train_lr = 1e-4
meta_test_lr = 1e-4

# train_meta
train_epochs = 1
ckpt_meta_path = '.\\ckpt_meta' # 用于保存在训练meta模型时的ckpt
# finetune
finetune_epochs = 5
ckpt_finetune_path = '.\\ckpt_finetune'