import glob
import random

import numpy as np
from PIL import Image
import cv2


class Generator:
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 n_way: int,
                 k_shot: int,
                 q_query: int,
                 image_size: tuple,
                 num_train_class:int,
                 **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.image_size = image_size
        self.num_train_class = num_train_class

        self.get_file_list()

    def get_file_list(self):
        total_file_list = np.array(glob.glob(self.data_path + '/**/character*', recursive=True))  # random没有局部随机种子，转为np
        # 使用np.random.RandomState()进行局部随机，局部随机种子也是仅一次有效。
        # 必须创建一次随机数生成器，再通过生成器调用函数，只有这样才能保证生成的随机数相同。
        rd1 = np.random.RandomState(1)
        rd1.shuffle(total_file_list)
        total_file_list = total_file_list.tolist()
        assert self.n_way < total_file_list.__len__(), 'n_way必须小于数据总类别数'
        self.train_labels = [i for i in range(self.num_train_class)]
        self.train_file_list = total_file_list[:self.num_train_class]
        self.test_labels= rd1.choice(range(len(self.train_file_list)),size=self.n_way)
        self.test_file_list = np.array(self.train_file_list)[self.test_labels].tolist()


    def get_train_len(self):
        return self.train_file_list.__len__() // self.batch_size

    def get_test_len(self):
        return (self.test_file_list.__len__() // self.batch_size)*10

    def get_one_task_data(self, file_list:list,labels:list,traing:bool):

        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []
        if traing:
            index = random.sample(labels,k=self.n_way)
            img_dirs = np.array(file_list)[index].tolist()
        else:
            img_dirs = file_list
            index = labels
        for label, img_dir in zip(index,img_dirs):
            img_list = glob.glob(pathname=img_dir + '/*.png', recursive=True)
            if traing:
                img_paths = img_list[:self.k_shot + self.q_query]
            else:
                img_paths = img_list[self.k_shot+self.q_query:]
            for img_path in random.sample(img_paths,k=self.n_way):
                image = cv2.imread(img_path)
                image = cv2.resize(image,dsize=(self.image_size),interpolation=cv2.INTER_CUBIC)
                image = image / 127.5-1
                image = np.clip(image,-1.,1.)
                support_data.append((image, label))
            for img_path in random.sample(img_paths,k=self.q_query):
                image = cv2.imread(img_path)
                image = cv2.resize(image, dsize=(self.image_size), interpolation=cv2.INTER_CUBIC)
                image = image / 127.5 - 1
                image = np.clip(image, -1., 1.)
                query_data.append((image, label))
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])
        return support_image, support_label, query_image, query_label

    def genrate(self, training: bool = True):
        support_sources, support_targets = [], []
        query_sources, query_targets = [], []
        while True:
            if training:
                file_list = self.train_file_list
                labels = self.train_labels
            else:
                file_list = self.test_file_list
                labels = self.test_labels

            support_image, support_label, query_image, query_label = self.get_one_task_data(file_list,labels,training)

            support_sources.append(support_image)
            support_targets.append(support_label)
            query_sources.append(query_image)
            query_targets.append(query_label)
            if support_sources.__len__() == self.batch_size:
                anno_support_sources = np.array(support_sources.copy())
                anno_support_targets = np.array(support_targets.copy())
                anno_query_sources = np.array(query_sources.copy())
                anno_query_targets = np.array(query_targets.copy())

                support_sources.clear()
                support_targets.clear()
                query_sources.clear()
                query_targets.clear()
                yield anno_support_sources, anno_support_targets, anno_query_sources, anno_query_targets


if __name__ == '__main__':
    from configure import config as cfg
    file_list = ['1','2',"3",'4','5','6']
    index = random.sample(range(len(file_list)), k=5)
    img_dirs = np.array(file_list)[index].tolist()
    a = random.sample([1,2,3,4,5],5)
    gen = Generator(data_path=cfg.data_path,
                    batch_size=cfg.batch_size,
                    n_way=cfg.n_way,
                    k_shot=cfg.k_shot,
                    q_query=cfg.q_query,
                    image_size=cfg.image_szie,
                    num_train_class=cfg.num_train_classes)
    train_gen = gen.genrate(training=True)
    print(1)
    a, b, c, d = next(train_gen)
    1
