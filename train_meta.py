import tensorflow as tf
from maml import MAML
from _uilts.generator import Generator
from configure import config as cfg
import os

if __name__ == '__main__':
    gen = Generator(data_path=cfg.data_path,
                    batch_size=cfg.batch_size,
                    n_way=cfg.n_way,
                    k_shot=cfg.k_shot,
                    q_query=cfg.q_query,
                    image_size=cfg.image_szie,
                    num_train_class=cfg.num_train_classes)
    maml = MAML(meta_train_lr=cfg.meta_train_lr,
                meta_test_lr=cfg.meta_test_lr,
                num_classes=cfg.num_classes)


    if not os.path.exists(cfg.ckpt_meta_path):
        os.makedirs(cfg.ckpt_meta_path)
    ckpt = tf.train.Checkpoint(network=maml.network,
                               optimizer=maml.optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=cfg.ckpt_meta_path,
                                              max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('最近检测点已加载！')

    train_gen = gen.genrate(training=True)

    for epoch in range(cfg.train_epochs):
        print('start training...')
        for i in range(gen.get_train_len()):
            support_scources,support_targets,query_scources,query_targets = next(train_gen)
            maml.train(support_scources,support_targets,query_scources,query_targets)
            print(f'support_loss:{maml.support_loss.result().numpy()}\t',
                  f'query_loss:{maml.query_loss.result().numpy()}\t',
                  f'support_acc:{maml.support_acc.result().numpy()*100}\t',
                  f'query_acc:{maml.query_acc.result().numpy()*100}\n')
        print(f'Epoch {epoch+1}:\n',
              f'support_loss:{maml.support_loss.result().numpy()}\t',
              f'query_loss:{maml.query_loss.result().numpy()}\t',
              f'support_acc:{maml.support_acc.result().numpy() * 100}\t',
              f'query_acc:{maml.query_acc.result().numpy() * 100}\n')
        new_ckpt_path = ckpt_manager.save()

        maml.support_loss.reset_states()
        maml.support_acc.reset_states()
        maml.query_loss.reset_states()
        maml.query_acc.reset_states()


