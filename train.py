import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 5  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 208  # 重新定义图片的大小，图片如果过大则训练比较慢
IMG_H = 208
BATCH_SIZE = 32  # 每批数据的大小
CAPACITY = 256
MAX_STEP = 15000  # 训练的步数，应当 >= 10000
learning_rate = 0.00001  # 学习率，建议刚开始的 learning_rate <= 0.0001


def run_training():
    # 数据集
    train_dir = '/Users/Nowidh/PycharmProjects/ccc/train/'  # My dir--20170727-csq
    # logs_train_dir 存放训练模型的过程的数据，在tensorboard 中查看
    logs_train_dir = '/Users/Nowidh/PycharmProjects/ccc/saveNet/'

    # 获取图片和标签集
    train, train_label = input_data.get_files(train_dir)
    # 生成批次
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    print('开始训练！')

    # 进入模型
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    # 获取 loss
    train_loss = model.losses(train_logits, train_label_batch)
    # 训练
    train_op = model.trainning(train_loss, learning_rate)
    # 获取准确率
    train__acc = model.evaluation(train_logits, train_label_batch)
    # 合并 summary
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    # 保存summary
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or step == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def eval():
    test_dir = '/Users/Nowidh/PycharmProjects/ccc/test1/'
    logs_dir = '/Users/Nowidh/PycharmProjects/ccc/saveNet/'

    sess = tf.Session()

    test_list, test_label = input_data.get_test(test_dir)
    image_test_bench, label_train_batch = input_data.get_batch( test_list,
                                                                test_label,
                                                                IMG_W,
                                                                IMG_H,
                                                                BATCH_SIZE,
                                                                CAPACITY)

    test_logits = model.inference(image_test_bench, BATCH_SIZE, N_CLASSES)
    test_logits = tf.nn.softmax(test_logits)

    saver = tf.train.Saver()
    print('载入！！！')

    saver.restore(sess, 'saveNet/model.ckpt-8000')

    print('clear！')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_count = 0

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image, prediction = sess.run([image_test_bench, test_logits])
            aa = 0
            for now_image in prediction:
                max_index = np.argmax(now_image)

                if max_index == 0:
                    label = 'this is a daisy.'
                elif max_index == 1:
                    label = 'this is a dandelion.'
                elif max_index == 2:
                    label = 'this is a rose.'
                elif max_index == 3:
                    label = 'this is a sunflower.'
                else:
                    label = 'this is a tulips.'
                plt.imshow(image[aa])
                aa = aa + 1
                plt.title(label)
                plt.savefig(r'C:\Users\Nowidh\PycharmProjects\ccc\pre\\' + str(image_count))
                image_count = image_count + 1
                plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

# train
#run_training()
eval()
