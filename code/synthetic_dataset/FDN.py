#!/usr/bin/env python
# coding=utf-8

import glob
import logging
import math
import random
import shutil
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.estimator.canned import metric_keys
import os as os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", 'fdn_train', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("num_threads", 64, "线程数，Number of threads")
tf.app.flags.DEFINE_integer("embedding_size", 32, "embedding大小，Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "训练轮次，Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1024, "训练批次大小，Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 100, "保存频率，save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0003, "学习率，正数，learning rate")
tf.app.flags.DEFINE_float("l1_reg", 0.01, "L1正则系数，L1 regularization")
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2正则系数，L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "损失评估类型，loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "训练优化器，optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "深度层设置，deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5', "dropout设置，与deep_layers设置对应，dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "是否进行批归一化，perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '../../data/synthetic_dataset/', "数据主目录，data dir")

tf.app.flags.DEFINE_string("model_dir", './model/fdn', "cp存放，code check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", './model/fdn',
                           "模型存放，export servable code for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "任务类型，task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "是否清除已存在的模型，clear existing code or not")
tf.app.flags.DEFINE_float("pos_weight", 1, "正样本权重")
tf.app.flags.DEFINE_string("pos_weights", "1,1", "正样本权重")
tf.app.flags.DEFINE_integer("experts_num", 8, "打分专家个数")
tf.app.flags.DEFINE_integer("task_num", 2, "任务数")
tf.app.flags.DEFINE_integer("units", 32, "打分专家神经元个数")
# 自编码器个数设置
tf.app.flags.DEFINE_string("audoencoder_nums", '2,2', "解码器深度层神经元设置")
# 共享编码器压缩个数设置
tf.app.flags.DEFINE_integer("shared_encoder_expert_num", 4, "解码器深度层神经元设置")
# 编码器设置
tf.app.flags.DEFINE_string("private_encoder_type", 'sparse', "任务私有编码器类型，默认是sparse，{sparse，denoise，both}")
tf.app.flags.DEFINE_float("denoise_encoder_scale", 0.8, "降噪编码器中噪声权重")
tf.app.flags.DEFINE_string("private_encoder_units", '128,64', "任务私有编码器深度层神经元设置")
tf.app.flags.DEFINE_string("shared_encoder_units", '128,64', "共享编码器深度层神经元设置")
# 解码器设置
tf.app.flags.DEFINE_string("decoder_units", '128,256', "解码器深度层神经元设置")
# 损失函数权重设置
tf.app.flags.DEFINE_string("loss_weights", '1.0,1.0', "loss权重")
# Tower层设置
tf.app.flags.DEFINE_string("tower_units", '128', "tower神经元设置")

# 日志打印设置
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
ch = logging.StreamHandler()  # 标准输出流
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def decode_line(line):
    """
    解码数据
    :param line:
    :return:
    """
    columns = tf.string_split([line], sep=',', skip_empty=False)
    # label解析
    labels = tf.string_to_number(columns.values[0 : 2], out_type=tf.float32)

    # 特征解析
    features = tf.string_to_number(columns.values[1026: ], out_type=tf.float32)

    return {
               'features': features
           }, labels


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    files = tf.data.Dataset.list_files(filenames)

    # 多线程解析libsvm数据
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TextLineDataset(filename, buffer_size=batch_size * 32, #compression_type='GZIP',
                                                     num_parallel_reads=10),
            cycle_length=len(filenames),
            buffer_output_elements=batch_size,
            prefetch_input_elements=batch_size,
            sloppy=True))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda line:
                                                               decode_line(line),
                                                               batch_size=batch_size,
                                                               num_parallel_batches=10))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 32)
    dataset = dataset.repeat(num_epochs)
    dataset.prefetch(400000)
    dataset.cache()

    return dataset


def valid_input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    files = tf.data.Dataset.list_files(filenames)

    # 多线程解析libsvm数据
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TextLineDataset(filename, buffer_size=batch_size * 32,
                                                     num_parallel_reads=10),
            cycle_length=len(filenames),
            buffer_output_elements=batch_size,
            prefetch_input_elements=batch_size,
            sloppy=True))
    if perform_shuffle:
        num_parallel_batches = 10
        buffer_size = 200000
    else:
        num_parallel_batches = 50
        buffer_size = 1000000
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda line:
                                                               decode_line(line),
                                                               batch_size=batch_size,
                                                               num_parallel_batches=num_parallel_batches))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 4)
    dataset = dataset.repeat(num_epochs)
    dataset.prefetch(500000)
    dataset.cache()
    return dataset


def model_fn(features, labels, mode, params):
    print("开启新一轮训练~")
    """构建Estimator模型"""
    # ------hyperparameters----
    dropout_ = FLAGS.dropout
    l2_rate = params["l2_reg"]
    l2_reg = tf.contrib.layers.l2_regularizer(l2_rate)
    learning_rate = params["learning_rate"]
    # 多目标优化参数设置
    experts_num = FLAGS.experts_num
    task_num = FLAGS.task_num
    units = FLAGS.units

    # ------获取特征输入-------
    features_ = features['features']
    feat_dim = 512
    # numerical_feat = tf.log1p(features_) # None * 499
    embedding = tf.reshape(features_, shape = [-1, feat_dim])

    # numerical_feat = tf.reshape(numerical_feat, shape=[-1, 499, 1])
    # numerical_w = tf.get_variable(name='numerical_w',
    #                               shape=[499, FLAGS.embedding_size], dtype=tf.float32,
    #                               initializer=tf.glorot_normal_initializer())
    # numerical_feat = tf.multiply(numerical_feat, numerical_w)  # None * 499 * E
    #
    # embedding = tf.concat([numerical_feat], axis=1)  # None * 499 * E
    # embedding = tf.reshape(embedding, shape = [-1, 499 * FLAGS.embedding_size])

    # 自动编码器架构：
    #   设计了一款新的自动编码器架构，用于服务多任务学习
    #   motivation: 利用自动编码器的性质，分解出原始特征中的任务私有特征和共享特征。
    #   way: 新的自编码器由两个编码器和一个解码器组成，其中两个编码器分别是正则编码器和普通编码器，
    #        正则编码器关注原始特征中对当前私有任务表征突出的特征（私有特征），另外一个普通解码器
    #        用于解析对当前任务表征不太突出的特征（可作共享特征处理）。解码器是普通解码器。

    # TSNE 收集器
    tsne_dict = {}
    # 各任务私有编码输出，表征当前任务，用来约束任务私有编码器
    private_encoder_outputs = []
    # 共享编码输入，表征共享特征
    shared_encoder_outputs = []
    # 自编码器
    with tf.variable_scope("autoencoder-part"):
        audoencoder_nums = list(map(int, FLAGS.audoencoder_nums.strip().split(',')))
        for i in range(FLAGS.task_num):
            private_encoder_output = []
            for j in range(audoencoder_nums[i]):
                # 组装自编码器
                # 任务私有编码器(正则编码器，用于提取显著性特征服务于私有任务的表达)
                private_encoder_units = list(map(int, FLAGS.private_encoder_units.strip().split(',')))
                private_encoder = embedding
                l1_reg = tf.contrib.layers.l1_regularizer(FLAGS.l1_reg)
                for k in range(len(private_encoder_units)):
                    private_encoder = tf.contrib.layers.fully_connected(inputs=private_encoder,
                                                                        num_outputs=private_encoder_units[k],
                                                                        activation_fn=tf.nn.relu,
                                                                        weights_regularizer=l1_reg,
                                                                        scope='private_encoder_%d_%d_%d' % (
                                                                            i, j, k))  # None * 64
                private_encoder_shape = private_encoder.get_shape().as_list()
                private_encoder_output.append(
                    tf.reshape(private_encoder, shape=[-1, 1, private_encoder_shape[-1]]))  # None * 1 * 64
                # 共享编码器
                shared_encoder_units = list(map(int, FLAGS.shared_encoder_units.strip().split(',')))
                shared_encoder = embedding
                for k in range(len(shared_encoder_units)):
                    shared_encoder = tf.contrib.layers.fully_connected(inputs=shared_encoder,
                                                                       num_outputs=shared_encoder_units[k],
                                                                       activation_fn=tf.nn.relu,
                                                                       scope='shared_encoder_%d_%d_%d' % (
                                                                           i, j, k))  # None * 64
                shared_encoder_shape = private_encoder.get_shape().as_list()
                shared_encoder_outputs.append(tf.reshape(shared_encoder, shape=[-1, 1, shared_encoder_shape[-1]]))
            private_encoder_outputs.append(private_encoder_output)

    # 共享编码器合并，并将其降维至与私有编码器结果输出的形状
    # 利用gate network 去调节各个shared-encoder的权重
    for i, encoder in enumerate(shared_encoder_outputs):
        tsne_dict['shared_encoder_%d' % (i)] = tf.squeeze(encoder)
    for i in range(len(private_encoder_outputs)):
        for j in range(len(private_encoder_outputs[i])):
            tsne_dict['private_encoder_%d_%d' % (i, j)] = tf.squeeze(private_encoder_outputs[i][j])
    shared_feat = tf.concat(shared_encoder_outputs, axis=1)  # None * N * F

    task_gate_outputs = []
    # 对每个任务的main_output进行建模
    for i in range(FLAGS.task_num):
        # gate wgts
        task_out = tf.concat(private_encoder_outputs[i] + [shared_feat], axis=1)
        task_out_shape = task_out.get_shape().as_list()
        task_gate_outputs.append(tf.reshape(task_out, shape=[-1, task_out_shape[1] * task_out_shape[2]]))

    # task1
    y_1 = tf.concat(task_gate_outputs[0], axis=-1)
    y_1 = tf.contrib.layers.fully_connected(inputs=y_1, num_outputs=1, activation_fn=None, \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_rate),
                                                scope='task1_out')
    y_1 = tf.reshape(y_1, [-1, ])

    # task2
    y_2 = tf.concat(task_gate_outputs[1], axis=-1)
    y_2 = tf.contrib.layers.fully_connected(inputs=y_2, num_outputs=1, activation_fn=None, \
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(l2_rate),
                                               scope='task2_out')
    y_2 = tf.reshape(y_2, [-1, ])


    # 预测结果导出格式设置
    predictions = {
        "y_1": y_1,
        "y_2": y_2
    }
    predictions = {**predictions, **tsne_dict}
    for i, (private_encoder, shared_encoder) in enumerate(zip([private_encoder for private_enncoders in private_encoder_outputs
                                                for private_encoder in private_enncoders], shared_encoder_outputs)):
        predictions['pe_%d' % i] = private_encoder
        predictions['se_%d' % i] = shared_encoder
    
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Estimator预测模式
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------拆分标签，构建损失------
    labels = tf.split(labels, num_or_size_splits=2, axis=-1)
    label_1 = tf.reshape(labels[0], shape=[-1, ])
    label_2 = tf.reshape(labels[1], shape=[-1, ])
    targets=[]
    targets.append(label_1)
    targets.append(label_2)
    loss = tf.reduce_mean(
        tf.losses.mean_squared_error(predictions=y_1, labels=label_1)) + \
           tf.reduce_mean(
               tf.losses.mean_squared_error(predictions=y_2, labels=label_2)) #+ \
           # l2_rate * tf.nn.l2_loss(numerical_w)
    # for wgt in gate_kernels:
    #     loss += l2_reg * tf.nn.l2_loss(wgt)
    # loss += l2_reg * tf.nn.l2_loss(expert_kernels)
    # 对任务私有编码器进行约束
    loss_weights = list(map(float, FLAGS.loss_weights.strip().split(',')))
    y_sigmoids = []
    for i in range(FLAGS.task_num):
        y_sigmoid = []
        for out in private_encoder_outputs[i]:
            y_ = tf.contrib.layers.fully_connected(inputs=tf.reshape(out, shape=[-1, private_encoder_shape[-1]]),
                                                   num_outputs=1,
                                                   activation_fn=None,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                       FLAGS.l2_reg)
                                                   )
            y_ = tf.reshape(y_, [-1, ])
            y_sigmoid.append(tf.sigmoid(y_))
            loss += loss_weights[i] * tf.reduce_mean(
                tf.losses.mean_squared_error(predictions=y_, labels=targets[i]))
            # loss += tf.reduce_mean(tf.losses.mean_squared_error(logits=y_, labels=targets[i]))
        y_sigmoids.append(y_sigmoid)

    # 对specific encoder和generic encoder进行正交约束处理
    def difference_loss(private_encoder, shared_encoder):
        """
        Frobenius norm
        :param private_encoder:
        :param shared_encoder:
        :return:
        """
        shared_encoder = tf.transpose(shared_encoder, perm=[0, 2, 1])  # None * 64 * 1
        matmul = tf.matmul(shared_encoder, private_encoder) + 1e-6  # None * 64 * 64
        cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(matmul), axis=-1), axis=-1)))
        return cost

    for private_encoder, shared_encoder in zip([private_encoder for private_enncoders in private_encoder_outputs
                                                for private_encoder in private_enncoders], shared_encoder_outputs):
        loss += difference_loss(private_encoder, shared_encoder)


    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "mse": tf.metrics.mean_absolute_error(label_1, y_1),
        "mse_1": tf.metrics.mean_absolute_error(label_1, y_1),
        "mse_2": tf.metrics.mean_absolute_error(label_2, y_2)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-6)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def batch_norm_layer(x, train_phase, scope_bn):
    """
    批标准化
    :param x:
    :param train_phase:
    :param scope_bn:
    :return:
    """
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def main(_):
    # ------init Envs------
    print(FLAGS.data_dir)
    tr_files = glob.glob('%s/train_data/*' % FLAGS.data_dir)
    random.shuffle([tr_files])
    print("tr_files:", tr_files)
    va_files = glob.glob('%s/test_data/*' % FLAGS.data_dir)
    print("va_files:", va_files)

    # ------bulid Tasks------
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
    }

    # strategy = tf.contrib.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2","/gpu:3"])  # train_distribute=strategy, eval_distribute=strategy
    strategy = tf.distribute.MirroredStrategy()  # train_distribute=strategy, eval_distribute=strategy
    # strategy = tf.distribute.experimental.CentralStorageStrategy(
    # )  # train_distribute=strategy, eval_distribute=strategy
    config_proto = tf.ConfigProto(allow_soft_placement=True,
                                  # device_count={'GPU': 0},
                                  intra_op_parallelism_threads=0,
                                  # 线程池中线程的数量，一些独立的操作可以在这指定的数量的线程中进行并行，如果设置为0代表让系统设置合适的数值
                                  inter_op_parallelism_threads=0,
                                  # 每个进程可用的为进行阻塞操作节点准备的线程池中线程的数量，设置为0代表让系统选择合适的数值，负数表示所有的操作在调用者的线程中进行。注意：如果在创建第一个Session的适合制定了该选项，那么之后创建的所有Session都会保持一样的设置，除非use_per_session_threads为true或配置了session_inter_op_thread_pool。
                                  log_device_placement=False,
                                  # gpu_options=gpu_options
                                  )
    config = tf.estimator.RunConfig(train_distribute=strategy, eval_distribute=strategy, session_config=config_proto,
                                    log_step_count_steps=FLAGS.log_steps, save_checkpoints_steps=FLAGS.log_steps * 10,
                                    save_summary_steps=FLAGS.log_steps * 10, tf_random_seed=2021)

    Model = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    feature_spec = {
        'features': tf.placeholder(dtype=tf.float32, shape=[None, 512], name='features'),
    }

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size),
            steps=None,
            # exporters=[model_best_exporter(FLAGS.job_name, feature_spec, exports_to_keep=1,
            #                                metric_key=metric_keys.MetricKeys.AUC, big_better=False)],
            start_delay_secs=10, throttle_secs=10
        )
        if FLAGS.clear_existing_model:
            try:
                shutil.rmtree(FLAGS.model_dir)
            except Exception as e:
                print(e, "at clear_existing_model")
        else:
            print("existing code cleaned at %s" % FLAGS.model_dir)
        tf.estimator.train_and_evaluate(Model, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        Model.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
