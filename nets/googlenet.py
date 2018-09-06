import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops


def get_inception_layer(inputs, conv11_size, conv33_11_size, conv33_size,
                        conv55_11_size, conv55_size, pool11_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d(inputs, conv11_size, [1, 1])
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d(inputs, conv33_11_size, [1, 1])
        conv33 = layers.conv2d(conv33_11, conv33_size, [3, 3])
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d(inputs, conv55_11_size, [1, 1])
        conv55 = layers.conv2d(conv55_11, conv55_size, [5, 5])
    with tf.variable_scope("pool_proj"):
        pool_proj = layers.max_pool2d(inputs, [3, 3], stride=1)
        pool11 = layers.conv2d(pool_proj, pool11_size, [1, 1])
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool11])
    return tf.concat([conv11, conv33, conv55, pool11], 3)


def aux_logit_layer(inputs, num_classes, is_training):
    with tf.variable_scope("pool2d"):
        pooled = layers.avg_pool2d(inputs, [5, 5], stride=3)
    with tf.variable_scope("conv11"):
        conv11 = layers.conv2d(pooled, 128, [1, 1])
    with tf.variable_scope("flatten"):
        flat = tf.reshape(conv11, [-1, 2048])
    with tf.variable_scope("fc"):
        fc = layers.fully_connected(flat, 1024, activation_fn=None)
    with tf.variable_scope("drop"):
        drop = layers.dropout(fc, 0.3, is_training=is_training)
    with tf.variable_scope("linear"):
        linear = layers.fully_connected(drop, num_classes, activation_fn=None)
    with tf.variable_scope("soft"):
        soft = tf.nn.softmax(linear)
    return soft


def googlenet(inputs,
              dropout_keep_prob=0.4,
              num_classes=1000,
              is_training=True,
              restore_logits=None,
              scope=''):
    '''
    Implementation of https://arxiv.org/pdf/1409.4842.pdf
    '''

    end_points = {}
    with tf.name_scope(scope, "googlenet", [inputs]):
        with ops.arg_scope([layers.max_pool2d], padding='SAME'):
            end_points['conv0'] = layers.conv2d(inputs, 64, [7, 7], stride=2, scope='conv0')
            end_points['pool0'] = layers.max_pool2d(end_points['conv0'], [3, 3], scope='pool0')
            end_points['conv1_a'] = layers.conv2d(end_points['pool0'], 64, [1, 1], scope='conv1_a')
            end_points['conv1_b'] = layers.conv2d(end_points['conv1_a'], 192, [3, 3], scope='conv1_b')
            end_points['pool1'] = layers.max_pool2d(end_points['conv1_b'], [3, 3], scope='pool1')

            with tf.variable_scope("inception_3a"):
                end_points['inception_3a'] = get_inception_layer(end_points['pool1'], 64, 96, 128, 16, 32, 32)

            with tf.variable_scope("inception_3b"):
                end_points['inception_3b'] = get_inception_layer(end_points['inception_3a'], 128, 128, 192, 32, 96, 64)

            end_points['pool2'] = layers.max_pool2d(end_points['inception_3b'], [3, 3], scope='pool2')

            with tf.variable_scope("inception_4a"):
                end_points['inception_4a'] = get_inception_layer(end_points['pool2'], 192, 96, 208, 16, 48, 64)

            with tf.variable_scope("aux_logits_1"):
                end_points['aux_logits_1'] = aux_logit_layer(end_points['inception_4a'], num_classes, is_training)

            with tf.variable_scope("inception_4b"):
                end_points['inception_4b'] = get_inception_layer(end_points['inception_4a'], 160, 112, 224, 24, 64, 64)

            with tf.variable_scope("inception_4c"):
                end_points['inception_4c'] = get_inception_layer(end_points['inception_4b'], 128, 128, 256, 24, 64, 64)

            with tf.variable_scope("inception_4d"):
                end_points['inception_4d'] = get_inception_layer(end_points['inception_4c'], 112, 144, 288, 32, 64, 64)

            with tf.variable_scope("aux_logits_2"):
                end_points['aux_logits_2'] = aux_logit_layer(end_points['inception_4d'], num_classes, is_training)

            with tf.variable_scope("inception_4e"):
                end_points['inception_4e'] = get_inception_layer(end_points['inception_4d'], 256, 160, 320, 32, 128,
                                                                 128)

            end_points['pool3'] = layers.max_pool2d(end_points['inception_4e'], [3, 3], scope='pool3')

            with tf.variable_scope("inception_5a"):
                end_points['inception_5a'] = get_inception_layer(end_points['pool3'], 256, 160, 320, 32, 128, 128)

            with tf.variable_scope("inception_5b"):
                end_points['inception_5b'] = get_inception_layer(end_points['inception_5a'], 384, 192, 384, 48, 128,
                                                                 128)

            end_points['pool4'] = layers.avg_pool2d(end_points['inception_5b'], [7, 7], stride=1, scope='pool4')

            end_points['reshape'] = tf.reshape(end_points['pool4'], [-1, 1024])

            end_points['dropout'] = layers.dropout(end_points['reshape'], dropout_keep_prob, is_training=is_training)

            end_points['logits'] = layers.fully_connected(end_points['dropout'], num_classes, activation_fn=None,
                                                          scope='logits')

            end_points['predictions'] = tf.nn.softmax(end_points['logits'], name='predictions')

    return end_points['logits'], end_points