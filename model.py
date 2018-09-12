#coding=utf-8
#tensorflow version by xyf

import tensorflow as tf
import numpy as np
tf.app.flags.DEFINE_integer('text_scale', 512, '')
from tensorflow.contrib import slim
from nets import resnet_v1
FLAGS = tf.app.flags.FLAGS
import tensorflow.contrib.layers as layers
from lib.deform_conv_layer.deform_conv_op import deform_conv_op
from lib.deform_psroi_pooling_layer import deform_psroi_pooling_op
from lib.cnn_tools.tools import *
from lib.rpn_tools.my_anchor_target_layer_modified import AnchorTargetLayer
from lib.rpn_tools.proposal_layer_modified import ProposalLayer_Chunk
from lib.rpn_tools.proposal_target_layer_modified import ProposalTargetLayer

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def deform_conv_2d(img, num_outputs, kernel_size=3, stride=2,
                   normalizer_fn=layers.batch_norm, activation_fn=lrelu, name=''):
    img = tf.transpose(img, [0, 3, 1, 2])
    img_shape = img.shape.as_list()
    print("img_shape:"+str(img_shape))
    assert(len(img_shape) == 4)
    N, C, H, W = img_shape
    with tf.variable_scope('deform_conv' + '_' + name,reuse=tf.AUTO_REUSE):
        img = tf.transpose(img, [0, 2, 3, 1])
        offset = layers.conv2d(img, num_outputs=2 * kernel_size**2, kernel_size=3,stride=2, activation_fn=None, data_format='NHWC')
        print("offset:"+str(offset.shape))
        offset = tf.transpose(offset, [0, 3, 1, 2])
        kernel = tf.get_variable(name='',shape=(num_outputs, C, kernel_size, kernel_size),initializer=tf.random_normal_initializer(0, 0.02))
        img = tf.transpose(img, [0, 3, 1, 2])
        res = deform_conv_op(img, filter=kernel, offset=offset, rates=[1, 1, 1, 1], padding='SAME',
                             strides=[1, 1, stride, stride], num_groups=1, deformable_group=1)
        #if normalizer_fn is not None:
        #    res = normalizer_fn(res)
        #if activation_fn is not None:
        #    res = activation_fn(res)

    return res

def get_inception_layer(inputs, conv11_size, conv33_11_size, conv33_size,
                        conv55_11_size, conv55_size, pool11_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d(inputs, conv11_size, [1, 1],padding="same")
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d(inputs, conv33_11_size, [1, 1],padding="same")
        conv33 = layers.conv2d(conv33_11, conv33_size, [3, 3],padding="same")
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d(inputs, conv55_11_size, [1, 1],padding="same")
        conv55 = layers.conv2d(conv55_11, conv55_size, [5, 5],padding="same")
    with tf.variable_scope("pool_proj"):
        pool_proj = layers.max_pool2d(inputs, [3, 3], stride=1,padding="same")
        pool11 = layers.conv2d(pool_proj, pool11_size, [1, 1],padding="same")
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool11])
    print("sss:"+str(tf.concat([conv11,conv33,conv55,pool11],3).shape))
    return tf.concat([conv11, conv33, conv55, pool11], 3)

def get_inception_text_layer(inputs, conv11_size, conv33_11_size, conv33_size,
                        conv55_11_size, conv55_size, conv_shortcut_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d(inputs, conv11_size, [1, 1])#(batch,32,32,256)
        conv11_s = deform_conv_2d(conv11, conv11_size,kernel_size=3,stride=1,normalizer_fn=layers.batch_norm, activation_fn=lrelu)#(batch,256,31,31)
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d(inputs, conv33_11_size, [1, 1])#(batch,32,32,256)
        conv33 = layers.conv2d(conv33_11, conv33_size, [3, 3],padding="same") #(batch,32,32,256)
        conv33_s = deform_conv_2d(conv33, conv33_size, kernel_size=3, stride=1,normalizer_fn=layers.batch_norm, activation_fn=lrelu)#(batch,256,31,31)
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d(inputs, conv55_11_size, [1, 1])#(batch,32,32,256)
        conv55 = layers.conv2d(conv55_11, conv55_size, [5, 5],padding="same")#(batch,32,32,256)
        conv55_s = deform_conv_2d(conv55, conv55_size, kernel_size=3, stride=1, normalizer_fn=layers.batch_norm,activation_fn=lrelu)#(batch,256,31,31)
    conv_shortcut = layers.conv2d(inputs, conv_shortcut_size,[1, 1])#(batch,32,32,256)
    #short_cut_result = layers.conv2d(conv_shortcut,conv_shortcut_size,[1, 1])#(batch,32,32,256)
    print("conv11_s.shape:"+str(conv11_s.shape))
    print("conv33_s.shape:"+str(conv11_s.shape))
    print("conv55_s.shape:"+str(conv11_s.shape))
        
    conv_left = tf.concat([tf.transpose(conv11_s,[0,2,3,1]), tf.transpose(conv33_s,[0,2,3,1]), tf.transpose(conv55_s,[0,2,3,1])],3)#(batch,768,16,16)
    print("conv_left:"+str(conv_left.shape))
    #conv_left = tf.transpose(conv_left, [0, 2, 3, 1])#(batch,16,16,768)
    #print("conv_left_new:"+str(conv_left.shape))
    conv_left_next = layers.conv2d(conv_left,32,[1,1])#(batch,16,16,256)
    last_result = tf.nn.relu(tf.add(conv_left_next,conv_shortcut))
    print("last_result:"+str(last_result.shape))
    return last_result

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            #resnet50
            #f = [end_points['pool3'],end_points['pool3'], end_points['pool3']]
            f =[end_points["resnet_v1_50/block2"],end_points["resnet_v1_50/block3"],end_points["resnet_v1_50/block4"]]
            for i in range(3):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            #num_outputs = [1024]
	    num_outputs = [1024]
            
            c_stage3 = slim.conv2d(f[0], num_outputs[0], [1,1]) #(batch,32,32,1024)
            print("c_stage3.shape:"+str(c_stage3.shape))
            unsample_stage4 = unpool(f[1]) #(batch,32,32,1024)
            print("unsample_stage4.shape:"+str(unsample_stage4.shape))
            h1 = tf.add(c_stage3, unsample_stage4) #(batch,32,32,1024)
            print("h1.shape:"+str(h1.shape))

            c_stage5 = slim.conv2d(f[2],num_outputs[0],[1,1])#(batch,32,32,1024)
            unsample_stage5 = unpool(c_stage5)#(batch,32,32,1024)
            h2 = tf.add(c_stage3, unsample_stage5)#(batch,32,32,1024)
            h12 = tf.concat([h1,h2],3)
            #Inception-Text
            I_all = get_inception_text_layer(h12,64,96,128,16,32,32)
            #I_12 = tf.nn.relu(tf.add(I1,I2))
            for i in range(2):
		I_all=unpool(I_all)
            #print("I_last.shape:"+str(I_last.shape))
            F_score = slim.conv2d(I_all, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(I_all, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(I_all, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)
            print("F_score.shape:"+str(F_score.shape))
	    print("F_geometry.shape:"+str(F_geometry.shape))
    return F_score, F_geometry



def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss



def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction of text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss

if __name__ == '__main__':
    pass
