#coding=utf-8
#tensorflow version by xyf

import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from nets import resnet_v1
import tensorflow.contrib.layers as layers
from lib.deform_conv_layer.deform_conv_op import deform_conv_op
from lib.deform_psroi_pooling_layer import deform_psroi_pooling_op
from lib.cnn_tools.tools import *
from lib.rpn_tools.my_anchor_target_layer_modified import AnchorTargetLayer
from lib.rpn_tools.proposal_layer_modified import ProposalLayer_Chunk
from lib.rpn_tools.proposal_target_layer_modified import ProposalTargetLayer

A = 9
height = 38
width = 63
_rpn_stat = True #train rpn layers
_add_l2 = True #include weight losses
_fc_stat = True # train "fc" -- which are hconv5 residual layers and up
gt_box = tf.placeholder(tf.int64)
gt_boxbatch = tf.reshape(tf.stack(gt_box), [-1, 5])

def proposal_target(rpn_rois, gt):
  rpn_rois = np.array(rpn_rois)
  gt = np.array(gt)
  proposer_target = ProposalTargetLayer()
  proposer_target.setup()
  rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      proposer_target.forward(rpn_rois, gt)
  return rois.astype(np.int32), labels.astype(np.int64), bbox_targets, \
      bbox_inside_weights, bbox_outside_weights

def proposal(cls, bbox):
  cls = np.array(cls)
  bbox = np.array(bbox)
  proposer = ProposalLayer_Chunk()
  proposer.setup(cls, bbox)
  blob = proposer.forward(cls, bbox)
  return blob

def anchor(x, g):
  x = np.array(x)
  g = np.array(g)
  anchor = AnchorTargetLayer()
  anchor.setup(x, g)
  labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, n = \
      anchor.forward(x, g)
  return labels.astype(np.int64), bbox_targets, bbox_inside_weights, \
      bbox_outside_weights, np.array(n).astype(np.float32)

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def deform_conv_2d(img, num_outputs, kernel_size=3, stride=2,
                   normalizer_fn=layers.batch_norm, activation_fn=lrelu, name=''):
    img_shape = img.shape.as_list()
    assert(len(img_shape) == 4)
    N, C, H, W = img_shape
    with tf.variable_scope('deform_conv' + '_' + name):
        offset = layers.conv2d(img, num_outputs=2 * kernel_size**2, kernel_size=3,
                           stride=2, activation_fn=None, data_format='NCHW')
        kernel = tf.get_variable(name='d_kernel', shape=(num_outputs, C, kernel_size, kernel_size),
                                 initializer=tf.random_normal_initializer(0, 0.02))
        res = deform_conv_op(img, filter=kernel, offset=offset, rates=[1, 1, 1, 1], padding='SAME',
                             strides=[1, 1, stride, stride], num_groups=1, deformable_group=1)
        if normalizer_fn is not None:
            res = normalizer_fn(res)
        if activation_fn is not None:
            res = activation_fn(res)

    return res

def get_inception_layer(inputs, conv11_size, conv33_11_size, conv33_size,
                        conv55_11_size, conv55_size, conv_shortcut_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d(inputs, conv11_size, [1, 1])
        #deformable conv
        conv11_s = deform_conv_2d(conv11, conv11_size,kernel_size=3,stride=2, activation_fn=lrelu)
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d(inputs, conv33_11_size, [1, 1])
        conv33 = layers.conv2d(conv33_11, conv33_size, [3, 3])
        # deformable conv
        conv33_s = deform_conv_2d(conv33, conv33_size, kernel_size=3, stride=2, activation_fn=lrelu)
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d(inputs, conv55_11_size, [1, 1])
        conv55 = layers.conv2d(conv55_11, conv55_size, [5, 5])
        # deformable conv
        conv55_s = deform_conv_2d(conv55, conv55_size, kernel_size=3, stride=2, activation_fn=lrelu)
    with tf.variable_scope("conv_shortcut"):
        conv_shortcut = layers.conv2d(inputs, conv_shortcut_size,[1, 1])
        short_cut_result = layers.conv2d(conv_shortcut,[1, 1])
    conv_left = tf.concat([conv11_s, conv33_s, conv55_s],3)
    conv_left_next = layers.conv2d(conv_left,256,[1,1])
    last_result = tf.nn.relu(tf.add(conv_left_next,conv_shortcut))

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
            f = [end_points['pool3'],end_points['pool4'], end_points['pool5']]
            for i in range(3):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            num_outputs = [1024]
            c_stage3 = slim.conv2d(f[0], num_outputs[0], [1,1])
            unsample_stage4 = unpool(f[1])
            h1 = tf.concat([c_stage3, unsample_stage4], axis=-1)
            unsample_stage5 = unpool(f[2])
            h2 = tf.concat([c_stage3, unsample_stage5], axis=-1)
            #Inception-Text
            I1 = get_inception_layer(h1,256,256,256,256,256,256)
            I2 = get_inception_layer(h2,256,256,256,256,256,256)
            #RPN part
            with tf.name_scope("rpn"):
                W_rpnbase = weight_variable([3, 3, 256, 256], "rpn", tr_stat=_rpn_stat, add_l2_stat=_add_l2)
                b_rpnbase = bias_variable([512], "rpn", tr_stat=_rpn_stat, add_l2_stat=_add_l2)
                h_rpn3 = tf.nn.relu(conv2d(I1, W_rpnbase) + b_rpnbase)

                W_cls_score = weight_variable([1, 1, 512, 18], "rpn", tr_stat=_rpn_stat, add_l2_stat=_add_l2)
                b_cls_score = bias_variable([18], "rpn", tr_stat=_rpn_stat, add_l2_stat=_add_l2)
                rpn_cls_score = (conv2d_nopad(h_rpn3, W_cls_score) + b_cls_score)

                W_bbox_pred = weight_variable_bbox([1, 1, 512, 36], "rpn", tr_stat=_rpn_stat, add_l2_stat=_add_l2)
                b_bbox_pred = bias_variable([36], "rpn", tr_stat=_rpn_stat, add_l2_stat=_add_l2)
                rpn_bbox_pred = (conv2d_nopad(h_rpn3, W_bbox_pred) + b_bbox_pred)

            # RPN loss and accuracy calculation
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [1, height, width, A * 4])
            rpn_cls_score_reshape = tf.reshape(rpn_cls_score, [-1, 2]) + 1e-20

            rpn_labels_ind, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, rpn_size = \
                tf.py_func(anchor, [rpn_cls_score, gt_boxbatch],
                           [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels_ind = tf.reshape(tf.stack(rpn_labels_ind), [-1])
            rpn_bbox_targets = tf.reshape(tf.stack(rpn_bbox_targets), [1, height, width, A * 4])
            rpn_bbox_inside_weights = tf.reshape(tf.stack(rpn_bbox_inside_weights), [1, height, width, A * 4])
            rpn_bbox_outside_weights = tf.reshape(tf.stack(rpn_bbox_outside_weights), [1, height, width, A * 4])

            rpn_cls_soft = tf.nn.softmax(rpn_cls_score_reshape)
            rpn_cls_score_x = tf.reshape(tf.gather(rpn_cls_score_reshape, tf.where(tf.not_equal(rpn_labels_ind, -1))),
                                         [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_labels_ind, tf.where(tf.not_equal(rpn_labels_ind, -1))), [-1])
            rpn_loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score_x, rpn_label))

            unique_rpn_cls, o_cls, o_cls_ind = tf.py_func(cls_unique,
                                                          [rpn_cls_soft, rpn_labels_ind],
                                                          [tf.float32, tf.float32, tf.float32])
            unique_rpn_cls = tf.stack(unique_rpn_cls)

            rpn_correct_prediction = tf.py_func(rpn_accuracy, [rpn_cls_soft, rpn_labels_ind], [tf.float32])
            rpn_correct_prediction = tf.reshape(tf.stack(rpn_correct_prediction), [-1])
            rpn_cls_accuracy = tf.reduce_mean(tf.cast(rpn_correct_prediction, tf.float32))

            sigma = 3 * 3

            smoothL1_sign = tf.cast(tf.less(tf.abs(tf.subtract(rpn_bbox_pred, rpn_bbox_targets)), 1 / sigma), tf.float32)
            rpn_loss_bbox = tf.multiply(tf.reduce_mean(tf.reduce_sum(tf.multiply(rpn_bbox_outside_weights, tf.add(
                tf.multiply(tf.multiply(tf.pow(tf.multiply(rpn_bbox_inside_weights,
                                                           tf.subtract(rpn_bbox_pred, rpn_bbox_targets)), 2), 0.5 * sigma), smoothL1_sign),
                                                           tf.multiply(tf.subtract(tf.abs(tf.subtract(rpn_bbox_pred, rpn_bbox_targets)), 0.5 / sigma),
                                                           tf.abs(smoothL1_sign - 1)))), reduction_indices=[1, 2])), 1)
            rpn_loss_bbox_label = rpn_loss_bbox
            zero_count, one_count = tf.py_func(bbox_counter, [rpn_labels_ind], [tf.float32, tf.float32])

            # ROI PROPOSAL
            rpn_cls_prob = rpn_cls_soft
            rpn_cls_prob_reshape = tf.reshape(rpn_cls_prob, [1, height, width, 18])

            rpn_rois = tf.py_func(proposal, [rpn_cls_prob_reshape, rpn_bbox_pred], [tf.float32])
            rpn_rois = tf.reshape(rpn_rois, [-1, 5])

            rcnn_rois, rcnn_labels_ind, rcnn_bbox_targets, rcnn_bbox_inside_w, rcnn_bbox_outside_w = \
                tf.py_func(proposal_target, [rpn_rois, gt_boxbatch],
                           [tf.int32, tf.int64, tf.float32, tf.float32, tf.float32])
            rcnn_rois = tf.cast(tf.reshape(tf.stack(rcnn_rois), [-1, 5]), tf.float32)
            rcnn_labels_ind = tf.reshape(tf.stack(rcnn_labels_ind), [-1])
            rcnn_bbox_targets = tf.reshape(tf.stack(rcnn_bbox_targets), [-1, 2 * 4])
            rcnn_bbox_inside_w = tf.reshape(tf.stack(rcnn_bbox_inside_w), [-1, 2 * 4])
            rcnn_bbox_outside_w = tf.reshape(tf.stack(rcnn_bbox_outside_w), [-1, 2 * 4])

            #rfcn-cls-seg and rfcn-box
            with tf.name_scope("fc"):
                W_end_base = weight_variable([1, 1, 2048, 1024], "fc", tr_stat=_fc_stat, add_l2_stat=_add_l2)
                b_end_base = bias_variable([1024], "fc", tr_stat=_fc_stat, add_l2_stat=_add_l2)
                h_end_base = tf.nn.relu(conv2d_nopad(I2, W_end_base) + b_end_base)

                W_rfcn_cls = weight_variable([1, 1, 1024, 1029], "fc", tr_stat=_fc_stat, add_l2_stat=_add_l2)
                b_rfcn_cls = bias_variable([1029], "fc", tr_stat=_fc_stat, add_l2_stat=_add_l2)
                h_rfcn_cls = tf.nn.relu(conv2d_nopad(h_end_base, W_rfcn_cls) + b_rfcn_cls)

                W_rfcn_bbox = weight_variable([1, 1, 1024, 392], "fc", tr_stat=_fc_stat, add_l2_stat=_add_l2)
                b_rfcn_bbox = bias_variable([392], "fc", tr_stat=_fc_stat, add_l2_stat=_add_l2)
                h_rfcn_bbox = tf.nn.relu(conv2d_nopad(h_end_base, W_rfcn_bbox) + b_rfcn_bbox)

                h_rfcn_cls = tf.transpose(h_rfcn_cls, [0, 3, 1, 2])
                [psroipooled_cls_rois, cls_channels] = deform_psroi_pooling_op.deform_psroi_pool(h_rfcn_cls, rcnn_rois, output_dim=21,
                                                                                   group_size=7, spatial_scale=1.0 / 16)
                psroipooled_cls_rois = tf.transpose(psroipooled_cls_rois, [0, 2, 3, 1])
                end_cls = tf.reduce_mean(psroipooled_cls_rois, [1, 2])
                end_cls = tf.reshape(end_cls, [-1, 21])

                h_rfcn_bbox = tf.transpose(h_rfcn_bbox, [0, 3, 1, 2])
                [psroipooled_loc_rois, loc_channels] = deform_psroi_pooling_op.deform_psroi_pool(h_rfcn_bbox, rcnn_rois, output_dim=8,
                                                                                   group_size=7, spatial_scale=1.0 / 16)
                psroipooled_loc_rois = tf.transpose(psroipooled_loc_rois, [0, 2, 3, 1])
                end_bbox = tf.reduce_mean(psroipooled_loc_rois, [1, 2])
                end_bbox = tf.reshape(end_bbox, [-1, 8])

    return end_cls,rcnn_labels_ind,end_bbox,rcnn_bbox_targets,rcnn_bbox_outside_w,rcnn_bbox_inside_w,rpn_loss_cls,rpn_loss_bbox

def loss(end_cls,rcnn_labels_ind,end_bbox,rcnn_bbox_targets,rcnn_bbox_outside_w,rcnn_bbox_inside_w,rpn_loss_cls,rpn_loss_bbox):
    # END_LOSS
    end_cls_soft = tf.nn.softmax(end_cls)
    loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(end_cls, rcnn_labels_ind))
    loss_cls_label = loss_cls

    pred = tf.argmax(end_cls_soft, 1)
    end_correct_prediction = tf.equal(pred, rcnn_labels_ind)
    end_cls_accuracy = tf.reduce_mean(tf.cast(end_correct_prediction, tf.float32))

    sigma2 = 1

    smoothL1_sign_bbox = tf.cast(tf.less(tf.abs(tf.subtract(end_bbox, rcnn_bbox_targets)), 1 / sigma2), tf.float32)

    loss_bbox = tf.multiply(tf.reduce_mean(tf.reduce_sum(tf.multiply(rcnn_bbox_outside_w, tf.add(tf.multiply(tf.multiply(tf.pow(tf.multiply(rcnn_bbox_inside_w, tf.subtract(end_bbox, rcnn_bbox_targets)), 2), 0.5 * sigma2),
               smoothL1_sign_bbox),tf.multiply(tf.subtract(tf.abs(tf.subtract(end_bbox, rcnn_bbox_targets)), 0.5 / sigma2), tf.abs(smoothL1_sign_bbox - 1)))),
                                                    reduction_indices=[1])), 1)
    total_loss = rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + (
                tf.add_n(tf.get_collection('weight_losses_trunk')) +
                tf.add_n(tf.get_collection('weight_losses_rpn')) + tf.add_n(
                tf.get_collection('weight_losses_rcnn')) + tf.add_n(tf.get_collection('weight_losses_fc')))

    return total_loss


if __name__ == '__main__':
    pass