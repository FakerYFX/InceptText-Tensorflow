import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np
import cv2
from math import ceil

conv_feat_shape = [38, 63, 2048]
wd = 1e-4


def conv_layer(inpt, filter_shape, stride, loc, tr_stat, bn_tr_stat, add_l2_stat, state):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape, loc, tr_stat, add_l2_stat)
    conv = tf.nn.conv2d(inpt, filter_, strides=[1, stride, stride, 1], padding="SAME")
    #mean, var = tf.nn.moments(conv, axes=[0,1,2])
    moving_mean = tf.Variable(tf.zeros([out_channels]), trainable=False, name="weights")
    moving_var = tf.Variable(tf.ones([out_channels]), trainable=False, name="weights")
    beta = tf.Variable(tf.zeros([out_channels]), trainable=bn_tr_stat, name="weights")
    gamma = tf.Variable(tf.ones([out_channels]), trainable=bn_tr_stat, name="weights")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9997)
    update_moving_variance = moving_averages.assign_moving_average(moving_var, var, 0.9997)
    tf.add_to_collection('update_ops', update_moving_mean)
    tf.add_to_collection('update_ops', update_moving_variance)
    train = tf.convert_to_tensor(True, dtype='bool')
    mean, var = control_flow_ops.cond(train, lambda: (mean, var), lambda: (moving_mean, moving_var))
    batch_norm = tf.nn.batch_normalization(conv, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-5)
    if state == "split":
        out = batch_norm
    elif state == "normal":
        out = tf.nn.relu(batch_norm)
    return out

def residual_block(inpt, output_depth, down_sample, loc, tr_stat, bn_tr_stat, add_l2_stat, branch):
    input_depth = inpt.get_shape().as_list()[3]
    if np.logical_and(input_depth != (output_depth * 4), branch != "near"):
        input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth * 4], 2, loc, tr_stat, bn_tr_stat, add_l2_stat, "split")
    elif np.logical_and(input_depth != (output_depth * 4), branch == "near"):
        input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth * 4], 1, loc, tr_stat, bn_tr_stat, add_l2_stat, "split")
    else:
      input_layer = inpt
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    conv1 = conv_layer(inpt, [1, 1, input_depth, output_depth], 1, loc, tr_stat, bn_tr_stat, add_l2_stat, "normal")
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1, loc, tr_stat, bn_tr_stat, add_l2_stat, "normal")
    conv3 = conv_layer(conv2, [1, 1, output_depth, output_depth * 4], 1, loc, tr_stat, bn_tr_stat, add_l2_stat, "split")
    res = conv3 + input_layer
    return tf.nn.relu(res)



def weight_variable(shape, loc, tr_stat, add_l2_stat):
  initial = tf.truncated_normal(shape, stddev=0.01)
  weight_decay = tf.mul(tf.nn.l2_loss(initial), wd)
  if np.logical_and(add_l2_stat == True, loc == "base"):
      tf.add_to_collection('weight_losses_base', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "trunk"):
      tf.add_to_collection('weight_losses_trunk', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rpn"):
      tf.add_to_collection('weight_losses_rpn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rcnn"):
      tf.add_to_collection('weight_losses_rcnn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "fc"):
      tf.add_to_collection('weight_losses_fc', weight_decay)
  if tr_stat == True:
      return tf.Variable(initial, name = "weights", trainable = True)
  elif tr_stat == False:
      return tf.Variable(initial, name = "weights", trainable = False)

def bias_variable(shape, loc, tr_stat, add_l2_stat):
  initial = tf.constant(0.0, shape=shape)
  weight_decay = tf.mul(tf.nn.l2_loss(initial), wd)
  if np.logical_and(add_l2_stat == True, loc == "base"):
      tf.add_to_collection('weight_losses_base', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "trunk"):
      tf.add_to_collection('weight_losses_trunk', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rpn"):
      tf.add_to_collection('weight_losses_rpn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rcnn"):
      tf.add_to_collection('weight_losses_rcnn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "fc"):
      tf.add_to_collection('weight_losses_fc', weight_decay)
  if tr_stat == True:
      return tf.Variable(initial, name = "biases", trainable = True)
  elif tr_stat == False:
      return tf.Variable(initial, name = "biases", trainable = False)

def weight_variable_bbox(shape, loc, tr_stat, add_l2_stat):
  initial = tf.truncated_normal(shape, stddev=0.001)
  weight_decay = tf.mul(tf.nn.l2_loss(initial), 0.0001)
  if np.logical_and(add_l2_stat == True, loc == "base"):
      tf.add_to_collection('weight_losses_base', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "trunk"):
      tf.add_to_collection('weight_losses_trunk', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rpn"):
      tf.add_to_collection('weight_losses_rpn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rcnn"):
      tf.add_to_collection('weight_losses_rcnn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "fc"):
      tf.add_to_collection('weight_losses_fc', weight_decay)
  if tr_stat == True:
      return tf.Variable(initial, name = "weights", trainable = True)
  elif tr_stat == False:
      return tf.Variable(initial, name = "weights", trainable = False)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_nopad(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def rpn_accuracy(x, rl):
  x = np.array(x)
  xmax = np.argmax(x, axis = 1)
  rl = np.array(rl)
  s = xmax == rl
  s = s[rl != -1]
  return s.astype(np.float32)

def bbox_counter(ind):
  zeros = (ind == 0).sum(0)
  ones = (ind == 1).sum(0)
  return zeros.astype(np.float32), ones.astype(np.float32)

def process_img(im, l):
  im = np.array(im)
  '''
  imtest = np.reshape(im, [600,1000,3]).astype(np.uint8)
  for i in range(np.array(l).shape[0]):
    cv2.rectangle(imtest, (l[i,0],l[i,1]), (l[i,2],l[i,3]), (0,255,0))
  cv2.imshow("",imtest)
  cv2.waitKey()
  '''
  imx = np.zeros(im.shape, dtype = np.uint8)
  imx[:,:,0] = im[:,:,2] - 102.9801
  imx[:,:,1] = im[:,:,1] - 115.9465
  imx[:,:,2] = im[:,:,0] - 122.7717
  return imx.astype(np.float32)


def debug_bool(x, y):
  l = y[:,0] == 1
  r = x == 1
  res = np.average(np.equal(l, r))
  return res.astype(np.float32)

def cls_unique(x, lb):
  x = np.array(x)
  lb = np.array(lb)
  u = x[:,0] < x[:,1]
  o_1 = u[np.logical_and((lb != -1),(lb != 0))]
  o_2 = np.sum(o_1)
  o_1_ind = np.asarray(np.where(np.logical_and((u == 1),(lb != -1))))
  n = (u == 1)
  s = np.sum(n)
  i = np.asarray(np.nonzero(n)).astype(np.float32)
  return s.astype(np.float32), o_2.astype(np.float32), i.astype(np.float32)

def flip(img, gt, im_info):
  r = np.random.randint(low=0,high=2)
  img = np.array(img).astype(np.uint8)
  gt = np.array(gt)
  if r == 0:
    return img, gt
  elif r == 1:
    f_img = cv2.flip(img,1)
    f_gt = np.zeros(gt.shape)
    f_gt[:,4] = gt[:,4]
    f_gt[:,0] = im_info[1] - gt[:,2]
    f_gt[:,2] = im_info[1] - gt[:,0]
    f_gt[:,1] = gt[:,1]
    f_gt[:,3] = gt[:,3]
    f_gt = f_gt
    return f_img, f_gt.astype(np.int64)

