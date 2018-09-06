# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os

import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False
my_RPN_BATCHSIZE = 256
my_fraction = 0.5
height = 38
width = 63
im_info = [600, 1000]

class AnchorTargetLayer():
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    
    def setup(self, bottom, gt):
        #layer_params = yaml.load(self.param_str_)
        #anchor_scales = ((0.5, 1, 2), (8, 16, 32))
        self._anchors = generate_anchors()
        #print self._anchors.shape
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 16

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0
        #height, width = 34, 62
        #height, width = bottom.shape[-2:]
        
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        #top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        #top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        #top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        #top[3].reshape(1, A * 4, height, width)
    
    
    def forward(self, bottom, gt):
       
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap34, 62

        assert bottom.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        #height, width = bottom[0].shape[-2:]
        #height, width = 34, 62
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = gt
        # im_info height / width
        
        

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        #print self._anchors
        all_anchors = (self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1) # used to be -1

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]


# CHANGE HERE !
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
	
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        num_fg = int(my_fraction * my_RPN_BATCHSIZE)
        
        fg_inds = np.where(labels == 1)[0]
        bg_inds = np.where(labels == 0)[0]
        
        '''
        if len(bg_inds) >= len(fg_inds) * 20:
	  enabled_bg_inds = npr.choice(
                bg_inds, size=(len(fg_inds) * 20), replace=False)
	  labels[bg_inds] = -1
	  labels[enabled_bg_inds] = 0
	  bg_inds_no = len(enabled_bg_inds)
	  fg_inds_no = len(fg_inds)
	
	elif len(bg_inds) < len(fg_inds) * 20:
	  enabled_fg_inds = npr.choice(
                fg_inds, size=(min(len(bg_inds), len(fg_inds))), replace=False)
	  labels[fg_inds] = -1
	  labels[enabled_fg_inds] = 1
	  fg_inds_no = len(enabled_fg_inds)
	  bg_inds_no = len(bg_inds)
	
	total_inds_no = bg_inds_no + fg_inds_no
        
        '''
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1 # used to be -1

        # subsample negative labels if we have too many
        num_bg = my_RPN_BATCHSIZE - np.sum(labels == 1)
        
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1 # used to be -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))
	
	total_inds_no = my_RPN_BATCHSIZE
	
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A))
        #labels = labels.reshape(-1)
        #labels = labels.reshape((1, 1, A * height, width))
        #labels = labels.reshape((1, A * height, width, 1)).transpose(0,2,1,3)
        #top[0].reshape(*labels.shape)
        #top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
        #top[1].reshape(*bbox_targets.shape)
        #top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
        #assert bbox_inside_weights.shape[1] == height
        #assert bbox_inside_weights.shape[2] == width
        #top[2].reshape(*bbox_inside_weights.shape)
        #top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
        #assert bbox_outside_weights.shape[1] == height
        #assert bbox_outside_weights.shape[2] == width
        #top[3].reshape(*bbox_outside_weights.shape)
        #top[3].data[...] = bbox_outside_weights
	
	return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, total_inds_no

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
