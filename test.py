import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import json
import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string(
    'test_data_path',
    '/data/20180809/icdar2017/test_images/',
    '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
#tf.app.flags.DEFINE_string(
#    'checkpoint_path',
#    '/workspace/imagenet-data/EAST/temp_test/east_icdar2015_resnet_v1_50_rbox/',
#    '')
tf.app.flags.DEFINE_string(
    'checkpoint_path',
    '/data/20180809/IncepText/model_save/',
    '')
tf.app.flags.DEFINE_string(
    'output_dir',
    '/data/20180809/IncepText/result/',
    '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_string('save_pic_jietu','/workspace/imagenet-data/EAST/xuyanqi/crop_result_lsc/','')
tf.app.flags.DEFINE_string('result_last_tsv','temp_tsv.tsv','')
import model
from icdar import restore_rectangle
from math import *

FLAGS = tf.app.flags.FLAGS

def rank_boxes(boxes):
    def getKey(item):
        return item[1] #sort by y1
    sorted_boxes = sorted(boxes,key=getKey)
    return sorted_boxes
def ndarray_sort(arr1):
    result_list=[]
    for arr in arr1:
        temp=[]
        for ss in arr:
            temp.append(ss[0])
            temp.append(ss[1])
        result_list.append(temp)
    result_list = rank_boxes(result_list)
    array_result = np.array(result_list).reshape(-1,4,2)
    return array_result

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG','bmp']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=768):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(
            max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    #print(resize_w)
    #print(resize_h)
    return im, (ratio_h, ratio_w)


def detect(
        score_map,
        geo_map,
        timer,
        score_map_thresh=0.8,
        box_thresh=0.1,
        nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(
        xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is
    # different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape(
            (-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) %
           4, (min_axis + 2) %
           4, (min_axis + 3) %
           4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(
            255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(
        matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(
        matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut


def filter_img(img):
    if img.shape[0] > img.shape[1] * 1.5:
        img = np.rot90(img)
    scale = float(img.shape[0]) / 32.0
    if scale == 0:
        return img
    w = int(float(img.shape[1]) / scale)
    if w > 280:
        w = 280
        img = cv2.resize(img, (w, 32), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (w, 32))
        expand = 280 - w

        r = img[:, img.shape[1] - 1, 0].mean()
        g = img[:, img.shape[1] - 1, 1].mean()
        b = img[:, img.shape[1] - 1, 2].mean()

        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            0,
            expand,
            cv2.BORDER_CONSTANT,
            value=(
                r,
                g,
                b))

    return img


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(
            tf.float32, shape=[
                None, None, None, 3], name='input_images')
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            #print(type(ckpt_state))
            model_path = os.path.join(
                FLAGS.checkpoint_path, os.path.basename(
                    ckpt_state.model_checkpoint_path))
	    #print(model_path)
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            with open(FLAGS.result_last_tsv,"w")as fw:
                for im_fn in im_fn_list:
		    #print(im_fn)
                    last_name = im_fn.split("/")[-1]
		    im_just_for_test = cv2.imread(im_fn)
		    #print("results:"+str(im_just_for_test.shape))
                    im = cv2.imread(im_fn)[:, :, ::-1]
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(im)

                    timer = {'net': 0, 'restore': 0, 'nms': 0}
                    start = time.time()
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={
                                               input_images: [im_resized]})
                    timer['net'] = time.time() - start

                    boxes, timer = detect(
                        score_map=score, geo_map=geometry, timer=timer)
                    print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                        im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                    duration = time.time() - start_time
                    print('[timing] {}'.format(duration))
                    temp_i = 0
                    # save to file
                    dict_result_temp = dict()
                    dict_result_temp["bboxes"] = list()
                    #print(type(boxes))
                    #boxes = ndarray_sort(boxes)
                    if boxes is not None:
                        res_file = os.path.join(
                            FLAGS.output_dir,
                            '{}.txt'.format(
                                "res_" + os.path.basename(im_fn).split('.')[0]))
                        save_name_pic = os.path.basename(im_fn).split('.')[0]
                        with open(res_file, 'w') as f:
                            for box in boxes:
                                #print(box)
                                single_temp = []
                                pt1 = []
                                pt2 = []
                                pt3 = []
                                pt4 = []
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                if np.linalg.norm(
                                        box[0] -
                                        box[1]) < 5 or np.linalg.norm(
                                        box[3] -
                                        box[0]) < 5:
                                    continue
				t_00 = int(box[0,0])
				t_01 = int(box[0,1])
				t_10 = int(box[1,0])
				t_11 = int(box[1,1])
				t_20 = int(box[2,0])
				t_21 = int(box[2,1])
				t_30 = int(box[3,0])
				t_31 = int(box[3,1])
				if t_00>=0 and t_01>=0 and t_10>=0 and t_11>=0 and t_20>=0 and t_21>=0 and t_30>=0 and t_31>=0:
                                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(int(box[0, 0]), int(box[0, 1]), int(
                                        box[1, 0]), int(box[1, 1]), int(box[2, 0]), int(box[2, 1]), int(box[3, 0]), int(box[3, 1]), ))
                                    for i in range(4):
                                        for j in range(2):
                                            single_temp.append(box[i][j])
                                    dict_result_temp["bboxes"].append(single_temp)
                                    #pt1.append(box[0, 0])
                                    #pt1.append(box[0, 1])
                                    #pt2.append(box[1, 0])
                                    #pt2.append(box[1, 1])
                                    #pt3.append(box[2, 0])
                                    #pt3.append(box[2, 1])
                                    #pt4.append(box[3, 0])
                                    #pt4.append(box[3, 1])
                                    #partImg = dumpRotateImage(im, degrees(atan2(box[1,1] - box[0,1], box[1,0] - box[0,0])), pt1, pt2, pt3, pt4)
                                    #partImg_new = filter_img(partImg)
                                    #cv2.imwrite(FLAGS.save_pic_jietu+save_name_pic+"_"+str(temp_i)+".png",partImg_new[:,:,::-1])
                                    #temp_i = temp_i+1
                                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape(
                                        (-1, 1, 2))], True, color=(255, 0, 0), thickness=2)
                    if not FLAGS.no_write_images:
                        img_path = os.path.join(
                            FLAGS.output_dir, os.path.basename(im_fn))
                        cv2.imwrite(img_path, im[:, :, ::-1])
                    fw.write(last_name+"\t"+str(dict_result_temp)+"\n")


if __name__ == '__main__':
    tf.app.run()
