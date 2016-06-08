#!/usr/bin/env python
#-*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import random
import matplotlib as mpl



CLASSES = ('__background__',
        "鸡蛋","玉米","茄子-长","茄子-圆","番茄","胡萝卜",
	"青椒-长","青椒-圆","冬瓜","黄瓜","苦瓜","丝瓜",
	"西葫芦","洋葱","蒜苔","青菜","菜花","西兰花",
	"菠菜","芹菜","生菜","莴笋","空心菜","苹果","梨",
	"葡萄","橙","金针菇","平菇","香菇","酸奶","豆腐")

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'JoyData','Joyside','JPEGImages', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    # print '----scores----'
    # print scores.shape
    # print '----boxes-----'
    # print boxes.shape
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5	# setting thresh value
    NMS_THRESH = 0.3
    fig=plt.figure(figsize=(16, 9))
    ax=fig.gca()
    plt.axis('off')
    plt.tight_layout()    
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    ax.set_title(('detections with confidence thresh {:.1f}, and NMS thresh {:.1f}')
                          .format(CONF_THRESH,NMS_THRESH),fontsize=14)
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        # print keep
        dets = dets[keep, :]
        
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
	if len(inds) == 0:
	    continue
        
	for i in inds:
	    bbox = dets[i, :4]
	    score = dets[i, -1]
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1],
                             fill=False,edgecolor='red', linewidth=2.0))
	    ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score),
		             fontsize=14, fontproperties=zhfont,color='white',
		             bbox=dict(facecolor='blue', alpha=0.5))            
	    print class_name, score     
    plt.show()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ZF]',
                        choices=NETS.keys(), default='ZF')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    default_encoding = 'utf-8'
    if sys.getdefaultencoding() != default_encoding:
        reload(sys)
    sys.setdefaultencoding(default_encoding)
    zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    # modify follow 3 items
    prototxt = os.path.join(cfg.ROOT_DIR,'models', 'joyoung','ZF',
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR,'output','faster_rcnn_end2end',
                            'joy_side_trainval','zf_faster_rcnn_iter_70000.caffemodel')
    testfile=os.path.join(cfg.DATA_DIR,'JoyData','Joyside','ImageSets','test.txt')
    if args.cpu_mode:	
        caffe.set_mode_cpu()
    else:	
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)


    with open(testfile) as f:
        im_names=[x.strip() for x in f.readlines()] 
    random.shuffle(im_names)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)

