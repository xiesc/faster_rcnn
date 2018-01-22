#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from time import time

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

RUNNING = False
IMAGE = Image()
PUBLISHER = None

def img_callback(msg):
    global IMAGE
    global RUNNING
    if RUNNING:
        pass # rospy.logwarn('Detection already running, message omitted')
    else:
        RUNNING = True
        IMAGE = msg

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    rospy.loginfo('Detection took {:.3f}s for '
           '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    for cls_ind, cls_name in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls_name, dets, thresh=CONF_THRESH)

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    global PUBLISHER
    global bridge

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        if class_name == 'person':
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 3)
        else:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 3)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1]),
                    0, 1, (0,0,0),3)
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')
    
    PUBLISHER.publish(bridge.cv2_to_imgmsg(im))
    rospy.loginfo('Detection results published')


def ros_init():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    global PUBLISHER
    rospy.init_node('faster_rcnn_processor', anonymous=True)

    rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)
    PUBLISHER = rospy.Publisher('/usb_cam/image_detect', Image, queue_size=5)

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    net_type = 'ZF'
    prototxt = '/home/openicv/detection/Faster_R-CNN/py-faster-rcnn/models/pascal_voc/' + net_type + '/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/openicv/detection/Faster_R-CNN/py-faster-rcnn/data/faster_rcnn_models/' + net_type + '_faster_rcnn_final.caffemodel'

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    rospy.loginfo('Starting Fast RCNN node')
    ros_init()

    rate = rospy.Rate(10)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        if RUNNING:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(IMAGE, "bgr8")
            # rospy.loginfo("Image received %s", cv2_img.shape)

            # run Faster RCNN
            demo(net, cv2_img)

            RUNNING = False

        rate.sleep()
