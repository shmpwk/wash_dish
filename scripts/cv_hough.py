import cv2
import math
import numpy as np
import rospy

from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from cv_bridge import CvBridge

def image_cb(msg):
    bridge = CvBridge()
    #img = bridge.imgmsg_to_cv2(msg, 'passthrough')
    img = bridge.imgmsg_to_cv2(msg, 'rgb8')
    #img = cv2.imread('ellipse2.png', cv2.IMREAD_COLOR)
    # gray
    #gray1 = cv2.bitwise_and(img[:,:,0], img[:,:,1])
    #gray1 = cv2.bitwise_and(gray1, img[:,:,2])
    #print(gray1)
    #gray1 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    #gray1 = np.array(gray1, dtype='int')
    gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('resimg', gray1)
    # binary
    _, binimg = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binimg = cv2.bitwise_not(binimg)

    # black to gray
    bimg = binimg // 4 + 255 * 3 //4
    resimg = cv2.merge((bimg,bimg,bimg)) 
    # get edge
    image,contours,hierarchy =  cv2.findContours(binimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        # fitting ellipse
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            print(ellipse)

            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])

            # write ellipse
            resimg = cv2.ellipse(resimg,ellipse,(255,0,0),2)
            cv2.drawMarker(resimg, (cx,cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
            cv2.putText(resimg, str(i+1), (cx+3,cy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 1,cv2.LINE_AA)

    cv2.imshow('resimg',resimg)
    cv2.waitKey()

rospy.init_node('touch_detect')
#position_sub = rospy.Subscriber('/kinect_head/rgb/image_rect_color', Image, image_cb)
position_sub = rospy.Subscriber('/colorize_float_image_filtered_heightmap/output', Image, image_cb)
#pub = rospy.Publisher('r_contact', Bool, queue_size=1)
#pub = rospy.Publisher('object_force', PoseStamped, queue_size=1)
pub = rospy.Publisher('hough_image', Image, queue_size=1)

rospy.spin()
 
