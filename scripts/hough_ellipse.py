#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import rospy
from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.color import rgb2gray,rgba2rgb
from cv_bridge import CvBridge

def image_cb(msg):
    # Load picture, convert to grayscale and detect edges
    bridge = CvBridge()
    image_rgb = bridge.imgmsg_to_cv2(msg, 'passthrough')
    #image_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    image_gray = rgb2gray(image_rgb)
    edges = canny(image_gray,sigma=1.0,low_threshold=0.1, high_threshold=0.8)

    fig, ax = plt.subplots(dpi=140)
    ax.imshow(edges, cmap=plt.cm.gray)
    plt.savefig('fabric_mark_ellipse_edge.jpg',dpi=130)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=4, threshold=1,
                           min_size=300, max_size=240)
    result.sort(order='accumulator')
    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True,dpi=150)
    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)


    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)
    plt.savefig("daen_hough.jpg",dpi=100)
    plt.show()

rospy.init_node('touch_detect')
position_sub = rospy.Subscriber('/kinect_head/rgb/image_rect_color', Image, image_cb)
#pub = rospy.Publisher('r_contact', Bool, queue_size=1)
#pub = rospy.Publisher('object_force', PoseStamped, queue_size=1)
pub = rospy.Publisher('hough_image', Image, queue_size=1)

rospy.spin()
 
