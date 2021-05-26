#!/usr/bin/env python
#c.f.https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
import sys
import time
import signal
import rospy
import tf
import csv
import pickle
from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *

def save_data(force_x):
    with open('x_force.pkl', 'wb') as f:
        pickle.dump(force_x, f)

def sig_handler(signum, frame):
    sys.exit(1)

def wrench_cb(msg):
    x_force =  msg.wrench.force.x
    y_force =  msg.wrench.force.y
    z_force =  msg.wrench.force.z
    #xy = x_force**2 + y_force**2
    #yz = y_force**2 + z_force**2
    #zx = z_force**2 + x_force**2
    #print("xy", xy)
    #print("yz", yz)
    #print("zx", zx)
    #print (y_force)
    force_x.append(x_force)


def main():
    signal.signal(signal.SIGTERM, sig_handler)
    try:
        rospy.init_node('object_force')
        #sub = rospy.Subscriber('/object_force', WrenchStamped, wrench_cb)
        sub = rospy.Subscriber('/left_endeffector/wrench', WrenchStamped, wrench_cb)
        #pub = rospy.Publisher('r_contact', Bool, queue_size=1)
        rospy.spin()

    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        save_data(force_x)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == "__main__":
    force_x = []
    force_y = []
    force_z = []
    sys.exit(main())
