#!/usr/bin/env python
#c.f.https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
import sys
import time
import signal
import rospy
import tf
import csv
import pickle
import message_filters
import numpy as np
from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *

global force_arr

def save_data(force_all):
    with open('all_force.pkl', 'wb') as f:
        pickle.dump(force_all, f)

def sig_handler(signum, frame):
    sys.exit(1)

def wrench_cb(l_wrench, r_wrench):
    global force_arr
    lx_force =  l_wrench.wrench.force.x
    ly_force =  l_wrench.wrench.force.y
    lz_force =  l_wrench.wrench.force.z
    rx_force =  r_wrench.wrench.force.x
    ry_force =  r_wrench.wrench.force.y
    rz_force =  r_wrench.wrench.force.z
    #xy = x_force**2 + y_force**2
    #yz = y_force**2 + z_force**2
    #zx = z_force**2 + x_force**2
    #print("xy", xy)
    #print("yz", yz)
    #print("zx", zx)
    #print (y_force)
    force = np.array([[lx_force, ly_force, lz_force, rx_force, ry_force, rz_force]])
    force_arr = np.append(force_arr, force, axis=0)

def main():
    signal.signal(signal.SIGTERM, sig_handler)
    try:
        rospy.init_node('object_force')
        #sub = rospy.Subscriber('/left_endeffector/wrench', WrenchStamped, wrench_cb)
        #pub = rospy.Publisher('r_contact', Bool, queue_size=1)
        left_end_wrench = message_filters.Subscriber('/left_endeffector/wrench', WrenchStamped)
        right_end_wrench = message_filters.Subscriber('/right_endeffector/wrench', WrenchStamped)
        ts = message_filters.TimeSynchronizer([left_end_wrench, right_end_wrench], 10)
        ts.registerCallback(wrench_cb)
        rospy.spin()

    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        save_data(force_arr)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == "__main__":
    force_arr = np.array([[]]).reshape(-1,6)
    sys.exit(main())
