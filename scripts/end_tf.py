import rospy
import sys
import time
import signal
import tf
import pickle
import tf2_py as tf2
import numpy as np
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

global pos_arr

def save_data(pos_all):
    with open('all_pos.pkl', 'wb') as f:
        pickle.dump(pos_all, f)

def sig_handler(signum, frame):
    sys.exit(1)

def main():
    signal.signal(signal.SIGTERM, sig_handler)
    try:
        rospy.init_node('end_tf_pub')
        global pos_arr
        pub = rospy.Publisher('/end_tf', PoseStamped, queue_size=10)
        target_frame = "l_gripper_tool_frame"
        source_frame = "torso_lift_link"
        #tf_buffer = tf2_ros.Buffer()
        #tf_listener = tf2_ros.TransformListener(tf_buffer)
        #tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(10.0))
        listener = tf.TransformListener()
        listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
        while not rospy.is_shutdown():
            try:
                (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
                posestamped = PoseStamped()
                pos = Point()
                pos = trans
                ori = Quaternion()
                ori = rot
                posestamped.pose = (pos, ori)
                transrot = trans + rot
                pos_arr = np.append(pos_arr, np.array(transrot).reshape(-1,7), axis=0)
                pub.publish(posestamped)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        save_data(pos_arr)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__=="__main__":
    pos_arr = np.array([[]]).reshape(-1,7)
    sys.exit(main())
