#!/usr/bin/env python

import rosbag
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from jsk_recognition_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *

def tf_converter(source):
    listener = tf.TransformListener()
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    #target_frame = "segmentation_decomposeroutput00"
    target_frame = "base_link"
    source_frame = "l_gripper_tool_frame"
    listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(100.0))
    listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
    transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(100))
    pose_transformed = tf2_geometry_msgs.do_transform_pose(source, transform)
    #target = listener.transformPose(target_frame, source)
    #return target
    return pose_transformed
    #while not rospy.is_shutdown():
    #    try:
    #        listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(10000000.0))
    #        target = listener.transformPose(target_frame, source)
    #        print("aaaa")
    #        return target
    #    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #        continue

def wrench_cb(msg):
    posestamped = PoseStamped()
    pose = posestamped.pose
    pose.position.x = msg.wrench.force.x
    pose.position.y = msg.wrench.force.y
    pose.position.z = msg.wrench.force.z
    x_torque =  msg.wrench.torque.x
    y_torque =  msg.wrench.torque.y
    z_torque =  msg.wrench.torque.z
    q = tf.transformations.quaternion_from_euler(x_torque, y_torque, z_torque)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    header = posestamped.header
    header.stamp = rospy.Time.now()
    header.frame_id = "l_gripper_tool_frame"
    obj_posestamped = tf_converter(posestamped)

    r =tf.transformations.euler_from_quaternion((obj_posestamped.pose.orientation.x, obj_posestamped.pose.orientation.y, obj_posestamped.pose.orientation.z, obj_posestamped.pose.orientation.w))
    obj_wrench = WrenchStamped()
    obj_wrench.wrench.force.x = obj_posestamped.pose.position.x
    obj_wrench.wrench.force.y = obj_posestamped.pose.position.y
    obj_wrench.wrench.force.z = obj_posestamped.pose.position.z
    obj_wrench.wrench.torque.x = r[0]
    obj_wrench.wrench.torque.y = r[1]
    obj_wrench.wrench.torque.z = r[2]
    obj_wrench.header = obj_posestamped.header
    obj_wrench.header.stamp = rospy.Time.now()
    obj_wrench.header.frame_id = "base_link"

    pub.publish(obj_wrench)

    #xy = x_force**2 + y_force**2
    #yz = y_force**2 + z_force**2
    #zx = z_force**2 + x_force**2
    #print("xy", xy)
    #print("yz", yz)
    #print("zx", zx)
    #print (y_force)
    #if y_force > 3:
    #    pub.publish(1)
    #    print("touch")
    #else :
    #    pub.publish(0)
    #    print("nothing")
 
 
rospy.init_node('touch_detect')
position_sub = rospy.Subscriber('/left_endeffector/wrench', WrenchStamped, wrench_cb)
#pub = rospy.Publisher('r_contact', Bool, queue_size=1)
#pub = rospy.Publisher('object_force', PoseStamped, queue_size=1)
pub = rospy.Publisher('object_force', WrenchStamped, queue_size=1)

rospy.spin()
       
