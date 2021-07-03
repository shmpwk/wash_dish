import rospy
import tf
import tf2_ros
import tf2_py as tf2
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

if __name__=="__main__":
    try:
        rospy.init_node('end_tf_pub')
        pub = rospy.Publisher('/end_tf', PoseStamped, queue_size=10)
        target_frame = "l_gripper_tool_frame"
        source_frame = "torso_lift_link"
        #tf_buffer = tf2_ros.Buffer()
        #tf_listener = tf2_ros.TransformListener(tf_buffer)
        #tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(10.0))
        listener = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
                posestamped = PoseStamped()
                pos = Point()
                pos = trans
                ori = Quaternion()
                ori = rot
                posestamped.pose = (pos, ori)
                pub.publish(posestamped)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
    except rospy.ROSInterruptException: pass
