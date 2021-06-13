import rospy
import tf2_ros
import tf2_py as tf2
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

def transform_callback(source):
    """
    Transform from world (or camera frame?) to local (segmentation_decomposeroutput00 or projected point?)/
    """
    #listener = tf.TransformListener()
    target_frame = "heightmap_center"
    source_frame = "head_mount_kinect_rgb_optical_frame"
    #listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    #tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(10.0))
    trans = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(10000))
    target = do_transform_cloud(source, trans)
    pub.publish(target)

if __name__=="__main__":
    try:
        rospy.init_node('pointcloud_server')
        rospy.Subscriber('/plane_extraction_ssd/output', PointCloud2, transform_callback, queue_size=1000)
        pub = rospy.Publisher('/hough_pointcloud', PointCloud2, queue_size=100)
        rospy.spin()
    except rospy.ROSInterruptException: pass
