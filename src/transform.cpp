#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

class Transform
{
  ros::NodeHandle nh;
  ros::Subscriber sub;
  ros::Publisher pub;
  tf::StampedTransform transform;


public:
  Transform()
  {
  ros::Subscriber sub = nh.subscribe("/plane_extraction_ssd/output", 100, &Transform::pc_callback, this);
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/hough_pointcloud", 100);
  }

  void pc_callback(sensor_msgs::PointCloud2 pc2){
    sensor_msgs::PointCloud2 pc2_transformed;
    tf::TransformListener listener;
    ROS_INFO("Sub!");
    listener.waitForTransform("heightmap_center", "head_mount_kinect_rgb_optical_frame", ros::Time(0), ros::Duration(100));
    listener.lookupTransform("heightmap_center", "head_mount_kinect_rgb_optical_frame", ros::Time(0), transform);
    pcl_ros::transformPointCloud("heightmap_center", transform, pc2, pc2_transformed);
    ROS_INFO("I got a transform!");
    pub.publish(pc2_transformed);
  }
};

int main(int argc, char** argv){
  ros::init(argc,argv,"transform_test");

  // wait until getting tf 
  //while(true){
  //  try{
  //    // save tf when getting tf
  //    listener.lookupTransform("heightmap_center", "head_mount_kinect_rgb_optical_frame", ros::Time(0), transform);
  //  }
  //  catch(tf::TransformException ex){
  //    ROS_ERROR("%s",ex.what());
  //    ros::Duration(1.0).sleep();
  //  }
  //}
  ros::spin();
  return 0;
}
