#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

tf::StampedTransform transform;
void pc_callback(sensor_msgs::PointCloud2 pc2){
  sensor_msgs::PointCloud2 pc2_transformed;

  pcl_ros::transformPointCloud("heightmap_center", transform, pc2, pc2_transformed);
}

int main(int argc, char** argv){
  ros::init(argc,argv,"transform_test");
  ros::NodeHandle nh;
  ros::Rate loop_rate(10);
  ros::Subscriber sub = nh.subscribe("/plane_extraction_ssd/output", 1, pc_callback);
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/hough_pointcloud", 1);

  tf::TransformListener listener;
  // wait until getting tf 
  while(true){
    try{
      // save tf when getting tf
      listener.lookupTransform("heightmap_center", "head_mount_kinect_rgb_optical_frame", ros::Time(0), transform);
      ROS_INFO("I got a transform!");
      break;
    }
    catch(tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
    }
  }
  // do nothing
  while(ros::ok()){
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
