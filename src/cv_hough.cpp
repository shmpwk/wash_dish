#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <jsk_recognition_msgs/RectArray.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher pub_;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/colorize_float_image_heightmap/output", 1,
      &ImageConverter::imageCb, this);
    pub_ = nh_.advertise<jsk_recognition_msgs::RectArray>("/hough_rect", 1);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      //cv::Mat src_img = cv::imread("./stuff.jpg", 1);
      cv::Mat src_img = cv_ptr->image;

      cv::Mat gray_img, bin_img;
      cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);

      std::vector<std::vector<cv::Point> > contours;
      // binary
      cv::threshold(gray_img, bin_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
      // get edge
      cv::findContours(bin_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
      jsk_recognition_msgs::RectArray rect_msg;
      rect_msg.header = msg->header;
      
      for(int i = 0; i < contours.size(); ++i) {
        size_t count = contours[i].size();
        if(count < 150 || count > 1000) continue; // remove too small or big edge

        cv::Mat pointsf;
        cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
        // fitting ellipse
        cv::RotatedRect box = cv::fitEllipse(pointsf);

        if (box.size.width < 80 || box.size.height < 80) continue; //remove too small or big edge
        // draw ellipse
        cv::ellipse(src_img, box, cv::Scalar(0,0,0), 2, CV_AA);
        cv::drawMarker(src_img, box.center, cv::Scalar(0,0,0));
        jsk_recognition_msgs::Rect rect;
        rect.x = box.center.x;
        rect.y = box.center.y;
        rect.width = box.size.width;
        rect.height = box.size.height;
        rect_msg.rects.push_back(rect);
      }

      cv::namedWindow("fit ellipse", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
      cv::namedWindow("bin image", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
      cv::imshow("fit ellipse", src_img);
      cv::imshow("bin image", bin_img);
      cv::waitKey(0);
      // Output 
      pub_.publish(rect_msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Draw an example circle on the video stream
    //if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //  cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::waitKey(3);

  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
