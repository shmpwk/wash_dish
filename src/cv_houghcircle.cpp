#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <ros/ros.h>
#include <math.h>
#include <cv.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <jsk_recognition_msgs/RectArray.h>

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Publisher pub_;

  public:
    ImageConverter() : it_(nh_){
      image_sub_ = it_.subscribe("/colorize_float_image_heightmap/output", 1, &ImageConverter::imageCb, this);
      pub_ = nh_.advertise<jsk_recognition_msgs::RectArray>("/hough_circle_rect", 1);
    }

    ~ImageConverter(){
      cv::destroyWindow(OPENCV_WINDOW);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg){
      cv::Point2f center, p1;
      float radius;

      cv_bridge::CvImagePtr cv_ptr;
      try{
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat src_img = cv_ptr->image;

        cv::Mat gray_img, bin_img, cv_image2, hsv_image, color_mask;
        //cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);
        cv::cvtColor(cv_ptr->image, hsv_image, CV_BGR2HSV);
        cv::inRange(hsv_image, cv::Scalar(0, 0, 0, 0) , cv::Scalar(20, 255, 255, 0), color_mask); 

        cv::bitwise_and(cv_ptr->image, cv_ptr->image, cv_image2, color_mask);
        cv::cvtColor(cv_image2, gray_img, CV_BGR2GRAY);
        cv::GaussianBlur(gray_img, gray_img, cv::Size(1, 1), 1, 1);
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(gray_img, circles, cv::HOUGH_GRADIENT,
                     4, 40, 200, 40, 40, 50);

        double tmp = 10000;
        jsk_recognition_msgs::RectArray rect_msg;
        rect_msg.header = msg->header;
        for(size_t i = 0; i < circles.size(); i++ )
        {
             cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
             int radius = cvRound(circles[i][2]);
             if (40<center.x && center.x<90 && 40<center.y && center.y<90){
                 // 円の中心を描画します．
                 //cv::Mat3b fuga;
                 //const cv::Vec3b& hoge = fuga(center);
                 //std::cout << typeid(hoge) << std::endl;
                 cv::circle(cv_image2, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
                 // 円を描画します．
                 cv::circle(cv_image2, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
                 // Get more center point
                 if (tmp-60 > center.x-60){
                     tmp = center.x;
                     jsk_recognition_msgs::Rect rect;
                     rect.x = center.x;
                     rect.y = center.y;
                     rect.width = 2 * radius;
                     rect.height = 2 * radius;
                     rect_msg.rects.push_back(rect);
                 }
             }
        }
        //namedWindow( "circles", 1 );
        cv::imshow( "circles", cv_image2 );
        cv::imshow("Gray Image", gray_img);
        cv::waitKey(3);
 
        //// エッジ画像をパブリッシュ。OpenCVからROS形式にtoImageMsg()で変換。                                                            
        pub_.publish(rect_msg);
    
      }catch(CvErrorCallback){
        
      }
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cv_circle");
  ImageConverter ic;
  ros::spin();
  return 0;
}

