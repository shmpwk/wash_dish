#include <ros/ros.h>
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
  image_transport::Publisher image_pub_;

  public:
    ImageConverter() : it_(nh_){
      image_sub_ = it_.subscribe("/colorize_float_image_heightmap/output", 1, &ImageConverter::imageCb, this);
      image_pub_ = it_.advertise("/hough_circle_rect", 1);
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
        //cv::Mat src_img = cv::imread("./stuff.jpg", 1);
        cv::Mat src_img = cv_ptr->image;

        cv::Mat gray_img, bin_img, cv_image2, hsv_image, color_mask;
        //cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);
        cv::cvtColor(cv_ptr->image, hsv_image, CV_BGR2HSV);
        cv::inRange(hsv_image, cv::Scalar(0, 0, 0, 0) , cv::Scalar(20, 255, 255, 0), color_mask); 

        cv::bitwise_and(cv_ptr->image, cv_ptr->image, cv_image2, color_mask);
        cv::cvtColor(cv_image2, gray_img, CV_BGR2GRAY);
        //cv::threshold(gray_img, bin_img, 0.00000000000000000001, 255, CV_THRESH_BINARY); 
        cv::adaptiveThreshold(gray_img, bin_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 5);

        std::vector<std::vector<cv::Point> > contours;
        // binary
        //cv::threshold(gray_img, bin_img, 28.9999999999999982236432, 255, cv::THRESH_BINARY);
        // get edge
        cv::findContours(bin_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        jsk_recognition_msgs::RectArray rect_msg;
        rect_msg.header = msg->header;

        // 各輪郭をcontourArea関数に渡し、最大面積を持つ輪郭を探す
        double max_area=0;
        int max_area_contour=-1;
        for(int j=0; j<contours.size(); j++){
            double area = cv::contourArea(contours.at(j));
            if(max_area<area){
                max_area=area;
                max_area_contour=j;
            }
        }

        // 最大面積を持つ輪郭の最小外接円を取得
        //cv::minEnclosingCircle(contours.at(max_area_contour), center, radius);
        for(int j=0; j<contours.size(); j++){
            cv::minEnclosingCircle(contours.at(j), center, radius);
            if(radius>30 && radius<60){ 
                ROS_INFO("radius = %f", radius);

                // 最小外接円を描画
                cv::circle(cv_ptr->image, center, radius, cv::Scalar(0,0,255),3,4);
                cv::circle(src_img, center, radius, cv::Scalar(0,0,255),3,4);
                cv::circle(bin_img, center, radius, cv::Scalar(0,0,255),3,4);
            }
        }

        // 画面中心から最小外接円の中心へのベクトルを描画
        //p1 = cv::Point2f(cv_ptr->image.size().width/2,cv_ptr->image.size().height/2);
        //cv::arrowedLine(cv_ptr->image, p1, center, cv::Scalar(0, 255, 0, 0), 3, 8, 0, 0.1);  

        // ウインドウ表示                                                                         
        cv::imshow("Result Image", src_img);
        cv::imshow("Gray Image", gray_img);
        cv::imshow("Color mask Image", color_mask);
        cv::imshow("Binary Image", bin_img);
        cv::waitKey(3);
  
        // エッジ画像をパブリッシュ。OpenCVからROS形式にtoImageMsg()で変換。                                                            
        image_pub_.publish(cv_ptr->toImageMsg());
    
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

