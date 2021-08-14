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

        cv::Mat gray_img, bin_img;
        cv::cvtColor(src_img, gray_img, CV_BGR2GRAY);

        std::vector<std::vector<cv::Point> > contours;
        // binary
        cv::threshold(gray_img, bin_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
        // get edge
        cv::findContours(bin_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        jsk_recognition_msgs::RectArray rect_msg;
        rect_msg.header = msg->header;

        // 各輪郭をcontourArea関数に渡し、最大面積を持つ輪郭を探す
        double max_area=0;
        int max_area_contour=-1;
        ROS_INFO("%f", contours.size());
        for(int j=0; j<contours.size(); j++){
            double area = cv::contourArea(contours.at(j));
            if(max_area<area){
                max_area=area;
                max_area_contour=j;
            }
        }

        // 最大面積を持つ輪郭の最小外接円を取得
        cv::minEnclosingCircle(contours.at(max_area_contour), center, radius);
        ROS_INFO("radius = %f", radius);

        // 最小外接円を描画
        cv::circle(cv_ptr->image, center, radius, cv::Scalar(0,0,255),3,4);
        cv::circle(src_img, center, radius, cv::Scalar(0,0,255),3,4);
        cv::circle(bin_img, center, radius, cv::Scalar(0,0,255),3,4);

        // 画面中心から最小外接円の中心へのベクトルを描画
        p1 = cv::Point2f(cv_ptr->image.size().width/2,cv_ptr->image.size().height/2);
        cv::arrowedLine(cv_ptr->image, p1, center, cv::Scalar(0, 255, 0, 0), 3, 8, 0, 0.1);  

        // 画像サイズは縦横1/4に変更
        cv::Mat cv_half_image, cv_half_image2, cv_half_image3, cv_half_image4, cv_half_image5;
        cv::resize(cv_ptr->image, cv_half_image,cv::Size(),0.5,0.5);
        cv::resize(src_img, cv_half_image2,cv::Size(),0.5,0.5);
        //cv::resize(cv_ptr3->image, cv_half_image3,cv::Size(),0.5,0.5);
        cv::resize(gray_img, cv_half_image4,cv::Size(),0.5,0.5);
        cv::resize(bin_img, cv_half_image5,cv::Size(),0.5,0.5);

        // ウインドウ表示                                                                         
        cv::imshow("Original Image", cv_half_image);
        cv::imshow("Result Image", cv_half_image2);
        //cv::imshow("Edge Image", cv_half_image3);
        //cv::imshow("Gray Image", cv_half_image4);
        cv::imshow("Binary Image", cv_half_image5);
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

