#include <ros/ros.h> 
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/String.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

cv::Mat latest_image;
bool image_received = false;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        // 使用cv_bridge将ROS图像消息转换为OpenCV图像
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        latest_image = cv_ptr->image;
        image_received = true;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert image: %s", e.what());
    }
}

void saveImage(const std::string& filename)
{
    if (image_received && !latest_image.empty())
    {
        cv::imwrite(filename, latest_image);  // 保存图像
        ROS_INFO("Image saved to %s", filename);
    }
    else
    {
        ROS_ERROR("No image received yet or image is empty!");
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "nav_photo");

    // 使用私有命名空间来获取私有参数
    ros::NodeHandle nh("~");

    // 获取参数，如果没有设置，则使用默认话题 "/ucar_camera/image_raw"
    std::string camera_topic = nh.param("camera_topic", std::string("/usb_cam/image_raw"));
    
    // 订阅摄像头图像话题
    ros::Subscriber image_sub = nh.subscribe(camera_topic, 1, imageCallback);

    // 等待图像回调接收图像
    ROS_INFO("Waiting for image...");
    ros::Rate loop_rate(1);
    while (ros::ok() && !image_received)
    {
        ros::spinOnce();  // 处理回调
        loop_rate.sleep();  // 等待
    }

    // 拍照并保存图像
    ROS_INFO("Taking picture...");
    saveImage("/home/ucar/ucar_ws/src/ucar_nav/maps/picture_first_point.jpg");  // 保存图像

    return 0;
}