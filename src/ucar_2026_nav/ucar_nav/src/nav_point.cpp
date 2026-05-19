/*#include <ros/ros.h>  
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

std::string scanQRCode(const cv::Mat& image)
{
    // 使用 OpenCV 的二维码解码器
    cv::QRCodeDetector qrDecoder;
    std::string qr_data;
    if (qrDecoder.detectAndDecode(image, qr_data))
    {
        return qr_data;  // 返回扫描到的二维码内容
    }
    else
    {
        ROS_ERROR("QR Code not detected");
        return "";
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "nav_point");
  
    // 使用私有命名空间来获取私有参数
    ros::NodeHandle nh("~");

    // 获取参数，如果没有设置，则使用默认话题 "/usb_cam/image_raw"
    std::string camera_topic = nh.param("camera_topic", std::string("/usb_cam/image_raw"));
    
    // 订阅摄像头图像话题
    ros::Subscriber image_sub = nh.subscribe(camera_topic, 1, imageCallback);

    // Tell the action client that we want to spin a thread by default
    MoveBaseClient ac("move_base", true);

    // Wait for the action server to come up
    while(!ac.waitForServer(ros::Duration(5.0)))
    {
        ROS_INFO("Waiting for the move_base action server to come up");
    }

    move_base_msgs::MoveBaseGoal goal;

    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();

    goal.target_pose.pose.position.x = 1.1;
    goal.target_pose.pose.position.y = 0.48;
    goal.target_pose.pose.orientation.x = 0.0;
    goal.target_pose.pose.orientation.y = 0.0;
    goal.target_pose.pose.orientation.z = 1.0;
    goal.target_pose.pose.orientation.w = 0.0;

    ROS_INFO("Sending first goal");
    ac.sendGoal(goal);

    ac.waitForResult();

    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("First mission complete!");
    else
    {
        ROS_INFO("First mission failed ...");
        return 0;  // Exit if the first goal fails
    }
  
    // 等待图像回调接收图像
    ROS_INFO("Waiting for image...");
    ros::Rate loop_rate(1);
    while (ros::ok() && !image_received)
    {
        ros::spinOnce();  // 处理回调
        loop_rate.sleep();  // 等待
    }

    // 扫描二维码
    ROS_INFO("Scanning QR code...");
    std::string qr_code_data = scanQRCode(latest_image);

    if (!qr_code_data.empty())
    {
        // 将扫描到的信息发送到参数服务器
        ROS_INFO("QR code data: %s", qr_code_data.c_str());
        nh.setParam("qr_code_info", qr_code_data);  // 将二维码信息设置为 ROS 参数
    }
    else
    {
        ROS_ERROR("Failed to scan QR code.");
        return 0;  // 如果二维码扫描失败，则退出
    }

    // Set the second goal (target 2)
    goal.target_pose.pose.position.x = 0.45;
    goal.target_pose.pose.position.y = 2.0;
    goal.target_pose.pose.orientation.x = 0.0;
    goal.target_pose.pose.orientation.y = 0.0;
    goal.target_pose.pose.orientation.z = 0.3827;
    goal.target_pose.pose.orientation.w = 0.9279;

    ROS_INFO("Sending second goal");
    ac.sendGoal(goal);
  
    ac.waitForResult();

    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("Second mission complete!");
    else
        ROS_INFO("Second mission failed ...");

    return 0;
}*/
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
  ros::init(argc, argv, "nav_point");
  
  // 使用私有命名空间来获取私有参数
  ros::NodeHandle nh("~");

  // 获取参数，如果没有设置，则使用默认话题 "/ucar_camera/image_raw"
  std::string camera_topic = nh.param("camera_topic", std::string("/usb_cam/image_raw"));
    
  // 订阅摄像头图像话题
  ros::Subscriber image_sub = nh.subscribe(camera_topic, 1, imageCallback);

  //tell the action client that we want to spin a thread by default
  MoveBaseClient ac("move_base", true);

  //wait for the action server to come up
  while(!ac.waitForServer(ros::Duration(5.0)))
  {
    ROS_INFO("Waiting for the move_base action server to come up");
  }

  move_base_msgs::MoveBaseGoal goal;

  goal.target_pose.header.frame_id = "map";
  goal.target_pose.header.stamp = ros::Time::now();

  goal.target_pose.pose.position.x = 1.1;
  goal.target_pose.pose.position.y = 0.48;
  goal.target_pose.pose.orientation.x = 0.0;
  goal.target_pose.pose.orientation.y = 0.0;
  goal.target_pose.pose.orientation.z = 1.0;
  goal.target_pose.pose.orientation.w = 0.0;

  ROS_INFO("Sending goal");
  ac.sendGoal(goal);

  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("First mission complete!");
  else
  {
    ROS_INFO("First mission failed ...");
    return 0;  // Exit if the first goal fails
  }
  
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

  // Set the 2 goal (target 2)
  goal.target_pose.pose.position.x = 0.45;
  goal.target_pose.pose.position.y = 2.0;
  goal.target_pose.pose.orientation.x = 0.0;
  goal.target_pose.pose.orientation.y = 0.0;
  goal.target_pose.pose.orientation.z = 0.3827;
  goal.target_pose.pose.orientation.w = 0.9279;

  ROS_INFO("Sending 2 goal");
  ac.sendGoal(goal);
  
  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Second mission complete!");
  else
    ROS_INFO("Second mission failed ...");
    
  // Set the 3 goal (target 3)
  goal.target_pose.pose.position.x = 1.2;
  goal.target_pose.pose.position.y = 3.43;
  goal.target_pose.pose.orientation.x = 0.0;
  goal.target_pose.pose.orientation.y = 0.0;
  goal.target_pose.pose.orientation.z = 0.7071;
  goal.target_pose.pose.orientation.w = 0.7071;

  ROS_INFO("Sending 3 goal");
  ac.sendGoal(goal);
  
  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Second mission complete!");
  else
    ROS_INFO("Second mission failed ...");

  // Set the 4 goal (target 4)
  goal.target_pose.pose.position.x = 1.85;
  goal.target_pose.pose.position.y = 3.95;
  goal.target_pose.pose.orientation.x = 0.0;
  goal.target_pose.pose.orientation.y = 0.0;
  goal.target_pose.pose.orientation.z = 0.0;
  goal.target_pose.pose.orientation.w = 1.0;

  ROS_INFO("Sending 4 goal");
  ac.sendGoal(goal);
  
  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Second mission complete!");
  else
    ROS_INFO("Second mission failed ...");
    
  // Set the 5 goal (target 5)
  goal.target_pose.pose.position.x = 3.20;//2.85
  goal.target_pose.pose.position.y = 4.15;//3.95
  goal.target_pose.pose.orientation.x = 0.0;
  goal.target_pose.pose.orientation.y = 0.0;
  goal.target_pose.pose.orientation.z = 0.7071;//0
  goal.target_pose.pose.orientation.w = 0.7071;//1

  ROS_INFO("Sending 5 goal");
  ac.sendGoal(goal);
  
  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Second mission complete!");
  else
    ROS_INFO("Second mission failed ...");
  
  // Set the 6 goal (target 6)
  goal.target_pose.pose.position.x = 2.80;
  goal.target_pose.pose.position.y = 3.45;
  goal.target_pose.pose.orientation.x = 0.0;
  goal.target_pose.pose.orientation.y = 0.0;
  goal.target_pose.pose.orientation.z = -0.643;
  goal.target_pose.pose.orientation.w = 0.766;

  ROS_INFO("Sending 6 goal");
  ac.sendGoal(goal);
  
  ac.waitForResult();

  if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Second mission complete!");
  else
    ROS_INFO("Second mission failed ...");
  return 0;
}