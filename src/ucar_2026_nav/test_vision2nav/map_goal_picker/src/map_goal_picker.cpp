#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <fstream>
#include <string>

class MapGoalPicker
{
public:
    MapGoalPicker(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    {
        // 读取 pgm 路径参数
        pnh.param<std::string>("pgm_path", pgm_path_, "");
        if (!pgm_path_.empty())
        {
            ROS_INFO("[map_goal_picker] PGM map path: %s", pgm_path_.c_str());
        }
        else
        {
            ROS_WARN("[map_goal_picker] pgm_path not set.");
        }

        // 订阅 rviz 发布的 2D Nav Goal
        goal_sub_ = nh.subscribe("/move_base_simple/goal", 1, &MapGoalPicker::goalCallback, this);

        ROS_INFO("[map_goal_picker] Waiting for 2D Nav Goal from rviz on /move_base_simple/goal ...");
    }

private:
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        double x = msg->pose.position.x;
        double y = msg->pose.position.y;

        // 四元数转 yaw
        tf::Quaternion q(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // 终端打印
        ROS_INFO("[map_goal_picker] ==========================================");
        ROS_INFO("[map_goal_picker] Received goal from rviz:");
        ROS_INFO("[map_goal_picker]   x   = %.4f", x);
        ROS_INFO("[map_goal_picker]   y   = %.4f", y);
        ROS_INFO("[map_goal_picker]   yaw = %.4f rad (%.2f deg)", yaw, yaw * 180.0 / M_PI);
        ROS_INFO("[map_goal_picker] ==========================================");

        // 写入 ROS 参数服务器，供包2读取
        // ros::param::set("/map_goal_picker/goal_x", x);
        // ros::param::set("/map_goal_picker/goal_y", y);
        // ros::param::set("/map_goal_picker/goal_yaw", yaw);
        // ROS_INFO("[map_goal_picker] Goal saved to parameter server: /map_goal_picker/goal_*");
    }

    ros::Subscriber goal_sub_;
    std::string pgm_path_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "map_goal_picker");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    MapGoalPicker picker(nh, pnh);

    ros::spin();

    return 0;
}
