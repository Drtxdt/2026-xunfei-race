#include "my_planner.h" 
#include <pluginlib/class_list_macros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

PLUGINLIB_EXPORT_CLASS(my_planner::MyPlanner, nav_core::BaseLocalPlanner)

double Kp = 3.0; // 5.5,6.0
double Ki = 0.0;
double Kd = 1.3; // 1.1,1.3

double angular_error = 0;
double last_error = 0;
double error_sum = 0;
double error_diff = 0;
double output = 0;

namespace my_planner
{
    MyPlanner::MyPlanner()
    {
        setlocale(LC_ALL, "");
    }
    MyPlanner::~MyPlanner()
    {
    }

    tf::TransformListener *tf_listener_;
    costmap_2d::Costmap2DROS *costmap_ros_;
    /**
     * @brief 规划器初始化函数
     * @details 该函数是ROS导航框架中BaseLocalPlanner接口的初始化方法，
     *          在规划器被加载时调用，用于初始化TF监听器和代价地图指针
     * 
     * @param name 规划器名称，用于ROS节点和参数命名
     * @param tf TF2坐标变换缓冲器指针，用于获取坐标系变换
     * @param costmap_ros 代价地图ROS封装类指针，提供地图数据和坐标变换服务
     */
    void MyPlanner::initialize(std::string name, tf2_ros::Buffer *tf, costmap_2d::Costmap2DROS *costmap_ros)
    {
        // 输出初始化提示信息
        ROS_WARN("该我上场表演了");
        
        // 初始化TF监听器，用于监听坐标系变换
        tf_listener_ = new tf::TransformListener();
        
        // 保存代价地图指针，后续路径规划时使用
        costmap_ros_ = costmap_ros;
    }

    std::vector<geometry_msgs::PoseStamped> global_plan_;
    int target_index_;    // 当前目标路径点索引
    bool pose_adjusting_; // 是否需要调整位姿
    bool goal_reached_;   // 是否到达目标点
/**
     * @brief 设置全局路径规划结果
     * @details 该函数用于接收上层全局规划器生成的路径，
     *          初始化路径跟踪所需的状态变量，为后续的局部路径跟踪做准备
     * 
     * @param plan 全局路径点序列，包含一系列带时间戳和坐标系信息的位姿
     * @return 返回true表示路径设置成功
     */
    bool MyPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped> &plan)
    {
        // 重置目标路径点索引，从头开始跟踪
        target_index_ = 0;
        
        // 保存全局路径到成员变量
        global_plan_ = plan;
        
        // 重置位姿调整标志，初始状态不需要调整
        pose_adjusting_ = false;
        
        // 重置目标到达标志
        goal_reached_ = false;
        
        return true;
    }

/**
     * @brief 计算速度控制指令
     * @details 该函数是局部路径跟踪的核心方法，实现了基于PID控制器的路径跟踪算法。
     *          主要功能包括：获取代价地图数据、检测路径上的障碍物、选择目标路径点、
     *          使用PID控制器计算角速度、处理最终姿态调整等。
     * 
     * @param cmd_vel 输出的速度指令，包含线速度和角速度
     * @return 返回true表示成功计算速度指令，返回false表示路径被障碍物阻挡
     */
    bool MyPlanner::computeVelocityCommands(geometry_msgs::Twist &cmd_vel)
    {
        // ========== 1. 获取代价地图数据 ==========
        costmap_2d::Costmap2D *costmap = costmap_ros_->getCostmap();  // 获取代价地图指针
        unsigned char *map_data = costmap->getCharMap();               // 获取地图数据数组
        unsigned int size_x = costmap->getSizeInCellsX();              // 获取地图宽度（单元格数）
        unsigned int size_y = costmap->getSizeInCellsY();              // 获取地图高度（单元格数）

        // 使用cv绘制代价地图
        /*cv::Mat map_image(size_y,size_x,CV_8UC3,cv::Scalar(128,128,128));
        for (unsigned int y =0;y<size_y;y++)
        {
            for(unsigned int x=0;x<size_x;x++)
            {
                int map_index = y*size_x + x;
                unsigned char cost=map_data[map_index];
                cv::Vec3b& pixel = map_image.at<cv::Vec3b>(map_index);

                if(cost==0)
                    pixel = cv::Vec3b(128,128,128);
                else if(cost==254)
                    pixel = cv::Vec3b(0,0,0);
                else if(cost==253)
                    pixel = cv::Vec3b(255,255,0);
                else
                {
                    unsigned char blue=255-cost;
                    unsigned char red = cost;
                    pixel = cv::Vec3b(blue,0,red);
                }
            }
        }*/

        // ========== 2. 障碍物检测 ==========
        // 遍历全局路径点，检测前方路径是否被障碍物阻挡
        for (int i = 0; i < global_plan_.size(); i++)
        {
            // 将路径点从map坐标系转换到odom坐标系
            geometry_msgs::PoseStamped pose_odom;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("odom", global_plan_[i], pose_odom);
            double odom_x = pose_odom.pose.position.x;
            double odom_y = pose_odom.pose.position.y;

            // 将odom坐标转换为代价地图的像素坐标
            double origin_x = costmap->getOriginX();
            double origin_y = costmap->getOriginY();
            double local_x = odom_x - origin_x;
            double local_y = odom_y - origin_y;
            int x = local_x / costmap->getResolution();
            int y = local_y / costmap->getResolution();

            // 检测前方10个路径点是否在禁行区域或障碍物中
            // cost >= 253 表示障碍物或致命障碍区域
            if (i >= target_index_ && i < target_index_ + 10)
            {
                int map_index = y * size_x + x;
                unsigned char cost = map_data[map_index];
                if (cost >= 253)
                {
                    ROS_WARN("前方路径被障碍物阻挡，索引=%d, cost=%d", i, cost);
                    return false;  // 返回false表示路径被阻挡，需要重新规划
                }
            }
        }

        // map_image.at<cv::Vec3b>(size_y/2,size_x/2)=cv::Vec3b(0,255,0);

        // 翻转地图
        /*cv::Mat flipped_image(size_x,size_y,CV_8UC3,cv::Scalar(128,128,128));
        for(unsigned int y=0;y<size_y;++y)
        {
            for(unsigned int x=0;x<size_x;++x)
            {
                cv::Vec3b& pixel = map_image.at<cv::Vec3b>(y,x);
                flipped_image.at<cv::Vec3b>((size_x-1-x),(size_y-1-y))=pixel;
            }
        }
        map_image = flipped_image;*/

        // 显示代价地图
        /*cv::namedWindow("Map");
        cv::resize(map_image,map_image,cv::Size(size_y*5,size_x*5),0,0,cv::INTER_NEAREST);
        cv::resizeWindow("Map",size_y*5,size_x*5);
        cv::imshow("Map",map_image);*/

        // ========== 3. 最终姿态调整 ==========
        // 当小车接近终点时，进行最终姿态调整
        int final_index = global_plan_.size() - 1;
        geometry_msgs::PoseStamped pose_final;
        global_plan_[final_index].header.stamp = ros::Time(0);
        // 将终点位姿转换到base_link坐标系
        tf_listener_->transformPose("base_link", global_plan_[final_index], pose_final);
        double dx = pose_final.pose.position.x;
        double dy = pose_final.pose.position.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        // 判断是否进入姿态调整阶段（距离终点小于0.1米）
        if (pose_adjusting_ == false)
        {
            if (dist < 0.1)
            {
                pose_adjusting_ = true;
                ROS_WARN("进入姿态调整阶段，距离终点=%.2f", dist);
            }
        }

        // 执行姿态调整
        if (pose_adjusting_ == true)
        {
            double final_yaw = tf::getYaw(pose_final.pose.orientation);
            ROS_WARN("调整最终姿态，final_yaw=%.2f", final_yaw);
            
            // 线速度与距离成正比，角速度与偏航角成正比
            cmd_vel.linear.x = pose_final.pose.position.x * 1.5;
            cmd_vel.angular.z = final_yaw * 0.5;
            
            // 到达终点条件：偏航角误差<0.2rad且距离<0.05m
            if (abs(final_yaw) < 0.2 && dist < 0.05)
            {
                goal_reached_ = true;
                ROS_WARN("到达终点");
                cmd_vel.linear.x = 0;
                cmd_vel.angular.z = 0;
            }
            return true;
        }

        // ========== 4. 选择目标路径点 ==========
        // 从当前目标索引开始，选择距离小车超过0.2米的路径点作为临时目标
        geometry_msgs::PoseStamped target_pose;
        for (int i = target_index_; i < global_plan_.size(); i++)
        {
            // 将路径点转换到base_link坐标系
            geometry_msgs::PoseStamped pose_base;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("base_link", global_plan_[i], pose_base);
            double dx = pose_base.pose.position.x;
            double dy = pose_base.pose.position.y;
            double dist = std::sqrt(dx * dx + dy * dy);

            // 选择第一个距离超过0.2米的路径点作为目标
            if (dist > 0.2)
            {
                target_pose = pose_base;
                target_index_ = i;
                ROS_WARN("选择第%d个路径点作为临时目标，距离=%.2f", target_index_, dist);
                break;
            }

            // 如果到达最后一个路径点
            if (i == global_plan_.size() - 1)
            {
                target_pose = pose_base;
            }
        }

        // ========== 5. PID控制器计算角速度 ==========
        // 计算角度误差（目标点相对于小车的方位角）
        angular_error = std::atan2(target_pose.pose.position.y, target_pose.pose.position.x);
        
        // 线速度：根据目标距离动态调节，最大不超过0.5m/s
        cmd_vel.linear.x = std::min(0.5, dist * 2.0);
        
        // PID控制计算
        error_sum += angular_error;      // 积分项
        error_diff = angular_error - last_error;  // 微分项
        output = Kp * angular_error + Ki * error_sum + Kd * error_diff;

        // 限制角速度范围 [-2.0, 2.0] rad/s
        if (output > 2.0)
            output = 2.0;
        else if (output < -2.0)
            output = -2.0;
        
        ROS_WARN("PID输出=%.2f", output);
        cmd_vel.angular.z = output;
        
        // 更新上一次误差，用于下一次微分项计算
        last_error = angular_error;
        
        return true;
    }

    bool MyPlanner::isGoalReached()
    {
        return goal_reached_;
    }
} // namespace my_planner