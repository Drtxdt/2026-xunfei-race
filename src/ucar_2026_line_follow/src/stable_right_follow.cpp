#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Twist.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class StableRightFollowNode
{
public:
    StableRightFollowNode()
    {
        ros::NodeHandle pnh("~");

        //------------------------------------------
        // 原有参数（完全保持不变）
        //------------------------------------------
        pnh.param("target_right_x", target_right_x_, 155);
        pnh.param("base_speed", base_speed_, 0.32);
        pnh.param("curve_speed", curve_speed_, 0.28);
        pnh.param("search_speed", search_speed_, 0.10);
        pnh.param("kp", kp_, 0.0052);
        pnh.param("kd", kd_, 0.0018);
        pnh.param("startup_time", startup_time_, 2.8);
        pnh.param("forward_time", forward_time_, 0.9);
        pnh.param("cross_area_threshold", cross_area_threshold_, 48000);
        pnh.param("show_debug", show_debug_, true);

        //------------------------------------------
        // 新增参数（弯道偏移）
        //------------------------------------------
        pnh.param("curve_threshold", curve_threshold_, 38.0);   // 位置误差超过此值视为弯道
        pnh.param("curve_offset", curve_offset_, 15.0);         // 弯道时目标右边界左移像素

        //------------------------------------------
        // 新增参数（姿态修正与最终前进）
        //------------------------------------------
        pnh.param("align_angular_speed", align_angular_speed_, 0.18);  // 修正旋转速度
        pnh.param("align_angle_threshold", align_angle_threshold_, 2.0); // 角度误差阈值（度）
        pnh.param("final_forward_speed", final_forward_speed_, 0.20);    // 最终前进速度
        pnh.param("final_forward_distance", final_forward_distance_, 0.70); // 最终前进距离（米）

        //------------------------------------------
        // ROS 接口
        //------------------------------------------
        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        image_sub_ = nh_.subscribe("/usb_cam/image_raw", 1,
                                   &StableRightFollowNode::imageCallback, this);

        //------------------------------------------
        // 内部状态初始化
        //------------------------------------------
        last_error_ = 0.0;
        filtered_error_ = 0.0;
        stage_ = 0;
        start_time_ = ros::Time::now();
        last_right_x_ = -1;
        last_right_angle_ = 0.0;
        stage4_enter_time_ = ros::Time::now();
        stage5_enter_time_ = ros::Time::now();

        ROS_INFO("=================================");
        ROS_INFO(" Stable Right Follow (enhanced)  ");
        ROS_INFO("=================================");
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber image_sub_;

    // 原有参数
    int target_right_x_;
    double base_speed_, curve_speed_, search_speed_;
    double kp_, kd_;
    double startup_time_, forward_time_;
    int cross_area_threshold_;
    bool show_debug_;

    // 新增参数
    double curve_threshold_, curve_offset_;
    double align_angular_speed_, align_angle_threshold_;
    double final_forward_speed_, final_forward_distance_;

    // PID 状态
    double last_error_, filtered_error_;

    // 状态机
    int stage_;                     // 0:启动,1:搜索,2:巡线,3:停止,4:姿态修正,5:前进,6:停车
    ros::Time start_time_;
    ros::Time stage4_enter_time_;   // 进入姿态修正的时刻
    ros::Time stage5_enter_time_;   // 进入最终前进的时刻

    int last_right_x_;
    double last_right_angle_;       // 右边线角度（度）

    //------------------------------------------
    // 图像回调
    //------------------------------------------
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("%s", e.what());
            return;
        }

        int h = frame.rows;
        int w = frame.cols;
        cv::Mat roi = frame(cv::Range(int(h*0.60), h), cv::Range(0, w));
        cv::Mat mask = extractWhiteMask(roi);

        geometry_msgs::Twist twist;

        //------------------------------------------
        // STAGE 0：启动直行（原封不动）
        //------------------------------------------
        if (stage_ == 0) {
            double elapsed = (ros::Time::now() - start_time_).toSec();
            if (elapsed < startup_time_) {
                twist.linear.x = 0.45;
                twist.angular.z = 0.0;
                cmd_pub_.publish(twist);
                return;
            }
            ROS_INFO("ENTER SEARCH MODE");
            stage_ = 1;
        }

        // 右边线检测（返回位置和角度）
        int right_x = -1;
        double right_angle = 0.0;
        bool line_found = findRightLine(mask, right_x, right_angle);

        //------------------------------------------
        // STAGE 1：搜索右边线（原封不动）
        //------------------------------------------
        if (stage_ == 1) {
            if (!line_found) {
                twist.linear.x = 0.10;
                twist.angular.z = -0.26;
                cmd_pub_.publish(twist);
                return;
            }
            ROS_INFO("RIGHT LINE FOUND");
            last_right_x_ = right_x;
            last_right_angle_ = right_angle;
            stage_ = 2;
        }

        //------------------------------------------
        // STAGE 2：巡线（原逻辑基础上增加弯道偏移）
        //------------------------------------------
        if (stage_ == 2) {
            // 停止线检测（保持原有面积阈值法，同时提取轮廓角度备用）
            bool stop_detected = false;
            int stop_area = cv::countNonZero(mask);
            if (stop_area > cross_area_threshold_) {
                stop_detected = true;
            }

            if (stop_detected) {
                ROS_INFO("STOP LINE DETECTED -> STOP & ALIGN");
                stopCar();
                stage_ = 3;   // 进入停车状态，下一步进行姿态修正
                // 保存停止线区域用于后续角度计算（在 stage 4 中使用）
                stop_line_mask_ = mask.clone();
                stage4_enter_time_ = ros::Time::now();  // 实际修正阶段会用到，这里先赋值
                return;
            }

            // 丢线处理（保留原逻辑）
            if (!line_found) {
                if (last_right_x_ >= 0) {
                    twist.linear.x = 0.14;
                    twist.angular.z = -0.22;
                } else {
                    twist.linear.x = 0.10;
                    twist.angular.z = -0.24;
                }
                cmd_pub_.publish(twist);
                return;
            }

            last_right_x_ = right_x;
            last_right_angle_ = right_angle;

            //---------------- 弯道偏移：动态调整目标位置 ----------------
            double current_target = target_right_x_;
            double raw_error = current_target - right_x;
            if (std::fabs(raw_error) > curve_threshold_) {
                current_target -= curve_offset_;   // 弯道时让目标更靠左，增大与边线距离
            }

            //---------------- PID 计算（位置，原算法不变）----------------
            double error = current_target - right_x;
            double alpha = 0.22;
            filtered_error_ = (1.0 - alpha) * filtered_error_ + alpha * error;
            double d_error = filtered_error_ - last_error_;
            last_error_ = filtered_error_;

            double angular = kp_ * filtered_error_ + kd_ * d_error;

            //---------------- 速度选择（原逻辑保留）----------------
            double linear_speed;
            if (std::fabs(filtered_error_) > 38) {
                linear_speed = curve_speed_;
                angular *= 1.18;
            } else {
                linear_speed = base_speed_;
            }

            //---------------- 限幅（原逻辑保留）----------------
            if (angular > 0.55) angular = 0.55;
            if (angular < -0.55) angular = -0.55;

            twist.linear.x = linear_speed;
            twist.angular.z = angular;
            cmd_pub_.publish(twist);

            //---------------- 调试窗口（增强显示）----------------
            if (show_debug_) {
                cv::Mat debug;
                cv::cvtColor(mask, debug, cv::COLOR_GRAY2BGR);
                cv::line(debug, cv::Point(current_target, 0),
                         cv::Point(current_target, mask.rows-1),
                         cv::Scalar(255,0,0), 2);
                cv::circle(debug, cv::Point(right_x, mask.rows/2), 5,
                           cv::Scalar(0,0,255), -1);
                char buf[150];
                sprintf(buf, "ERR:%.1f TGT:%d", filtered_error_, (int)current_target);
                cv::putText(debug, buf, cv::Point(20,40),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
                cv::imshow("right_follow", debug);
                cv::waitKey(1);
            }
            return;
        }

        //------------------------------------------
        // STAGE 3：停止线检测后停车（等待姿态修正）
        //------------------------------------------
        if (stage_ == 3) {
            // 已停车，短暂延迟确保图像稳定，然后进入修正
            if ((ros::Time::now() - stage4_enter_time_).toSec() > 0.3) {
                stage_ = 4;   // 进入姿态修正阶段
                stage4_enter_time_ = ros::Time::now();
            }
            // 保持停车
            stopCar();
            return;
        }

        //------------------------------------------
        // STAGE 4：姿态修正（使车身与停止线垂直）
        //------------------------------------------
        if (stage_ == 4) {
            // 优先使用停止线角度，若不可见则使用右边线角度
            double target_angle = 0.0;
            bool have_stop_angle = getStopLineAngle(stop_line_mask_, target_angle);
            if (!have_stop_angle) {
                // 退而求其次，使用右边线角度（期望车身与右边线平行，这里近似垂直需要调整策略）
                // 对于横向停止线，期望角度为0；对于右边线，车身应平行，我们无法直接获得垂直误差。
                // 简单起见，若停止线角度不可得，尝试使用右边线角度但实际意义不同，此处保持安全：直接认为修正完成。
                target_angle = 0.0;
                ROS_WARN_THROTTLE(1.0, "Cannot get stop line angle, skip align");
                stage_ = 5;   // 跳过修正，直接前进
                stage5_enter_time_ = ros::Time::now();
                return;
            }

            // 停止线角度：水平线为0，我们希望车身与线垂直，即要求该角度为0（图像中停止线是水平的）
            if (std::fabs(target_angle) < align_angle_threshold_) {
                ROS_INFO("ALIGN DONE -> FORWARD %.2fm", final_forward_distance_);
                stopCar();
                stage_ = 5;
                stage5_enter_time_ = ros::Time::now();
                return;
            }

            // 固定角速度修正：若停止线向右倾斜（正角度），车头偏左，需左转（负角速度）
            double ang_cmd = (target_angle > 0) ? -align_angular_speed_ : align_angular_speed_;
            twist.linear.x = 0.0;
            twist.angular.z = ang_cmd;
            cmd_pub_.publish(twist);

            if (show_debug_) {
                cv::Mat debug;
                cv::cvtColor(stop_line_mask_, debug, cv::COLOR_GRAY2BGR);
                char buf[100];
                sprintf(buf, "Align Angle: %.1f", target_angle);
                cv::putText(debug, buf, cv::Point(20,40),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
                cv::imshow("right_follow", debug);
                cv::waitKey(1);
            }
            return;
        }

        //------------------------------------------
        // STAGE 5：最终前进（精确距离）
        //------------------------------------------
        if (stage_ == 5) {
            double elapsed = (ros::Time::now() - stage5_enter_time_).toSec();
            // 前进时间 = 距离 / 速度
            double need_time = final_forward_distance_ / final_forward_speed_;
            if (elapsed < need_time) {
                twist.linear.x = final_forward_speed_;
                twist.angular.z = 0.0;
                cmd_pub_.publish(twist);
                return;
            }
            stage_ = 6;
        }

        //------------------------------------------
        // STAGE 6：最终停车（原最终停车逻辑）
        //------------------------------------------
        if (stage_ == 6) {
            stopCar();
            ROS_INFO_THROTTLE(1.0, "FINAL STOP");
            return;
        }
    }

    //------------------------------------------
    // 白线提取（原函数未修改）
    //------------------------------------------
    cv::Mat extractWhiteMask(const cv::Mat& roi)
    {
        cv::Mat blur, hsv, mask;
        cv::GaussianBlur(roi, blur, cv::Size(5,5), 0);
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0,0,200), cv::Scalar(180,45,255), mask);

        cv::Mat kernel = cv::Mat::ones(5,5,CV_8U);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(mask, mask, 5);
        cv::GaussianBlur(mask, mask, cv::Size(5,5), 0);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat clean_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        for (auto& cnt : contours) {
            if (cv::contourArea(cnt) > 260)
                cv::drawContours(clean_mask, std::vector<std::vector<cv::Point>>{cnt},
                                 -1, cv::Scalar(255), -1);
        }
        return clean_mask;
    }

    //------------------------------------------
    // 右边线检测（增强：返回角度）
    //------------------------------------------
    bool findRightLine(const cv::Mat& mask, int& x_out, double& angle_out)
    {
        int h = mask.rows;
        std::vector<int> rows;
        rows.push_back(int(h * 0.50));
        rows.push_back(int(h * 0.60));
        rows.push_back(int(h * 0.70));

        std::vector<cv::Point2f> points;
        for (auto y : rows) {
            const uchar* ptr = mask.ptr<uchar>(y);
            for (int x = mask.cols - 1; x >= 0; x--) {
                if (ptr[x] > 0) {
                    points.push_back(cv::Point2f(x, y));
                    break;
                }
            }
        }

        if (points.size() < 2) {
            x_out = -1;
            angle_out = 0.0;
            return false;
        }

        // 计算平均 x 坐标
        double sum = 0.0;
        for (auto p : points) sum += p.x;
        x_out = static_cast<int>(sum / points.size());

        // 拟合直线求角度
        cv::Vec4f line;
        cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);
        double vx = line[0], vy = line[1];
        double angle_rad = atan2(vy, vx);          // 与 x 轴夹角
        angle_out = angle_rad * 180.0 / CV_PI;     // 角度制，0 度表示水平线
        return true;
    }

    //------------------------------------------
    // 停止线角度检测（新增）
    //------------------------------------------
    bool getStopLineAngle(const cv::Mat& mask, double& angle)
    {
        // 在 mask 中寻找最大轮廓，且宽度 > 150，高度 < 40
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto& cnt : contours) {
            cv::Rect rect = cv::boundingRect(cnt);
            if (rect.width > 150 && rect.height < 40 && rect.height > 5) {
                // 拟合该轮廓的最小外接矩形以获取角度
                cv::RotatedRect rrect = cv::minAreaRect(cnt);
                float ang = rrect.angle;  // 范围 [-90,0)，水平时为 -90 或接近0？
                // 需要归一化到与水平线的夹角
                if (rrect.size.width < rrect.size.height)
                    ang += 90.0;
                // 将角度转换到 [-45,45] 附近，0 表示水平
                ang = ang > 45 ? ang - 90 : (ang < -45 ? ang + 90 : ang);
                angle = ang;
                return true;
            }
        }
        return false;
    }

    //------------------------------------------
    // 停车（原函数未修改）
    //------------------------------------------
    void stopCar()
    {
        geometry_msgs::Twist twist;
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
        cmd_pub_.publish(twist);
    }

    // 用于保存停止线 mask 的成员（新增）
    cv::Mat stop_line_mask_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stable_right_follow_cpp");
    StableRightFollowNode node;
    ros::spin();
    return 0;
}