#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Twist.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

class StableRightFollowNode
{
public:
    StableRightFollowNode()
    {
        ros::NodeHandle pnh("~");

        loadParams(pnh);

        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        image_sub_ = nh_.subscribe(
            "/usb_cam/image_raw",
            1,
            &StableRightFollowNode::imageCallback,
            this);

        last_pos_error_ = 0.0;
        filtered_pos_error_ = 0.0;
        last_right_x_ = -1;
        stage_ = STARTUP;
        start_time_ = ros::Time::now();

        predicted_right_x_ = 160;
        lost_line_count_ = 0;
        last_angular_ = 0.0;
        last_line_time_ = ros::Time::now();

        start_moving_time_ = ros::Time::now();

        ROS_INFO("=================================");
        ROS_INFO(" Stable Right Follow (Robust Version) ");
        ROS_INFO("=================================");
    }

private:
    enum Stage
    {
        STARTUP = 0,
        SEARCH_RIGHT_LINE = 1,
        FOLLOW_RIGHT_LINE = 2,
        STOP_LINE_FOUND = 3,
        ALIGN_WITH_RIGHT_LINE = 4,
        GO_FORWARD = 5,
        FINAL_STOP = 6
    };

    struct LineInfo
    {
        bool found = false;
        int x = -1;
        double angle_deg = 0.0;
        std::vector<cv::Point> points;
        cv::Vec4f fit_line;
    };

    struct StopLineInfo
    {
        bool found = false;
        cv::Rect rect;
        double angle_deg = 0.0;
        int center_x = 0;
    };

    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber image_sub_;

    // ========== 可调参数（可通过 launch 文件覆盖） ==========
    int target_right_x_;                // 目标右线位置（像素）
    double base_speed_;                 // 直道速度 (m/s)
    double curve_speed_;                // 弯道速度 (m/s)
    double search_speed_;               // 搜索速度 (m/s)
    double lost_line_speed_;            // 丢线速度 (m/s)
    double startup_speed_;              // 启动速度 (m/s)

    double kp_pos_;                     // 位置比例增益
    double kd_pos_;                     // 位置微分增益
    double kp_angle_;                   // 角度比例增益

    double curve_threshold_;            // 判断弯道的误差阈值
    double curve_offset_;               // 弯道偏移量
    double curve_gain_;                 // 弯道角速度增益

    double max_angular_;                // 最大角速度 (rad/s)
    double error_filter_alpha_;         // 低通滤波系数

    double startup_time_;               // 启动阶段持续时间 (s)

    int cross_area_threshold_;          // 大白色区域后备阈值
    int stop_line_min_width_;           // 停车线最小宽度 (像素)
    int stop_line_max_height_;          // 停车线最大高度 (像素)
    int stop_line_min_area_;            // 停车线最小面积 (像素²)

    double align_speed_;                // 对齐时旋转速度上限
    double align_angle_threshold_;      // 对齐角度阈值 (度)
    double align_stop_time_;            // 停车后等待时间 (s)
    double desired_angle_deg_;          // 期望角度 (0° 表示水平)

    double final_speed_;                // 直走速度 (m/s)
    double final_distance_;             // 直走距离 (m)

    bool show_debug_;                   // 是否显示调试窗口

    // ========== 运行时变量 ==========
    double last_pos_error_;
    double filtered_pos_error_;
    int last_right_x_;

    Stage stage_;
    ros::Time start_time_;
    ros::Time stage_start_time_;
    ros::Time forward_start_time_;

    // 预测与容错变量
    int predicted_right_x_;
    int lost_line_count_;
    double last_angular_;
    ros::Time last_line_time_;
    double last_vx_ = 0.0;
    double last_vy_ = 1.0;

    // 时间控制
    ros::Time start_moving_time_;
    double stop_line_ignore_time_ = 10.0;  // 前10秒忽略停车线

    // ========== 参数加载 ==========
    void loadParams(ros::NodeHandle& pnh)
    {
        pnh.param("target_right_x", target_right_x_, 145);

        pnh.param("base_speed", base_speed_, 0.22);        // 降低速度，提高稳定性
        pnh.param("curve_speed", curve_speed_, 0.18);      // 弯道更慢
        pnh.param("search_speed", search_speed_, 0.12);
        pnh.param("lost_line_speed", lost_line_speed_, 0.14);
        pnh.param("startup_speed", startup_speed_, 0.45);

        pnh.param("kp_pos", kp_pos_, 0.0035);              // 降低比例，减少过冲
        pnh.param("kd_pos", kd_pos_, 0.0025);              // 增大微分，抑制震荡
        pnh.param("kp_angle", kp_angle_, 0.25);            // 降低角度增益

        pnh.param("curve_threshold", curve_threshold_, 35.0);
        pnh.param("curve_offset", curve_offset_, 15.0);
        pnh.param("curve_gain", curve_gain_, 0.9);         // 弯道增益降低

        pnh.param("max_angular", max_angular_, 0.40);      // 限制最大转向
        pnh.param("error_filter_alpha", error_filter_alpha_, 0.30); // 更平滑

        pnh.param("startup_time", startup_time_, 2.8);

        pnh.param("cross_area_threshold", cross_area_threshold_, 48000);
        pnh.param("stop_line_min_width", stop_line_min_width_, 120);
        pnh.param("stop_line_max_height", stop_line_max_height_, 40);
        pnh.param("stop_line_min_area", stop_line_min_area_, 800);

        pnh.param("align_speed", align_speed_, 0.18);
        pnh.param("align_angle_threshold", align_angle_threshold_, 1.0);
        pnh.param("align_stop_time", align_stop_time_, 1.5); // 停车1.5秒
        pnh.param("desired_angle_deg", desired_angle_deg_, 0.0);

        pnh.param("final_speed", final_speed_, 0.20);
        pnh.param("final_distance", final_distance_, 0.60);  // 直走60cm

        pnh.param("show_debug", show_debug_, true);
    }

    // ========== 图像回调 ==========
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv::Mat frame;

        try
        {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("%s", e.what());
            return;
        }

        if(frame.empty())
        {
            stopCar();
            return;
        }

        const int h = frame.rows;
        const int w = frame.cols;

        // ROI 扩大至 45% 高度
        cv::Mat roi = frame(
            cv::Range(static_cast<int>(h * 0.45), h),
            cv::Range(0, w));

        cv::Mat mask = extractWhiteMask(roi);
        LineInfo right_line = findRightLine(mask);
        StopLineInfo stop_line = findStopLine(mask);

        geometry_msgs::Twist twist;

        switch(stage_)
        {
        case STARTUP:
            handleStartup(twist);
            break;

        case SEARCH_RIGHT_LINE:
            handleSearch(twist, right_line);
            break;

        case FOLLOW_RIGHT_LINE:
            handleFollow(twist, right_line, stop_line);
            break;

        case STOP_LINE_FOUND:
            handleStopLineFound(twist);
            break;

        case ALIGN_WITH_RIGHT_LINE:
            handleAlign(twist, stop_line);
            break;

        case GO_FORWARD:
            handleGoForward(twist);
            break;

        case FINAL_STOP:
        default:
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            break;
        }

        // 角速度低通滤波
        twist.angular.z = 0.7 * last_angular_ + 0.3 * twist.angular.z;
        last_angular_ = twist.angular.z;

        cmd_pub_.publish(twist);

        if(show_debug_)
        {
            showDebug(mask, right_line, stop_line, twist);
        }
    }

    // ========== 状态机处理函数 ==========
    void handleStartup(geometry_msgs::Twist& twist)
    {
        const double elapsed = (ros::Time::now() - start_time_).toSec();

        if(elapsed < startup_time_)
        {
            twist.linear.x = startup_speed_;
            twist.angular.z = 0.0;
            return;
        }

        start_moving_time_ = ros::Time::now();  // 记录开始移动时间
        enterStage(SEARCH_RIGHT_LINE, "ENTER SEARCH MODE");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleSearch(
        geometry_msgs::Twist& twist,
        const LineInfo& right_line)
    {
        if(!right_line.found)
        {
            twist.linear.x = search_speed_;
            twist.angular.z = -0.26;
            return;
        }

        last_right_x_ = right_line.x;
        resetPid();
        lost_line_count_ = 0;
        predicted_right_x_ = right_line.x;
        enterStage(FOLLOW_RIGHT_LINE, "RIGHT LINE FOUND");

        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleFollow(
        geometry_msgs::Twist& twist,
        const LineInfo& right_line,
        const StopLineInfo& stop_line)
    {
        // 前10秒忽略停车线检测
        double elapsed_since_move = (ros::Time::now() - start_moving_time_).toSec();
        bool ignore_stop_line = (elapsed_since_move < stop_line_ignore_time_);

        if(stop_line.found && !ignore_stop_line)
        {
            resetPid();
            enterStage(STOP_LINE_FOUND, "STOP LINE DETECTED (TIME>=10s)");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            return;
        }

        // 长时间丢线（超过30帧）强制搜索
        if(lost_line_count_ > 30)
        {
            enterStage(SEARCH_RIGHT_LINE, "LINE LOST FOR TOO LONG");
            twist.linear.x = search_speed_;
            twist.angular.z = -0.2;
            return;
        }

        // 短时间丢线使用预测
        int current_x = right_line.found ? right_line.x : predicted_right_x_;
        double current_angle = right_line.found ? right_line.angle_deg : desired_angle_deg_;

        if(right_line.found)
        {
            last_right_x_ = right_line.x;
            predicted_right_x_ = 0.7 * predicted_right_x_ + 0.3 * right_line.x;
            lost_line_count_ = 0;
            last_line_time_ = ros::Time::now();
        }
        else
        {
            lost_line_count_++;
        }

        const bool in_curve =
            std::fabs(filtered_pos_error_) > curve_threshold_ ||
            std::fabs(current_angle - desired_angle_deg_) > align_angle_threshold_;

        const double target = target_right_x_ - (in_curve ? curve_offset_ : 0.0);

        double pos_error = target - current_x;
        // 死区：误差小于3像素不修正
        if (std::fabs(pos_error) < 3.0) pos_error = 0.0;

        // 低通滤波
        filtered_pos_error_ =
            (1.0 - error_filter_alpha_) * filtered_pos_error_ +
            error_filter_alpha_ * pos_error;

        const double d_pos_error = filtered_pos_error_ - last_pos_error_;
        last_pos_error_ = filtered_pos_error_;

        const double angle_error = current_angle - desired_angle_deg_;

        double angular =
            kp_pos_ * filtered_pos_error_ +
            kd_pos_ * d_pos_error +
            kp_angle_ * deg2rad(angle_error);

        double linear_speed = base_speed_;
        if (in_curve || std::fabs(filtered_pos_error_) > 40) {
            linear_speed = curve_speed_;
        }
        // 大误差进一步减速
        if (std::fabs(filtered_pos_error_) > 60) {
            linear_speed *= 0.8;
        }

        if(in_curve)
        {
            angular *= curve_gain_;
        }

        angular = clamp(angular, -max_angular_, max_angular_);

        twist.linear.x = linear_speed;
        twist.angular.z = angular;
    }

    void handleStopLineFound(geometry_msgs::Twist& twist)
    {
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();

        // 必须停住
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        // 停车1.5秒（align_stop_time_ 已设为1.5）
        if(elapsed >= align_stop_time_)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "ENTER ALIGN MODE");
        }

        if(elapsed > 3.0)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "STOPLINE TIMEOUT");
        }
    }

    void handleAlign(geometry_msgs::Twist& twist, const StopLineInfo& stop_line)
    {
        twist.linear.x = 0.0;   // 原地对齐

        if(!stop_line.found)
        {
            twist.angular.z = -0.12;
            if((ros::Time::now() - stage_start_time_).toSec() > 3.0)
            {
                enterStage(GO_FORWARD, "ALIGN LINE LOST, FORCE GO");
            }
            return;
        }

        double angle_error = stop_line.angle_deg - 0.0;
        double angular_cmd = clamp(angle_error * 0.035, -0.18, 0.18);

        // 死区，防止小角度震荡
        if(std::fabs(angle_error) < 0.2)
        {
            angular_cmd = 0.0;
        }

        twist.angular.z = angular_cmd;

        // 对齐确认：角度小于 0.5° 并保持 0.15 秒
        static ros::Time align_ok_time;
        if(std::fabs(angle_error) < 0.5)
        {
            if(align_ok_time.isZero()) align_ok_time = ros::Time::now();
            if((ros::Time::now() - align_ok_time).toSec() > 0.15)
            {
                enterStage(GO_FORWARD, "ALIGN OK");
                align_ok_time = ros::Time(0);
            }
            twist.angular.z = 0.0;  // 等待期间停止转动
        }
        else
        {
            align_ok_time = ros::Time(0); // 重置
        }

        // 超时保护
        if((ros::Time::now() - stage_start_time_).toSec() > 4.5)
        {
            enterStage(GO_FORWARD, "ALIGN TIMEOUT");
        }
    }

    void handleGoForward(geometry_msgs::Twist& twist)
    {
        const double speed = std::max(0.01, std::fabs(final_speed_));
        const double forward_time = final_distance_ / speed;
        const double elapsed = (ros::Time::now() - forward_start_time_).toSec();

        if(elapsed < forward_time)
        {
            twist.linear.x = final_speed_;
            twist.angular.z = 0.0;
            return;
        }

        enterStage(FINAL_STOP, "FINAL STOP");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    // ========== 图像处理 ==========
    cv::Mat extractWhiteMask(const cv::Mat& roi)
    {
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

        // 自适应阈值（处理光照不均）
        cv::Mat adaptive;
        cv::adaptiveThreshold(gray, adaptive, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY, 21, 10);

        // 形态学闭运算填充孔洞
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
        cv::morphologyEx(adaptive, adaptive, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(adaptive, adaptive, cv::MORPH_OPEN, kernel);

        // 保留大连通域（去除噪声）
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(adaptive, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat clean = cv::Mat::zeros(adaptive.size(), CV_8UC1);
        for(auto& cnt : contours) {
            if(cv::contourArea(cnt) > 200) {
                cv::drawContours(clean, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), -1);
            }
        }
        return clean;
    }

    LineInfo findRightLine(const cv::Mat& mask)
    {
        LineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;

        for(int y = static_cast<int>(h * 0.20);
            y < static_cast<int>(h * 0.92);
            y += 4)
        {
            const uchar* ptr = mask.ptr<uchar>(y);

            for(int x = w - 1; x >= 0; --x)
            {
                if(ptr[x] > 0)
                {
                    info.points.push_back(cv::Point(x, y));
                    break;
                }
            }
        }

        if(info.points.size() < 6)
        {
            info.found = false;
            info.x = predicted_right_x_;
            info.angle_deg = desired_angle_deg_;
            lost_line_count_++;
            // 外推：如果最近有方向，就用方向预测
            if (last_line_time_ != ros::Time(0) && (ros::Time::now() - last_line_time_).toSec() < 0.5) {
                info.x = predicted_right_x_ + (int)(last_vx_ * 2);
            }
            return info;
        }

        double x_sum = 0.0;
        int x_count = 0;

        for(const auto& p : info.points)
        {
            if(p.y > h * 0.45)
            {
                x_sum += p.x;
                ++x_count;
            }
        }

        if(x_count == 0)
        {
            for(const auto& p : info.points)
            {
                x_sum += p.x;
            }
            x_count = static_cast<int>(info.points.size());
        }

        cv::fitLine(
            info.points,
            info.fit_line,
            cv::DIST_L2,
            0.0,
            0.01,
            0.01);

        const double vx = info.fit_line[0];
        const double vy = info.fit_line[1];

        info.found = true;
        info.x = static_cast<int>(x_sum / x_count);
        info.angle_deg = rad2deg(std::atan2(vx, vy));

        // 更新预测
        predicted_right_x_ = 0.7 * predicted_right_x_ + 0.3 * info.x;
        lost_line_count_ = 0;
        last_line_time_ = ros::Time::now();
        last_vx_ = vx;
        last_vy_ = vy;

        return info;
    }

    StopLineInfo findStopLine(const cv::Mat& mask)
    {
        StopLineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;

        // 只关注底部区域（停车线通常在底部）
        cv::Mat bottom_roi = mask(
            cv::Range(static_cast<int>(h * 0.65), h),
            cv::Range(0, w));

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(
            bottom_roi.clone(),
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        double max_area = 0;
        int best_idx = -1;

        for(size_t i = 0; i < contours.size(); ++i)
        {
            const auto& cnt = contours[i];
            double area = cv::contourArea(cnt);
            if(area < stop_line_min_area_) continue;

            cv::Rect rect = cv::boundingRect(cnt);

            // 严格约束：位置在底部、宽高比大、高度小
            if(rect.y < h * 0.65) continue;          // 必须在底部区域
            if(rect.width < rect.height * 5) continue; // 宽高比 > 5
            if(rect.height > stop_line_max_height_) continue;
            if(rect.width < stop_line_min_width_) continue;

            if(cnt.size() >= 5)
            {
                cv::Vec4f line;
                cv::fitLine(cnt, line, cv::DIST_L2, 0, 0.01, 0.01);
                double angle = rad2deg(std::atan2(line[1], line[0]));
                // 必须接近水平（±15°内）
                if(std::fabs(angle) > 15.0) continue;
            }

            if(area > max_area)
            {
                max_area = area;
                best_idx = i;
            }
        }

        if(best_idx >= 0)
        {
            const auto& cnt = contours[best_idx];
            cv::Rect rect = cv::boundingRect(cnt);
            cv::Vec4f line;
            cv::fitLine(cnt, line, cv::DIST_L2, 0, 0.01, 0.01);

            info.found = true;
            info.rect = rect;
            info.angle_deg = rad2deg(std::atan2(line[1], line[0]));
            info.center_x = rect.x + rect.width / 2;
        }

        // 后备：大区域白色检测（仅当且阈值大于0）
        if(!info.found && cross_area_threshold_ > 0)
        {
            int total_white = cv::countNonZero(mask);
            if(total_white > cross_area_threshold_)
            {
                info.found = true;
                info.rect = cv::Rect(0, 0, w, h);
                info.angle_deg = 0.0;
                info.center_x = w / 2;
            }
        }

        return info;
    }

    // ========== 调试显示 ==========
    void showDebug(
        const cv::Mat& mask,
        const LineInfo& right_line,
        const StopLineInfo& stop_line,
        const geometry_msgs::Twist& twist)
    {
        cv::Mat debug;
        cv::cvtColor(mask, debug, cv::COLOR_GRAY2BGR);

        const bool in_curve =
            std::fabs(filtered_pos_error_) > curve_threshold_;
        const int active_target =
            target_right_x_ - (in_curve ? static_cast<int>(curve_offset_) : 0);

        cv::line(
            debug,
            cv::Point(target_right_x_, 0),
            cv::Point(target_right_x_, mask.rows),
            cv::Scalar(255, 0, 0),
            2);

        cv::line(
            debug,
            cv::Point(active_target, 0),
            cv::Point(active_target, mask.rows),
            cv::Scalar(0, 255, 255),
            2);

        if(right_line.found)
        {
            for(const auto& p : right_line.points)
            {
                cv::circle(debug, p, 2, cv::Scalar(0, 180, 255), -1);
            }

            drawFitLine(debug, right_line.fit_line, cv::Scalar(0, 0, 255));
            cv::circle(
                debug,
                cv::Point(right_line.x, mask.rows / 2),
                5,
                cv::Scalar(0, 0, 255),
                -1);
        }

        if(stop_line.found)
        {
            cv::rectangle(debug, stop_line.rect, cv::Scalar(0, 255, 0), 2);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "Ang:%.1f", stop_line.angle_deg);
            cv::putText(debug, buf, cv::Point(stop_line.rect.x, stop_line.rect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        double elapsed = (ros::Time::now() - start_moving_time_).toSec();
        drawText(debug, 20, 30, "stage: " + stageName(stage_));
        drawText(debug, 20, 60, format("target: %d", active_target));
        drawText(debug, 20, 90, format("err: %.2f", filtered_pos_error_));
        drawText(debug, 20, 120, format("time: %.1fs", elapsed));
        drawText(debug, 20, 150, format("cmd w: %.3f", twist.angular.z));
        drawText(debug, 20, 180, format("lost: %d", lost_line_count_));

        cv::imshow("right_follow", debug);
        cv::waitKey(1);
    }

    void drawFitLine(
        cv::Mat& image,
        const cv::Vec4f& line,
        const cv::Scalar& color)
    {
        const float vx = line[0];
        const float vy = line[1];
        const float x0 = line[2];
        const float y0 = line[3];

        if(std::fabs(vy) < 1e-5)
        {
            return;
        }

        const int y1 = 0;
        const int y2 = image.rows - 1;
        const int x1 = static_cast<int>(x0 + (y1 - y0) * vx / vy);
        const int x2 = static_cast<int>(x0 + (y2 - y0) * vx / vy);

        cv::line(
            image,
            cv::Point(clampInt(x1, 0, image.cols - 1), y1),
            cv::Point(clampInt(x2, 0, image.cols - 1), y2),
            color,
            2);
    }

    void drawText(
        cv::Mat& image,
        int x,
        int y,
        const std::string& text)
    {
        cv::putText(
            image,
            text,
            cv::Point(x, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 255, 0),
            2);
    }

    // ========== 辅助函数 ==========
    void enterStage(Stage stage, const char* message)
    {
        stage_ = stage;
        stage_start_time_ = ros::Time::now();
        // 进入直走阶段时，同时复位计时器
        if (stage == GO_FORWARD) {
            forward_start_time_ = ros::Time::now();
        }
        ROS_INFO("%s", message);
    }

    void resetPid()
    {
        last_pos_error_ = 0.0;
        filtered_pos_error_ = 0.0;
    }

    void stopCar()
    {
        geometry_msgs::Twist twist;
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
        cmd_pub_.publish(twist);
    }

    std::string stageName(Stage stage) const
    {
        switch(stage)
        {
        case STARTUP:
            return "STARTUP";
        case SEARCH_RIGHT_LINE:
            return "SEARCH";
        case FOLLOW_RIGHT_LINE:
            return "FOLLOW";
        case STOP_LINE_FOUND:
            return "STOP_LINE";
        case ALIGN_WITH_RIGHT_LINE:
            return "ALIGN";
        case GO_FORWARD:
            return "FORWARD";
        case FINAL_STOP:
            return "FINAL_STOP";
        default:
            return "UNKNOWN";
        }
    }

    std::string format(const char* fmt, double value)
    {
        char buf[80];
        std::snprintf(buf, sizeof(buf), fmt, value);
        return std::string(buf);
    }

    std::string format(const char* fmt, int value)
    {
        char buf[80];
        std::snprintf(buf, sizeof(buf), fmt, value);
        return std::string(buf);
    }

    double deg2rad(double deg) const
    {
        return deg * CV_PI / 180.0;
    }

    double rad2deg(double rad) const
    {
        return rad * 180.0 / CV_PI;
    }

    double clamp(double value, double low, double high) const
    {
        return std::max(low, std::min(value, high));
    }

    int clampInt(int value, int low, int high) const
    {
        return std::max(low, std::min(value, high));
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stable_right_follow_cpp");

    StableRightFollowNode node;
    ros::spin();

    return 0;
}