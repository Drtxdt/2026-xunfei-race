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
        lost_line_count_ = 0.0;
        last_angular_ = 0.0;
        last_line_time_ = ros::Time::now();

        start_moving_time_ = ros::Time::now();

        // ===== 修改 ===== 左转预处理标志（丢线后先停下原地左转）
        need_left_turn_ = false;

        // ===== 修改 ===== PID 积分项 & 微分限幅相关运行时变量
        integral_pos_error_ = 0.0;

        // ===== 修改 ===== “检测到右转”触发计数
        right_turn_count_ = 0.0;
        right_turn_cooldown_until_ = ros::Time::now();

        ROS_INFO("=================================");
        ROS_INFO(" Stable Right Follow (Tuned) ");
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

    // ========== 参数 ==========
    int target_right_x_;
    double base_speed_;
    double curve_speed_;
    double search_speed_;
    double lost_line_speed_;
    double startup_speed_;

    double kp_pos_;
    double kd_pos_;
    double kp_angle_;
    // ===== 修改 ===== 新增积分项，消除弯道后长期贴右线的稳态误差
    double ki_pos_;
    double integral_clamp_;
    // ===== 修改 ===== 微分项限幅，抑制入弯瞬间误差跳变造成的“微分冲击”导致的过度左转
    double d_error_clamp_;
    // ===== 修改 ===== 硬安全下限：距离右线过近时强制的最小左转角速度 & 限速比例
    double hard_safety_margin_px_;
    double safety_min_angular_;
    double safety_speed_scale_;

    // ===== 修改 ===== 用“检测到右转”代替“丢失右线”作为停车左转修正的触发条件
    double right_turn_angle_threshold_deg_;
    int right_turn_trigger_count_;
    double right_turn_count_;
    // ===== 修改 ===== 触发后的冷却时间，防止角度噪声导致反复停车-左转（"一抽一抽"）
    double right_turn_cooldown_sec_;
    ros::Time right_turn_cooldown_until_;

    double curve_threshold_;
    double curve_offset_;
    double curve_gain_;

    double max_angular_;
    double error_filter_alpha_;

    double startup_time_;

    int cross_area_threshold_;
    int stop_line_min_width_;
    int stop_line_max_height_;
    int stop_line_min_area_;
    double stop_line_min_fill_ratio_;
    double stop_line_bottom_margin_ratio_;

    double align_speed_;
    double align_angle_threshold_;
    double align_stop_time_;
    double desired_angle_deg_;
    double align_angular_speed_;

    double final_speed_;
    double final_distance_;

    bool show_debug_;

    // ========== 运行时变量 ==========
    double last_pos_error_;
    double filtered_pos_error_;
    // ===== 修改 ===== 积分误差累计
    double integral_pos_error_;
    int last_right_x_;

    Stage stage_;
    ros::Time start_time_;
    ros::Time stage_start_time_;
    ros::Time forward_start_time_;

    int predicted_right_x_;
    // ===== 修改 ===== 改为浮点数并支持“找到线时缓慢衰减”而不是瞬间清零，
    // 避免弯道处右线检测偶尔闪现导致丢线计数被强制清零、从而第二次不再触发左转
    double lost_line_count_;
    double last_angular_;
    ros::Time last_line_time_;

    ros::Time start_moving_time_;
    double stop_line_ignore_time_ = 10.0;

    // ===== 修改 ===== 丢线后先停车、原地左转的相关参数与状态
    bool need_left_turn_;
    ros::Time left_turn_start_time_;
    double lost_line_stop_duration_;   // 停车+左转总时长（1~2秒）
    double lost_line_turn_angle_deg_;  // 期望原地左转角度（约12°）

    // ===== 修改 ===== 左转修正完成后，缓慢向右扫描找线的速度参数
    double search_right_speed_;
    double search_right_angular_;

    // ===== 修改 ===== 左转时若右线仍可见且过近，额外增大角速度，主动远离右线
    double left_turn_safety_margin_px_;
    double left_turn_extra_gain_;
    double left_turn_max_angular_;

    // ========== 参数加载 ==========
    void loadParams(ros::NodeHandle& pnh)
    {
        pnh.param("target_right_x", target_right_x_, 145);

        pnh.param("base_speed", base_speed_, 0.30);
        pnh.param("curve_speed", curve_speed_, 0.24);
        pnh.param("search_speed", search_speed_, 0.12);
        pnh.param("lost_line_speed", lost_line_speed_, 0.14);
        pnh.param("startup_speed", startup_speed_, 0.45);

        pnh.param("kp_pos", kp_pos_, 0.0040);
        pnh.param("kd_pos", kd_pos_, 0.0020);
        pnh.param("kp_angle", kp_angle_, 0.30);
        // ===== 修改 ===== 积分项 & 微分限幅
        pnh.param("ki_pos", ki_pos_, 0.00035);
        pnh.param("integral_clamp", integral_clamp_, 60.0);
        pnh.param("d_error_clamp", d_error_clamp_, 25.0);
        pnh.param("hard_safety_margin_px", hard_safety_margin_px_, 20.0);
        pnh.param("safety_min_angular", safety_min_angular_, 0.22);
        pnh.param("safety_speed_scale", safety_speed_scale_, 0.6);

        // ===== 修改 ===== 用“检测到右转”代替“丢失右线”触发停车+左转修正
        pnh.param("right_turn_angle_threshold_deg", right_turn_angle_threshold_deg_, 15.0);
        pnh.param("right_turn_trigger_count", right_turn_trigger_count_, 6);
        pnh.param("right_turn_cooldown_sec", right_turn_cooldown_sec_, 2.5);

        pnh.param("curve_threshold", curve_threshold_, 35.0);
        pnh.param("curve_offset", curve_offset_, 15.0);
        pnh.param("curve_gain", curve_gain_, 1.0);

        pnh.param("max_angular", max_angular_, 0.45);
        pnh.param("error_filter_alpha", error_filter_alpha_, 0.18);

        // ===== 修改 ===== 开机直行距离增加到之前的1.7倍（1.4 * 1.7 ≈ 2.4）
        pnh.param("startup_time", startup_time_, 2.4);

        pnh.param("cross_area_threshold", cross_area_threshold_, 48000);
        pnh.param("stop_line_min_width", stop_line_min_width_, 120);
        pnh.param("stop_line_max_height", stop_line_max_height_, 40);
        pnh.param("stop_line_min_area", stop_line_min_area_, 800);
        // ===== 修改 ===== 放宽识别门槛以提升召回率，同时仍能过滤明显噪声
        pnh.param("stop_line_min_fill_ratio", stop_line_min_fill_ratio_, 0.35);
        pnh.param("stop_line_bottom_margin_ratio", stop_line_bottom_margin_ratio_, 0.40);

        pnh.param("align_speed", align_speed_, 0.18);
        pnh.param("align_angle_threshold", align_angle_threshold_, 1.0);
        // ===== 修改 ===== 停车确认等待时间（很短，只是防抖），角度修正在 ALIGN 阶段进行
        pnh.param("align_stop_time", align_stop_time_, 0.3);
        pnh.param("desired_angle_deg", desired_angle_deg_, -5.0);
        // ===== 修改 ===== ALIGN 阶段盲转角速度，需与 desired_angle_deg_ 配合，
        // 使 “停车确认 + 原地转5°” 总时长落在 1~2 秒
        pnh.param("align_angular_speed", align_angular_speed_, 0.10);

        pnh.param("final_speed", final_speed_, 0.20);
        // ===== 修改 ===== 直行距离与原来一致（此项不是用户要求减半的那个直行距离）
        pnh.param("final_distance", final_distance_, 0.60);

        pnh.param("show_debug", show_debug_, true);

        // ===== 修改 ===== 丢线后先停车、原地左转参数
        pnh.param("lost_line_stop_duration", lost_line_stop_duration_, 1.5);
        // ===== 修改 ===== 25° → 12°，左转幅度减小
        pnh.param("lost_line_turn_angle_deg", lost_line_turn_angle_deg_, 12.0);

        // ===== 修改 ===== 左转修正完成后，缓慢向右扫描找线（比之前的0.12/-0.26更慢更稳）
        pnh.param("search_right_speed", search_right_speed_, 0.08);
        pnh.param("search_right_angular", search_right_angular_, 0.14);

        pnh.param("left_turn_safety_margin_px", left_turn_safety_margin_px_, 40.0);
        pnh.param("left_turn_extra_gain", left_turn_extra_gain_, 0.01);
        pnh.param("left_turn_max_angular", left_turn_max_angular_, 0.9);
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
        start_moving_time_ = ros::Time::now();
        enterStage(SEARCH_RIGHT_LINE, "ENTER SEARCH MODE (half distance startup done)");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleSearch(geometry_msgs::Twist& twist, const LineInfo& right_line)
    {
        // ===== 修改 ===== 丢线恢复：先原地停车（linear=0）在1~2秒内完成左转角度修正，
        // 不再一边前进一边转，避免第二次曲线处因视觉闪烁导致修正被跳过/距离右线过近
        if(need_left_turn_)
        {
            double elapsed = (ros::Time::now() - left_turn_start_time_).toSec();

            double planned_angular =
                deg2rad(lost_line_turn_angle_deg_) / std::max(0.1, lost_line_stop_duration_);

            // 若右线仍可见且距离过近，适当加大角速度，主动增加与右线的距离
            if(right_line.found)
            {
                double closeness =
                    (target_right_x_ - left_turn_safety_margin_px_) - right_line.x;
                if(closeness > 0.0)
                {
                    planned_angular += closeness * left_turn_extra_gain_;
                }
            }
            planned_angular = clamp(planned_angular, 0.0, left_turn_max_angular_);

            if(elapsed < lost_line_stop_duration_)
            {
                twist.linear.x = 0.0;          // 完全停车，只做原地旋转修正
                twist.angular.z = planned_angular;
                return;
            }
            else
            {
                need_left_turn_ = false;   // 左转完成，进入正常搜索
            }
        }

        // ===== 修改 ===== 原地左转修正完成后，改为缓慢向右扫描找线（降低角速度，避免又转过头）
        if(!right_line.found)
        {
            twist.linear.x = search_right_speed_;
            twist.angular.z = -search_right_angular_;
            return;
        }

        last_right_x_ = right_line.x;
        resetPid();
        lost_line_count_ = 0.0;
        right_turn_count_ = 0.0;
        predicted_right_x_ = right_line.x;
        enterStage(FOLLOW_RIGHT_LINE, "RIGHT LINE FOUND");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleFollow(geometry_msgs::Twist& twist,
                      const LineInfo& right_line,
                      const StopLineInfo& stop_line)
    {
        double elapsed_since_move = (ros::Time::now() - start_moving_time_).toSec();
        bool ignore_stop_line = (elapsed_since_move < stop_line_ignore_time_);

        if(stop_line.found && !ignore_stop_line)
        {
            resetPid();
            enterStage(STOP_LINE_FOUND, "STOP LINE DETECTED");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            return;
        }

        // ===== 修改 ===== 触发条件由“丢失右线”改为“检测到右转”：
        // 只要还能看到右线，就用其拟合角度判断是否正在进入右转弯道，
        // 提前原地停车+左转修正，而不是等到线完全丢失才反应（丢失往往已经太晚/太近了）
        int current_x = right_line.found ? right_line.x : predicted_right_x_;
        double current_angle = right_line.found ? right_line.angle_deg : desired_angle_deg_;

        if(right_line.found)
        {
            last_right_x_ = right_line.x;
            predicted_right_x_ = 0.7 * predicted_right_x_ + 0.3 * right_line.x;
            last_line_time_ = ros::Time::now();

            // ===== 修改 ===== 必须角度偏差 且 位置误差同时偏大，才算真正进入右转弯道；
            // 单纯角度读数噪声抖一下（直道上也可能发生）不会累积触发，避免反复停车-左转
            double angle_dev = std::fabs(current_angle - desired_angle_deg_);
            bool angle_bad = angle_dev > right_turn_angle_threshold_deg_;
            bool pos_bad = std::fabs(filtered_pos_error_) > curve_threshold_;
            if(angle_bad && pos_bad)
            {
                right_turn_count_ += 1.0;
            }
            else
            {
                right_turn_count_ = std::max(0.0, right_turn_count_ - 1.0);
            }
            lost_line_count_ = std::max(0.0, lost_line_count_ - 2.0);
        }
        else
        {
            lost_line_count_ += 1.0;
        }

        // ===== 修改 ===== 冷却期内不重新触发，防止刚做完一次停车-左转、
        // 车还没稳定住又被噪声立刻拉回来，导致“一抽一抽”走不动
        bool in_cooldown = ros::Time::now() < right_turn_cooldown_until_;

        if(!in_cooldown && right_turn_count_ >= right_turn_trigger_count_)
        {
            right_turn_count_ = 0.0;
            right_turn_cooldown_until_ = ros::Time::now() + ros::Duration(right_turn_cooldown_sec_);
            need_left_turn_ = true;
            left_turn_start_time_ = ros::Time::now();
            enterStage(SEARCH_RIGHT_LINE, "RIGHT TURN DETECTED, STOP & LEFT TURN");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            return;
        }

        // 丢线兜底：即便角度检测没提前抓到，线真的完全丢失时依然要触发同一套停车+左转
        if(!in_cooldown && lost_line_count_ > 10.0)
        {
            right_turn_cooldown_until_ = ros::Time::now() + ros::Duration(right_turn_cooldown_sec_);
            need_left_turn_ = true;
            left_turn_start_time_ = ros::Time::now();
            enterStage(SEARCH_RIGHT_LINE, "LINE LOST (fallback), STOP & LEFT TURN");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            return;
        }

        const bool in_curve =
            std::fabs(filtered_pos_error_) > curve_threshold_ ||
            std::fabs(current_angle - desired_angle_deg_) > align_angle_threshold_;

        const double target = target_right_x_ - (in_curve ? curve_offset_ : 0.0);
        const double pos_error = target - current_x;

        filtered_pos_error_ =
            (1.0 - error_filter_alpha_) * filtered_pos_error_ +
            error_filter_alpha_ * pos_error;

        // ===== 修改 ===== 微分项限幅：防止入弯瞬间误差跳变产生“微分冲击”，
        // 这是之前“第一次拐弯处向左转过多、压到左边线”的主要原因之一
        double d_pos_error = filtered_pos_error_ - last_pos_error_;
        d_pos_error = clamp(d_pos_error, -d_error_clamp_, d_error_clamp_);
        last_pos_error_ = filtered_pos_error_;

        // ===== 修改 ===== 积分项（带抗饱和限幅）：消除弯道之后长期贴右线的稳态偏差
        integral_pos_error_ = clamp(
            integral_pos_error_ + filtered_pos_error_,
            -integral_clamp_, integral_clamp_);

        const double angle_error = current_angle - desired_angle_deg_;

        double angular =
            kp_pos_ * filtered_pos_error_ +
            kd_pos_ * d_pos_error +
            ki_pos_ * integral_pos_error_ +
            kp_angle_ * deg2rad(angle_error);

        double linear_speed = in_curve ? curve_speed_ : base_speed_;

        if(in_curve)
        {
            angular *= curve_gain_;
        }

        angular = clamp(angular, -max_angular_, max_angular_);

        // ===== 修改 ===== 硬安全下限：只要距离右线过近（不管是否在弯道/PID算出多少），
        // 强制保证至少有这么大的左转修正，并降低线速度给出更多反应时间。
        // 这是针对“一直贴右线太近、压线”这个持续性问题的兜底措施。
        if(current_x < target_right_x_ - hard_safety_margin_px_)
        {
            angular = std::max(angular, safety_min_angular_);
            linear_speed = std::min(linear_speed, base_speed_ * safety_speed_scale_);
        }

        twist.linear.x = linear_speed;
        twist.angular.z = angular;
    }

    void handleStopLineFound(geometry_msgs::Twist& twist)
    {
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        // ===== 修改 ===== 这里只是短暂防抖确认停止线（不做转向），
        // 真正的角度修正在紧接着的 ALIGN 阶段（同样保持停车状态）完成，
        // 两段相加控制在 1~2 秒内
        if(elapsed >= align_stop_time_)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "ENTER ALIGN MODE (stop & correct 5 deg)");
        }
        if(elapsed > 2.0)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "STOPLINE TIMEOUT");
        }
    }

    // ===== 修改 ===== 对齐阶段：全程保持停车（linear=0），原地左转 |desired_angle_deg_|（5°），
    // 不依赖视觉持续跟踪，转动时长由 align_angular_speed_ 决定，
    // 与 STOP_LINE_FOUND 的短暂确认时间相加，总停车+修正时间落在 1~2 秒
    void handleAlign(geometry_msgs::Twist& twist, const StopLineInfo& /*stop_line*/)
    {
        twist.linear.x = 0.0;

        const double turn_duration =
            std::fabs(deg2rad(desired_angle_deg_)) / std::max(0.01, align_angular_speed_);
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();

        if(elapsed < turn_duration)
        {
            twist.angular.z = align_angular_speed_;   // 正值 = 左转
            return;
        }

        twist.angular.z = 0.0;
        enterStage(GO_FORWARD, "ALIGN DONE (5 DEG LEFT TURN)");
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

    // ========== 图像处理（保持不变） ==========
    cv::Mat extractWhiteMask(const cv::Mat& roi)
    {
        cv::Mat blur;
        cv::GaussianBlur(roi, blur, cv::Size(5,5),0);
        cv::Mat hsv;
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(0,0,200), cv::Scalar(180,45,255), mask);

        cv::Mat kernel = cv::Mat::ones(5,5, CV_8U);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(mask, mask, 5);
        cv::GaussianBlur(mask, mask, cv::Size(5,5), 0);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat clean = cv::Mat::zeros(mask.size(), CV_8UC1);
        for(const auto& cnt : contours)
        {
            if(cv::contourArea(cnt) > 260.0)
                cv::drawContours(clean, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), -1);
        }
        return clean;
    }

    LineInfo findRightLine(const cv::Mat& mask)
    {
        LineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;
        for(int y = static_cast<int>(h*0.20); y < static_cast<int>(h*0.92); y+=4)
        {
            const uchar* ptr = mask.ptr<uchar>(y);
            for(int x = w-1; x>=0; --x)
            {
                if(ptr[x] > 0)
                {
                    info.points.push_back(cv::Point(x,y));
                    break;
                }
            }
        }
        if(info.points.size() < 6)
        {
            // ===== 修改 ===== 这里不再直接修改 lost_line_count_（避免和 handleFollow 中的
            // 累计逻辑重复计数，导致丢线计数增长过快或不一致）
            info.found = false;
            info.x = predicted_right_x_;
            info.angle_deg = desired_angle_deg_;
            return info;
        }
        double x_sum = 0.0;
        int x_count = 0;
        for(const auto& p : info.points)
        {
            if(p.y > h*0.45) { x_sum += p.x; x_count++; }
        }
        if(x_count == 0)
        {
            for(const auto& p : info.points) x_sum += p.x;
            x_count = static_cast<int>(info.points.size());
        }
        cv::fitLine(info.points, info.fit_line, cv::DIST_L2, 0.0, 0.01, 0.01);
        const double vx = info.fit_line[0];
        const double vy = info.fit_line[1];
        info.found = true;
        info.x = static_cast<int>(x_sum/x_count);
        info.angle_deg = rad2deg(std::atan2(vx,vy));
        return info;
    }

    // ===== 修改 ===== 停车横线识别：放宽召回（fill_ratio / bottom_margin 阈值降低），
    // 同时保留基本几何过滤，减少“识别不到第一条横线”的漏检
    StopLineInfo findStopLine(const cv::Mat& mask)
    {
        StopLineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;
        cv::Mat bottom = mask(cv::Range(static_cast<int>(h*0.65), h), cv::Range(0,w));
        const int bottom_h = bottom.rows;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bottom.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double max_area = 0;
        int best_idx = -1;
        double best_reject_area = 0.0;
        cv::Rect best_reject_rect;
        for(size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if(area < stop_line_min_area_) continue;
            cv::Rect rect = cv::boundingRect(contours[i]);
            if(rect.width < rect.height*4) continue;
            if(rect.height > stop_line_max_height_) continue;
            if(rect.width < stop_line_min_width_) continue;

            double fill_ratio = area / std::max(1.0, static_cast<double>(rect.width * rect.height));
            double bottom_edge = rect.y + rect.height;

            if(fill_ratio < stop_line_min_fill_ratio_ || bottom_edge < bottom_h * stop_line_bottom_margin_ratio_)
            {
                // ===== 修改 ===== 诊断日志：记录“差一点被判定为停车线”的候选框，
                // 方便在没有画面的情况下通过 rosout 判断具体是哪个阈值卡住了
                if(area > best_reject_area) { best_reject_area = area; best_reject_rect = rect; }
                continue;
            }

            if(contours[i].size() >= 5)
            {
                cv::Vec4f line;
                cv::fitLine(contours[i], line, cv::DIST_L2, 0, 0.01, 0.01);
                if(std::fabs(rad2deg(std::atan2(line[1], line[0]))) > 15.0) continue;
            }
            if(area > max_area) { max_area = area; best_idx = i; }
        }
        if(best_idx < 0 && best_reject_area > 0.0)
        {
            double fr = best_reject_area / std::max(1.0, static_cast<double>(best_reject_rect.width * best_reject_rect.height));
            double be = best_reject_rect.y + best_reject_rect.height;
            ROS_INFO_THROTTLE(1.0,
                "[StopLine reject] area=%.0f w=%d h=%d fill=%.2f(need>=%.2f) bottom=%.0f(need>=%.0f)",
                best_reject_area, best_reject_rect.width, best_reject_rect.height,
                fr, stop_line_min_fill_ratio_, be, bottom_h * stop_line_bottom_margin_ratio_);
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
            info.center_x = rect.x + rect.width/2;
        }
        if(!info.found && cross_area_threshold_ > 0)
        {
            if(cv::countNonZero(mask) > cross_area_threshold_)
            {
                info.found = true;
                info.rect = cv::Rect(0,0,w,h);
                info.angle_deg = 0.0;
                info.center_x = w/2;
            }
        }

        // ===== 修改 ===== 行密度兜底检测：如果轮廓法仍未识别到，
        // 逐行检查底部区域是否有一整条“横向大片白色”（覆盖率高的行），
        // 这种情况通常就是停车线，但因形状/断裂被轮廓过滤掉了
        if(!info.found)
        {
            const double row_white_ratio_thresh = 0.5;
            const int min_consecutive_rows = 4;
            int run_start = -1;
            int run_len = 0;
            int best_start = -1, best_len = 0;
            for(int y = 0; y < bottom_h; ++y)
            {
                int white_count = cv::countNonZero(bottom.row(y));
                double ratio = static_cast<double>(white_count) / std::max(1, w);
                if(ratio >= row_white_ratio_thresh)
                {
                    if(run_start < 0) run_start = y;
                    run_len++;
                }
                else
                {
                    if(run_len > best_len) { best_len = run_len; best_start = run_start; }
                    run_start = -1;
                    run_len = 0;
                }
            }
            if(run_len > best_len) { best_len = run_len; best_start = run_start; }

            if(best_len >= min_consecutive_rows)
            {
                info.found = true;
                info.rect = cv::Rect(0, best_start, w, best_len);
                info.angle_deg = 0.0;
                info.center_x = w/2;
                ROS_INFO_THROTTLE(1.0,
                    "[StopLine row-scan fallback] rows=%d start=%d (ratio>=%.2f)",
                    best_len, best_start, row_white_ratio_thresh);
            }
        }
        return info;
    }

    // ========== 调试显示（保持不变） ==========
    void showDebug(const cv::Mat& mask, const LineInfo& right_line,
                   const StopLineInfo& stop_line, const geometry_msgs::Twist& twist)
    {
        cv::Mat debug;
        cv::cvtColor(mask, debug, cv::COLOR_GRAY2BGR);
        const bool in_curve = std::fabs(filtered_pos_error_) > curve_threshold_;
        const int active_target = target_right_x_ - (in_curve ? static_cast<int>(curve_offset_) : 0);
        cv::line(debug, cv::Point(target_right_x_,0), cv::Point(target_right_x_, mask.rows), cv::Scalar(255,0,0),2);
        cv::line(debug, cv::Point(active_target,0), cv::Point(active_target, mask.rows), cv::Scalar(0,255,255),2);
        if(right_line.found)
        {
            for(const auto& p : right_line.points) cv::circle(debug, p, 2, cv::Scalar(0,180,255), -1);
            drawFitLine(debug, right_line.fit_line, cv::Scalar(0,0,255));
            cv::circle(debug, cv::Point(right_line.x, mask.rows/2), 5, cv::Scalar(0,0,255), -1);
        }
        if(stop_line.found)
        {
            cv::rectangle(debug, stop_line.rect, cv::Scalar(0,255,0), 2);
            char buf[32];
            snprintf(buf, sizeof(buf), "Ang:%.1f", stop_line.angle_deg);
            cv::putText(debug, buf, cv::Point(stop_line.rect.x, stop_line.rect.y-10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
        }
        double elapsed = (ros::Time::now() - start_moving_time_).toSec();
        drawText(debug, 20,30, "stage: " + stageName(stage_));
        drawText(debug, 20,60, format("target: %d", active_target));
        drawText(debug, 20,90, format("err: %.2f", filtered_pos_error_));
        drawText(debug, 20,120, format("time: %.1fs", elapsed));
        drawText(debug, 20,150, format("cmd w: %.3f", twist.angular.z));
        drawText(debug, 20,180, format("lost: %.1f", lost_line_count_));
        drawText(debug, 20,210, format("rturn: %.1f", right_turn_count_));
        cv::imshow("right_follow", debug);
        cv::waitKey(1);
    }

    void drawFitLine(cv::Mat& img, const cv::Vec4f& line, const cv::Scalar& color)
    {
        float vx=line[0], vy=line[1], x0=line[2], y0=line[3];
        if(std::fabs(vy)<1e-5) return;
        int y1=0, y2=img.rows-1;
        int x1 = static_cast<int>(x0 + (y1-y0)*vx/vy);
        int x2 = static_cast<int>(x0 + (y2-y0)*vx/vy);
        cv::line(img, cv::Point(clampInt(x1,0,img.cols-1), y1),
                 cv::Point(clampInt(x2,0,img.cols-1), y2), color, 2);
    }

    void drawText(cv::Mat& img, int x, int y, const std::string& text)
    {
        cv::putText(img, text, cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,255,0),2);
    }

    // ========== 辅助函数 ==========
    void enterStage(Stage stage, const char* message)
    {
        stage_ = stage;
        stage_start_time_ = ros::Time::now();
        if(stage == GO_FORWARD) forward_start_time_ = ros::Time::now();
        ROS_INFO("%s", message);
    }

    void resetPid()
    {
        last_pos_error_ = 0.0;
        filtered_pos_error_ = 0.0;
        integral_pos_error_ = 0.0;   // ===== 修改 ===== 每次重新找到线时清空积分，避免历史误差带入
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
        case STARTUP: return "STARTUP";
        case SEARCH_RIGHT_LINE: return "SEARCH";
        case FOLLOW_RIGHT_LINE: return "FOLLOW";
        case STOP_LINE_FOUND: return "STOP_LINE";
        case ALIGN_WITH_RIGHT_LINE: return "ALIGN";
        case GO_FORWARD: return "FORWARD";
        case FINAL_STOP: return "FINAL_STOP";
        default: return "UNKNOWN";
        }
    }

    std::string format(const char* fmt, double value)
    {
        char buf[80]; std::snprintf(buf, sizeof(buf), fmt, value); return buf;
    }
    std::string format(const char* fmt, int value)
    {
        char buf[80]; std::snprintf(buf, sizeof(buf), fmt, value); return buf;
    }

    double deg2rad(double deg) const { return deg * CV_PI / 180.0; }
    double rad2deg(double rad) const { return rad * 180.0 / CV_PI; }
    double clamp(double v, double lo, double hi) const { return std::max(lo, std::min(v, hi)); }
    int clampInt(int v, int lo, int hi) const { return std::max(lo, std::min(v, hi)); }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stable_right_follow_cpp");
    StableRightFollowNode node;
    ros::spin();
    return 0;
}