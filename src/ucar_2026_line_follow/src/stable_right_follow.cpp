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
#include <deque>

// ---------- 一维卡尔曼滤波器（位置 + 速度）----------
class KalmanFilter1D
{
public:
    KalmanFilter1D()
    {
        reset();
    }

    void reset()
    {
        x_ = 0.0;
        v_ = 0.0;
        P_ = cv::Mat::eye(2, 2, CV_64F) * 1000.0;
        initialized_ = false;
    }

    void predict(double dt = 1.0)
    {
        // 状态转移矩阵 F = [1, dt; 0, 1]
        cv::Mat F = (cv::Mat_<double>(2, 2) << 1.0, dt, 0.0, 1.0);
        // 过程噪声协方差 Q
        cv::Mat Q = (cv::Mat_<double>(2, 2) << 0.01, 0.0, 0.0, 0.01);

        x_ = F.at<double>(0,0) * x_ + F.at<double>(0,1) * v_;
        v_ = F.at<double>(1,0) * x_ + F.at<double>(1,1) * v_;  // 实际上 v_ 不变，但为通用性保留
        P_ = F * P_ * F.t() + Q;
    }

    void update(double measurement, double dt = 1.0)
    {
        if (!initialized_)
        {
            x_ = measurement;
            v_ = 0.0;
            P_ = cv::Mat::eye(2, 2, CV_64F) * 100.0;
            initialized_ = true;
            return;
        }

        predict(dt);

        cv::Mat H = (cv::Mat_<double>(1, 2) << 1.0, 0.0);
        cv::Mat R = (cv::Mat_<double>(1, 1) << 5.0);  // 测量噪声

        cv::Mat y = (cv::Mat_<double>(1, 1) << measurement - x_);
        cv::Mat S = H * P_ * H.t() + R;
        cv::Mat K = P_ * H.t() * S.inv();

        cv::Mat correction = K * y;
        x_ += correction.at<double>(0,0);
        v_ += correction.at<double>(1,0);

        cv::Mat I = cv::Mat::eye(2, 2, CV_64F);
        P_ = (I - K * H) * P_;
    }

    double getPosition() const { return x_; }
    double getVelocity() const { return v_; }

private:
    double x_, v_;
    cv::Mat P_;
    bool initialized_;
};

// ---------- 主节点 ----------
class StableRightFollowNode
{
public:
    StableRightFollowNode()
    {
        ros::NodeHandle pnh("~");
        loadParams(pnh);

        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        image_sub_ = nh_.subscribe(
            "/usb_cam/image_raw", 1,
            &StableRightFollowNode::imageCallback, this);

        // 状态机初始化
        stage_ = STARTUP;
        start_time_ = ros::Time::now();

        // 卡尔曼初始化
        kalman_.reset();

        // 右转防压线相关
        curve_offset_accum_ = 0.0;
        target_curve_offset_ = 0.0;
        curve_recovery_step_ = 0.0;
        lost_right_frames_ = 0;
        last_angle_ = 0.0;

        // 停车对准相关
        align_angle_ok_frames_ = 0;
        stop_line_confirm_count_ = 0;
        last_stop_line_ = StopLineInfo();

        // PID 重置
        resetPid();

        ROS_INFO("=================================");
        ROS_INFO(" Stable Right Follow (Kalman+RANSAC) ");
        ROS_INFO("=================================");
    }

private:
    // ---------- 状态枚举 ----------
    enum Stage
    {
        STARTUP = 0,
        SEARCH_RIGHT_LINE,
        FOLLOW_RIGHT_LINE,
        RIGHT_CURVE_PREDICT,   // 右转弯丢失预测
        STOP_LINE_FOUND,
        ALIGN_WITH_RIGHT_LINE,
        GO_FORWARD,
        FINAL_STOP
    };

    // ---------- 数据结构 ----------
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

    // ---------- ROS 接口 ----------
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber image_sub_;

    // ---------- 参数 ----------
    int target_right_x_;
    double base_speed_, curve_speed_, search_speed_, lost_line_speed_, startup_speed_;
    double kp_pos_, kd_pos_, kp_angle_;
    double curve_threshold_, curve_offset_base_, curve_offset_max_;
    double max_angular_, error_filter_alpha_;
    double startup_time_;

    int stop_line_min_width_, stop_line_max_height_, stop_line_min_area_;
    double align_speed_, align_angle_threshold_, align_stop_time_;
    double desired_angle_deg_;
    double final_speed_, final_distance_;

    bool show_debug_;

    // 曲线预测新增参数
    double curve_predict_speed_;      // 曲线预测时的线速度
    double curve_angular_hold_;       // 曲线预测时保持的角速度
    int curve_lost_frames_threshold_; // 触发预测的连续丢线帧数
    double curve_angle_threshold_;    // 上一次角度大于此值才触发
    double curve_offset_step_;        // 每次偏移增量 (像素)
    double curve_recovery_rate_;      // 回归速率 (每帧恢复像素)

    // ---------- 运行时变量 ----------
    Stage stage_;
    ros::Time start_time_;
    ros::Time stage_start_time_;
    ros::Time forward_start_time_;

    double last_pos_error_;
    double filtered_pos_error_;
    int last_right_x_;

    KalmanFilter1D kalman_;           // 卡尔曼滤波器

    // 右转防压线
    double curve_offset_accum_;       // 当前累计偏移量
    double target_curve_offset_;      // 目标偏移量
    double curve_recovery_step_;      // 恢复步长
    int lost_right_frames_;           // 连续丢失帧数
    double last_angle_;               // 上一帧右线角度

    // 停车对准
    int align_angle_ok_frames_;
    int stop_line_confirm_count_;
    StopLineInfo last_stop_line_;

    double last_angular_;
    double last_vx_, last_vy_;

    // 辅助
    std::deque<double> angle_history_; // 角度历史用于平滑

    // ---------- 参数加载 ----------
    void loadParams(ros::NodeHandle& pnh)
    {
        pnh.param("target_right_x", target_right_x_, 145);

        pnh.param("base_speed", base_speed_, 0.22);
        pnh.param("curve_speed", curve_speed_, 0.18);
        pnh.param("search_speed", search_speed_, 0.12);
        pnh.param("lost_line_speed", lost_line_speed_, 0.14);
        pnh.param("startup_speed", startup_speed_, 0.45);

        pnh.param("kp_pos", kp_pos_, 0.0035);
        pnh.param("kd_pos", kd_pos_, 0.0025);
        pnh.param("kp_angle", kp_angle_, 0.25);

        pnh.param("curve_threshold", curve_threshold_, 35.0);
        pnh.param("curve_offset_base", curve_offset_base_, 35.0);
        pnh.param("curve_offset_max", curve_offset_max_, 50.0);

        pnh.param("max_angular", max_angular_, 0.40);
        pnh.param("error_filter_alpha", error_filter_alpha_, 0.30);

        pnh.param("startup_time", startup_time_, 2.8);

        pnh.param("stop_line_min_width", stop_line_min_width_, 120);
        pnh.param("stop_line_max_height", stop_line_max_height_, 40);
        pnh.param("stop_line_min_area", stop_line_min_area_, 800);

        pnh.param("align_speed", align_speed_, 0.18);
        pnh.param("align_angle_threshold", align_angle_threshold_, 0.5);
        pnh.param("align_stop_time", align_stop_time_, 1.5);
        pnh.param("desired_angle_deg", desired_angle_deg_, 0.0);

        pnh.param("final_speed", final_speed_, 0.20);
        pnh.param("final_distance", final_distance_, 0.60);

        pnh.param("show_debug", show_debug_, true);

        // 曲线预测参数
        pnh.param("curve_predict_speed", curve_predict_speed_, 0.12);
        pnh.param("curve_angular_hold", curve_angular_hold_, -0.30);
        pnh.param("curve_lost_frames_threshold", curve_lost_frames_threshold_, 3);
        pnh.param("curve_angle_threshold", curve_angle_threshold_, 18.0);
        pnh.param("curve_offset_step", curve_offset_step_, 35.0);
        pnh.param("curve_recovery_rate", curve_recovery_rate_, 5.0);
    }

    // ---------- 图像回调 ----------
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
        cv::Mat roi = frame(cv::Range(static_cast<int>(h * 0.45), h), cv::Range(0, w));

        // 新的白线提取 (HSV + 自适应阈值 + 形态学)
        cv::Mat mask = extractWhiteMaskAdvanced(roi);

        // 检测右线、左线、停车线
        LineInfo right_line = findRightLineAdvanced(mask);
        LineInfo left_line = findLeftLine(mask);
        StopLineInfo stop_line = findStopLine(mask);

        // 更新卡尔曼滤波器
        if(right_line.found)
        {
            kalman_.update(right_line.x);
            last_right_x_ = static_cast<int>(kalman_.getPosition());
        }
        else
        {
            kalman_.predict(); // 仅预测
        }
        int predicted_x = static_cast<int>(kalman_.getPosition());

        // 角度平滑
        if(right_line.found)
        {
            last_angle_ = right_line.angle_deg;
            angle_history_.push_back(right_line.angle_deg);
            if(angle_history_.size() > 10) angle_history_.pop_front();
        }

        geometry_msgs::Twist twist;
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        switch(stage_)
        {
        case STARTUP:
            handleStartup(twist);
            break;
        case SEARCH_RIGHT_LINE:
            handleSearch(twist, right_line, predicted_x);
            break;
        case FOLLOW_RIGHT_LINE:
            handleFollow(twist, right_line, left_line, stop_line, predicted_x);
            break;
        case RIGHT_CURVE_PREDICT:
            handleCurvePredict(twist, right_line, predicted_x);
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

        // 角速度平滑
        twist.angular.z = 0.7 * last_angular_ + 0.3 * twist.angular.z;
        last_angular_ = twist.angular.z;

        cmd_pub_.publish(twist);

        if(show_debug_)
        {
            showDebug(mask, right_line, left_line, stop_line, twist, predicted_x);
        }
    }

    // ---------- 状态机处理 ----------
    void handleStartup(geometry_msgs::Twist& twist)
    {
        double elapsed = (ros::Time::now() - start_time_).toSec();
        if(elapsed < startup_time_)
        {
            twist.linear.x = startup_speed_;
            twist.angular.z = 0.0;
            return;
        }
        enterStage(SEARCH_RIGHT_LINE, "ENTER SEARCH");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleSearch(geometry_msgs::Twist& twist, const LineInfo& right_line, int predicted_x)
    {
        if(!right_line.found)
        {
            twist.linear.x = search_speed_;
            twist.angular.z = -0.26;
            return;
        }

        resetPid();
        kalman_.update(right_line.x);
        last_right_x_ = right_line.x;
        lost_right_frames_ = 0;
        curve_offset_accum_ = 0.0;
        target_curve_offset_ = 0.0;
        enterStage(FOLLOW_RIGHT_LINE, "RIGHT LINE FOUND");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleFollow(geometry_msgs::Twist& twist,
                      const LineInfo& right_line,
                      const LineInfo& left_line,
                      const StopLineInfo& stop_line,
                      int predicted_x)
    {
        // 停车线检测（连续确认）
        if(stop_line.found)
        {
            if(++stop_line_confirm_count_ >= 3)
            {
                resetPid();
                enterStage(STOP_LINE_FOUND, "STOP LINE DETECTED");
                twist.linear.x = 0.0;
                twist.angular.z = 0.0;
                return;
            }
            last_stop_line_ = stop_line;
        }
        else
        {
            stop_line_confirm_count_ = 0;
        }

        // 检查是否进入曲线预测模式
        if(!right_line.found)
        {
            lost_right_frames_++;
            // 条件：连续丢线 >= 阈值，上一帧角度大于阈值
            if(lost_right_frames_ >= curve_lost_frames_threshold_ &&
               std::fabs(last_angle_) > curve_angle_threshold_)
            {
                enterStage(RIGHT_CURVE_PREDICT, "ENTER CURVE PREDICT");
                // 设置目标偏移量
                target_curve_offset_ = curve_offset_base_;
                curve_recovery_step_ = 0.0;
                return;
            }
        }
        else
        {
            lost_right_frames_ = 0;
        }

        // 正常跟随逻辑
        if(right_line.found)
        {
            // 动态目标点：如果曲线偏移累积大于0，表示正在回归
            int active_target = target_right_x_ - static_cast<int>(curve_offset_accum_);
            double pos_error = active_target - right_line.x;

            filtered_pos_error_ = (1.0 - error_filter_alpha_) * filtered_pos_error_ +
                                  error_filter_alpha_ * pos_error;

            double d_pos_error = filtered_pos_error_ - last_pos_error_;
            last_pos_error_ = filtered_pos_error_;

            double angle_error = right_line.angle_deg - desired_angle_deg_;
            double angular = kp_pos_ * filtered_pos_error_ +
                             kd_pos_ * d_pos_error +
                             kp_angle_ * deg2rad(angle_error);

            bool in_curve = std::fabs(filtered_pos_error_) > curve_threshold_ ||
                            std::fabs(right_line.angle_deg) > curve_angle_threshold_;

            double linear_speed = base_speed_;
            if(in_curve) linear_speed = curve_speed_;
            if(std::fabs(filtered_pos_error_) > 60) linear_speed *= 0.8;

            angular = clamp(angular, -max_angular_, max_angular_);
            twist.linear.x = linear_speed;
            twist.angular.z = angular;

            // 如果累积偏移>0且右线已找到，逐步恢复偏移
            if(curve_offset_accum_ > 0.0)
            {
                curve_offset_accum_ -= curve_recovery_rate_;
                if(curve_offset_accum_ < 0.0) curve_offset_accum_ = 0.0;
            }
        }
        else
        {
            // 短暂丢线但未达到曲线预测条件，使用预测位置继续跟随
            double pos_error = target_right_x_ - predicted_x;
            double angular = kp_pos_ * pos_error;
            angular = clamp(angular, -max_angular_, max_angular_);
            twist.linear.x = lost_line_speed_;
            twist.angular.z = angular;
        }
    }

    void handleCurvePredict(geometry_msgs::Twist& twist,
                            const LineInfo& right_line,
                            int predicted_x)
    {
        // 线速度降低，角速度保持转弯
        twist.linear.x = curve_predict_speed_;
        // 如果上一帧角速度有效，沿用，否则用预设值
        twist.angular.z = (std::fabs(last_angular_) > 0.05) ? last_angular_ : curve_angular_hold_;

        // 逐步增大偏移量，让小车主动远离右线
        if(curve_offset_accum_ < target_curve_offset_)
        {
            curve_offset_accum_ += curve_offset_step_ * 0.1; // 缓慢增加
            if(curve_offset_accum_ > target_curve_offset_) curve_offset_accum_ = target_curve_offset_;
        }

        // 如果重新检测到右线且连续稳定，回归到 FOLLOW
        if(right_line.found)
        {
            static int recovery_frames = 0;
            recovery_frames++;
            if(recovery_frames >= 3)
            {
                recovery_frames = 0;
                // 保留当前偏移，后续在 FOLLOW 中逐步恢复
                enterStage(FOLLOW_RIGHT_LINE, "RETURN FROM CURVE PREDICT");
                return;
            }
        }
        else
        {
            // 如果长时间丢失，也可以考虑转为 SEARCH
            if(lost_right_frames_ > 50)
            {
                enterStage(SEARCH_RIGHT_LINE, "CURVE LOST TOO LONG");
                curve_offset_accum_ = 0.0;
                target_curve_offset_ = 0.0;
                return;
            }
        }
    }

    void handleStopLineFound(geometry_msgs::Twist& twist)
    {
        double elapsed = (ros::Time::now() - stage_start_time_).toSec();
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        // 必须停留 align_stop_time_ 秒
        if(elapsed >= align_stop_time_)
        {
            // 清空角度历史，准备对准
            angle_history_.clear();
            align_angle_ok_frames_ = 0;
            enterStage(ALIGN_WITH_RIGHT_LINE, "ENTER ALIGN");
        }
    }

    void handleAlign(geometry_msgs::Twist& twist, const StopLineInfo& stop_line)
    {
        twist.linear.x = 0.0;

        if(!stop_line.found)
        {
            // 横线丢失，尝试缓慢旋转寻找
            twist.angular.z = -0.12;
            if((ros::Time::now() - stage_start_time_).toSec() > 3.0)
            {
                enterStage(GO_FORWARD, "ALIGN LINE LOST, FORCE GO");
            }
            return;
        }

        double angle_error = stop_line.angle_deg - desired_angle_deg_;
        // 使用简单的比例控制旋转
        double angular_cmd = clamp(angle_error * 0.035, -0.18, 0.18);

        // 死区
        if(std::fabs(angle_error) < align_angle_threshold_)
            angular_cmd = 0.0;

        twist.angular.z = angular_cmd;

        // 连续满足角度要求的帧计数
        if(std::fabs(angle_error) < align_angle_threshold_)
        {
            align_angle_ok_frames_++;
            if(align_angle_ok_frames_ >= 15)
            {
                enterStage(GO_FORWARD, "ALIGN OK (15 frames)");
                align_angle_ok_frames_ = 0;
            }
        }
        else
        {
            align_angle_ok_frames_ = 0;
        }

        // 超时保护
        if((ros::Time::now() - stage_start_time_).toSec() > 4.5)
        {
            enterStage(GO_FORWARD, "ALIGN TIMEOUT");
        }
    }

    void handleGoForward(geometry_msgs::Twist& twist)
    {
        double speed = std::max(0.01, std::fabs(final_speed_));
        double forward_time = final_distance_ / speed;
        double elapsed = (ros::Time::now() - forward_start_time_).toSec();

        if(elapsed < forward_time)
        {
            twist.linear.x = final_speed_;
            twist.angular.z = 0.0;
        }
        else
        {
            enterStage(FINAL_STOP, "FINAL STOP");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
        }
    }

    // ---------- 高级白线提取 (HSV + 自适应阈值 + 形态学) ----------
    cv::Mat extractWhiteMaskAdvanced(const cv::Mat& roi)
    {
        cv::Mat hsv, gray, mask1, mask2, combined;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

        // HSV 白色范围
        cv::inRange(hsv, cv::Scalar(0, 0, 160), cv::Scalar(180, 30, 255), mask1);

        // 灰度自适应阈值
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(gray, mask2, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY, 11, -3);

        // 合并
        combined = mask1 | mask2;

        // 形态学去噪
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(combined, combined, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(combined, combined, 5);

        // 保留最大连通域（可选）
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(combined, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat clean = cv::Mat::zeros(combined.size(), CV_8UC1);
        double max_area = 0;
        int max_idx = -1;
        for(size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if(area > 200 && area > max_area)
            {
                max_area = area;
                max_idx = i;
            }
        }
        if(max_idx >= 0)
        {
            cv::drawContours(clean, contours, max_idx, cv::Scalar(255), -1);
        }

        return clean;
    }

    // ---------- 右线检测（使用 RANSAC 拟合） ----------
    LineInfo findRightLineAdvanced(const cv::Mat& mask)
    {
        LineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;

        // 提取右边界点
        std::vector<cv::Point> points;
        for(int y = static_cast<int>(h * 0.2); y < static_cast<int>(h * 0.92); y += 3)
        {
            const uchar* ptr = mask.ptr<uchar>(y);
            for(int x = w - 1; x >= 0; --x)
            {
                if(ptr[x] > 0)
                {
                    points.push_back(cv::Point(x, y));
                    break;
                }
            }
        }

        if(points.size() < 6)
        {
            info.found = false;
            // 使用预测值
            info.x = static_cast<int>(kalman_.getPosition());
            info.angle_deg = last_angle_;
            return info;
        }

        // RANSAC 直线拟合
        std::vector<cv::Point> inliers;
        cv::Vec4f line;
        if(fitLineRANSAC(points, inliers, line, 3.0, 100))
        {
            info.found = true;
            info.points = inliers;
            info.fit_line = line;

            double vx = line[0];
            double vy = line[1];
            info.angle_deg = rad2deg(std::atan2(vx, vy));

            // 计算底部区域平均 x 作为控制点
            double sum_x = 0.0;
            int count = 0;
            for(const auto& p : inliers)
            {
                if(p.y > h * 0.5)
                {
                    sum_x += p.x;
                    count++;
                }
            }
            if(count == 0)
            {
                for(const auto& p : inliers) { sum_x += p.x; count++; }
            }
            info.x = static_cast<int>(sum_x / count);

            // 保存最新角度
            last_angle_ = info.angle_deg;
        }
        else
        {
            info.found = false;
            info.x = static_cast<int>(kalman_.getPosition());
            info.angle_deg = last_angle_;
        }
        return info;
    }

    // RANSAC 直线拟合实现
    bool fitLineRANSAC(const std::vector<cv::Point>& points,
                       std::vector<cv::Point>& inliers,
                       cv::Vec4f& line,
                       double dist_threshold,
                       int max_iter)
    {
        if(points.size() < 2) return false;

        std::vector<cv::Point> best_inliers;
        cv::Vec4f best_line;
        int best_count = 0;

        cv::RNG rng;
        for(int iter = 0; iter < max_iter; ++iter)
        {
            // 随机选两个点
            int idx1 = rng.uniform(0, static_cast<int>(points.size()));
            int idx2 = rng.uniform(0, static_cast<int>(points.size()));
            if(idx1 == idx2) continue;

            cv::Point p1 = points[idx1];
            cv::Point p2 = points[idx2];

            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double len = std::sqrt(dx*dx + dy*dy);
            if(len < 1e-6) continue;

            double vx = dx / len;
            double vy = dy / len;
            cv::Vec4f candidate(vx, vy, p1.x, p1.y);

            // 统计内点
            std::vector<cv::Point> temp_inliers;
            for(const auto& p : points)
            {
                double dist = std::fabs((p.y - p1.y) * vx - (p.x - p1.x) * vy);
                if(dist < dist_threshold)
                {
                    temp_inliers.push_back(p);
                }
            }

            if(static_cast<int>(temp_inliers.size()) > best_count)
            {
                best_count = temp_inliers.size();
                best_inliers = temp_inliers;
                best_line = candidate;
            }
        }

        if(best_count < 6) return false;

        // 用所有内点重新拟合精确直线
        cv::Mat pts_mat(best_inliers.size(), 1, CV_32FC2);
        for(size_t i = 0; i < best_inliers.size(); ++i)
        {
            pts_mat.at<cv::Vec2f>(i,0) = cv::Vec2f(best_inliers[i].x, best_inliers[i].y);
        }
        cv::fitLine(pts_mat, line, cv::DIST_L2, 0, 0.01, 0.01);

        inliers = best_inliers;
        return true;
    }

    // ---------- 左线检测（用于辅助） ----------
    LineInfo findLeftLine(const cv::Mat& mask)
    {
        LineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;

        std::vector<cv::Point> points;
        for(int y = static_cast<int>(h * 0.2); y < static_cast<int>(h * 0.92); y += 4)
        {
            const uchar* ptr = mask.ptr<uchar>(y);
            for(int x = 0; x < w; ++x)
            {
                if(ptr[x] > 0)
                {
                    points.push_back(cv::Point(x, y));
                    break;
                }
            }
        }

        if(points.size() < 6)
        {
            info.found = false;
            return info;
        }

        cv::fitLine(points, info.fit_line, cv::DIST_L2, 0, 0.01, 0.01);
        double vx = info.fit_line[0];
        double vy = info.fit_line[1];
        info.found = true;
        info.angle_deg = rad2deg(std::atan2(vx, vy));

        double sum_x = 0.0;
        int count = 0;
        for(const auto& p : points)
        {
            if(p.y > h * 0.5) { sum_x += p.x; count++; }
        }
        info.x = static_cast<int>(sum_x / std::max(1, count));
        return info;
    }

    // ---------- 停车线检测（基本沿用，可适当增强） ----------
    StopLineInfo findStopLine(const cv::Mat& mask)
    {
        StopLineInfo info;
        const int h = mask.rows;
        const int w = mask.cols;

        cv::Mat bottom = mask(cv::Range(static_cast<int>(h * 0.65), h), cv::Range(0, w));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bottom.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double max_area = 0;
        int best_idx = -1;
        for(size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if(area < stop_line_min_area_) continue;
            cv::Rect rect = cv::boundingRect(contours[i]);
            if(rect.y < h * 0.65) continue;
            if(rect.width < rect.height * 3) continue;
            if(rect.height > stop_line_max_height_) continue;
            if(rect.width < stop_line_min_width_) continue;

            // 角度检查
            if(contours[i].size() >= 5)
            {
                cv::Vec4f line;
                cv::fitLine(contours[i], line, cv::DIST_L2, 0, 0.01, 0.01);
                double angle = rad2deg(std::atan2(line[1], line[0]));
                if(std::fabs(angle) > 25.0) continue;
            }

            if(area > max_area)
            {
                max_area = area;
                best_idx = i;
            }
        }

        if(best_idx >= 0)
        {
            cv::Rect rect = cv::boundingRect(contours[best_idx]);
            cv::Vec4f line;
            cv::fitLine(contours[best_idx], line, cv::DIST_L2, 0, 0.01, 0.01);
            info.found = true;
            info.rect = rect;
            info.angle_deg = rad2deg(std::atan2(line[1], line[0]));
            info.center_x = rect.x + rect.width / 2;
        }

        return info;
    }

    // ---------- 调试显示 ----------
    void showDebug(const cv::Mat& mask,
                   const LineInfo& right_line,
                   const LineInfo& left_line,
                   const StopLineInfo& stop_line,
                   const geometry_msgs::Twist& twist,
                   int predicted_x)
    {
        cv::Mat debug;
        cv::cvtColor(mask, debug, cv::COLOR_GRAY2BGR);

        // 绘制目标线
        int active_target = target_right_x_ - static_cast<int>(curve_offset_accum_);
        cv::line(debug, cv::Point(target_right_x_, 0), cv::Point(target_right_x_, mask.rows),
                 cv::Scalar(255,0,0), 2);
        cv::line(debug, cv::Point(active_target, 0), cv::Point(active_target, mask.rows),
                 cv::Scalar(0,255,255), 2);

        // 预测位置
        cv::circle(debug, cv::Point(predicted_x, mask.rows/2), 7, cv::Scalar(255,0,255), -1);

        if(right_line.found)
        {
            for(const auto& p : right_line.points)
                cv::circle(debug, p, 2, cv::Scalar(0,180,255), -1);
            drawFitLine(debug, right_line.fit_line, cv::Scalar(0,0,255));
            cv::circle(debug, cv::Point(right_line.x, mask.rows/2), 5, cv::Scalar(0,0,255), -1);
        }

        if(left_line.found)
        {
            for(const auto& p : left_line.points)
                cv::circle(debug, p, 2, cv::Scalar(255,180,0), -1);
            drawFitLine(debug, left_line.fit_line, cv::Scalar(0,255,0));
        }

        if(stop_line.found)
        {
            cv::rectangle(debug, stop_line.rect, cv::Scalar(0,255,0), 2);
            char buf[32];
            snprintf(buf, sizeof(buf), "Ang:%.1f", stop_line.angle_deg);
            cv::putText(debug, buf, cv::Point(stop_line.rect.x, stop_line.rect.y-10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
        }

        drawText(debug, 20, 30,  "Stage: " + stageName());
        drawText(debug, 20, 60,  "Target: " + std::to_string(active_target));
        drawText(debug, 20, 90,  "Err: " + std::to_string(filtered_pos_error_));
        drawText(debug, 20, 120, "Kalman x: " + std::to_string(static_cast<int>(kalman_.getPosition())));
        drawText(debug, 20, 150, "v: " + std::to_string(kalman_.getVelocity()));
        drawText(debug, 20, 180, "Offset: " + std::to_string(curve_offset_accum_));
        drawText(debug, 20, 210, "Cmd w: " + std::to_string(twist.angular.z));

        cv::imshow("right_follow", debug);
        cv::waitKey(1);
    }

    void drawFitLine(cv::Mat& img, const cv::Vec4f& line, const cv::Scalar& color)
    {
        float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
        if(std::fabs(vy) < 1e-5) return;
        int y1 = 0, y2 = img.rows - 1;
        int x1 = static_cast<int>(x0 + (y1 - y0) * vx / vy);
        int x2 = static_cast<int>(x0 + (y2 - y0) * vx / vy);
        cv::line(img, cv::Point(clampInt(x1,0,img.cols-1), y1),
                 cv::Point(clampInt(x2,0,img.cols-1), y2), color, 2);
    }

    void drawText(cv::Mat& img, int x, int y, const std::string& text)
    {
        cv::putText(img, text, cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,255,0), 2);
    }

    // ---------- 辅助函数 ----------
    void enterStage(Stage s, const char* msg)
    {
        stage_ = s;
        stage_start_time_ = ros::Time::now();
        if(s == GO_FORWARD) forward_start_time_ = ros::Time::now();
        ROS_INFO("%s", msg);
    }

    void resetPid()
    {
        last_pos_error_ = 0.0;
        filtered_pos_error_ = 0.0;
        stop_line_confirm_count_ = 0;
        align_angle_ok_frames_ = 0;
    }

    void stopCar()
    {
        geometry_msgs::Twist twist;
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
        cmd_pub_.publish(twist);
    }

    std::string stageName()
    {
        switch(stage_)
        {
        case STARTUP: return "STARTUP";
        case SEARCH_RIGHT_LINE: return "SEARCH";
        case FOLLOW_RIGHT_LINE: return "FOLLOW";
        case RIGHT_CURVE_PREDICT: return "CURVE_PREDICT";
        case STOP_LINE_FOUND: return "STOP_LINE";
        case ALIGN_WITH_RIGHT_LINE: return "ALIGN";
        case GO_FORWARD: return "FORWARD";
        case FINAL_STOP: return "FINAL_STOP";
        default: return "UNKNOWN";
        }
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