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

        // ===== 新增变量初始化 =====
        predicted_right_x_ = 160;
        lost_line_count_ = 0;
        last_angular_ = 0.0;
        last_line_time_ = ros::Time::now();

        ROS_INFO("=================================");
        ROS_INFO(" Stable Right Follow Competition (Enhanced) ");
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
    };

    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber image_sub_;

    int target_right_x_;

    double base_speed_;
    double curve_speed_;
    double search_speed_;
    double lost_line_speed_;
    double startup_speed_;

    double kp_pos_;
    double kd_pos_;
    double kp_angle_;

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

    double align_speed_;
    double align_angle_threshold_;
    double align_stop_time_;
    double desired_angle_deg_;

    double final_speed_;
    double final_distance_;

    bool show_debug_;

    double last_pos_error_;
    double filtered_pos_error_;
    int last_right_x_;

    Stage stage_;
    ros::Time start_time_;
    ros::Time stage_start_time_;
    ros::Time forward_start_time_;

    // ===== 新增变量 =====
    int predicted_right_x_;
    int lost_line_count_;
    double last_angular_;
    ros::Time last_line_time_;

    void loadParams(ros::NodeHandle& pnh)
    {
        pnh.param("target_right_x", target_right_x_, 145);

        pnh.param("base_speed", base_speed_, 0.34);
        pnh.param("curve_speed", curve_speed_, 0.27);
        pnh.param("search_speed", search_speed_, 0.12);
        pnh.param("lost_line_speed", lost_line_speed_, 0.14);
        pnh.param("startup_speed", startup_speed_, 0.45);

        pnh.param("kp_pos", kp_pos_, 0.0055);
        pnh.param("kd_pos", kd_pos_, 0.0018);
        pnh.param("kp_angle", kp_angle_, 0.40);

        pnh.param("curve_threshold", curve_threshold_, 35.0);
        pnh.param("curve_offset", curve_offset_, 15.0);
        pnh.param("curve_gain", curve_gain_, 1.2);

        pnh.param("max_angular", max_angular_, 0.55);
        pnh.param("error_filter_alpha", error_filter_alpha_, 0.22);

        pnh.param("startup_time", startup_time_, 2.8);

        pnh.param("cross_area_threshold", cross_area_threshold_, 48000);
        pnh.param("stop_line_min_width", stop_line_min_width_, 180);
        pnh.param("stop_line_max_height", stop_line_max_height_, 30);
        pnh.param("stop_line_min_area", stop_line_min_area_, 1200);

        pnh.param("align_speed", align_speed_, 0.18);
        pnh.param("align_angle_threshold", align_angle_threshold_, 2.0);
        pnh.param("align_stop_time", align_stop_time_, 0.2);
        pnh.param("desired_angle_deg", desired_angle_deg_, 0.0);

        pnh.param("final_speed", final_speed_, 0.20);
        pnh.param("final_distance", final_distance_, 0.70);

        pnh.param("show_debug", show_debug_, true);
    }

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

        // ===== ROI 扩大：从 0.60 → 0.45 =====
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
            handleAlign(twist, right_line);
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

        // ===== 角速度低通滤波 =====
        twist.angular.z = 0.7 * last_angular_ + 0.3 * twist.angular.z;
        last_angular_ = twist.angular.z;

        cmd_pub_.publish(twist);

        if(show_debug_)
        {
            showDebug(mask, right_line, stop_line, twist);
        }
    }

    void handleStartup(geometry_msgs::Twist& twist)
    {
        const double elapsed = (ros::Time::now() - start_time_).toSec();

        if(elapsed < startup_time_)
        {
            twist.linear.x = startup_speed_;
            twist.angular.z = 0.0;
            return;
        }

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
        if(stop_line.found)
        {
            resetPid();
            enterStage(STOP_LINE_FOUND, "STOP LINE DETECTED");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            return;
        }

        // ===== 长时间丢线 → 进入搜索 =====
        if(lost_line_count_ > 10)
        {
            enterStage(SEARCH_RIGHT_LINE, "LINE LOST, SEARCHING");
            twist.linear.x = search_speed_;
            twist.angular.z = -0.2;
            return;
        }

        // ===== 短时丢线：使用预测值 =====
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
            // 使用预测值继续控制
        }

        const bool in_curve =
            std::fabs(filtered_pos_error_) > curve_threshold_ ||
            std::fabs(current_angle - desired_angle_deg_) > align_angle_threshold_;

        const double target = target_right_x_ - (in_curve ? curve_offset_ : 0.0);

        const double pos_error = target - current_x;

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

        double linear_speed = in_curve ? curve_speed_ : base_speed_;

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

        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        if(elapsed >= align_stop_time_)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "ENTER ALIGN MODE");
        }

        // ===== 超时保护（若长时间未进入ALIGN） =====
        if(elapsed > 3.0)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "STOPLINE TIMEOUT");
        }
    }

    void handleAlign(
        geometry_msgs::Twist& twist,
        const LineInfo& right_line)
    {
        // ===== 关键：慢速前进，避免原地盲转 =====
        twist.linear.x = 0.05;

        double angle_error = 0.0;

        if(right_line.found)
        {
            angle_error = right_line.angle_deg - desired_angle_deg_;
            // 更新预测值
            predicted_right_x_ = 0.7 * predicted_right_x_ + 0.3 * right_line.x;
            lost_line_count_ = 0;
            last_line_time_ = ros::Time::now();
        }
        else
        {
            lost_line_count_++;
            // ===== 丢线：使用记忆方向 =====
            angle_error = (last_right_x_ > 160) ? -8.0 : 8.0;
        }

        // ===== 角度闭环 =====
        twist.angular.z = clamp(angle_error * 0.03, -0.18, 0.18);

        // ===== 成功条件 =====
        if(right_line.found &&
           std::fabs(angle_error) < align_angle_threshold_)
        {
            enterStage(GO_FORWARD, "ALIGN OK");
        }

        // ===== 防卡死超时 =====
        if((ros::Time::now() - stage_start_time_).toSec() > 3.5)
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

    cv::Mat extractWhiteMask(const cv::Mat& roi)
    {
        cv::Mat blur;
        cv::GaussianBlur(roi, blur, cv::Size(5, 5), 0);

        cv::Mat hsv;
        cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(
            hsv,
            cv::Scalar(0, 0, 200),
            cv::Scalar(180, 45, 255),
            mask);

        cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(mask, mask, 5);
        cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(
            mask,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        cv::Mat clean_mask = cv::Mat::zeros(mask.size(), CV_8UC1);

        for(const auto& cnt : contours)
        {
            if(cv::contourArea(cnt) > 260.0)
            {
                cv::drawContours(
                    clean_mask,
                    std::vector<std::vector<cv::Point> >{cnt},
                    -1,
                    cv::Scalar(255),
                    -1);
            }
        }

        return clean_mask;
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
            // ===== 丢线处理：使用预测值 =====
            info.found = false;
            info.x = predicted_right_x_;
            info.angle_deg = desired_angle_deg_;
            lost_line_count_++;
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

        // ===== 更新预测 =====
        if(info.found)
        {
            predicted_right_x_ = 0.7 * predicted_right_x_ + 0.3 * info.x;
            lost_line_count_ = 0;
            last_line_time_ = ros::Time::now();
        }

        return info;
    }

    StopLineInfo findStopLine(const cv::Mat& mask)
    {
        StopLineInfo info;

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(
            mask.clone(),
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        int best_y = -1;
        int total_white = 0;

        if(cross_area_threshold_ > 0)
        {
            total_white = cv::countNonZero(mask);
        }

        for(const auto& cnt : contours)
        {
            const cv::Rect rect = cv::boundingRect(cnt);
            const double area = cv::contourArea(cnt);

            // ===== 双条件校验（增加位置约束） =====
            const bool wide_enough = rect.width >= stop_line_min_width_;
            const bool flat_enough = rect.height <= stop_line_max_height_;
            const bool area_enough = area >= stop_line_min_area_;
            const bool low_enough = rect.y > mask.rows * 0.3;  // 避免误判上部

            if(wide_enough && flat_enough && area_enough && low_enough)
            {
                if(rect.y > best_y)
                {
                    best_y = rect.y;
                    info.rect = rect;
                    info.found = true;
                }
            }
        }

        if(!info.found &&
           cross_area_threshold_ > 0 &&
           total_white > cross_area_threshold_)
        {
            info.found = true;
            info.rect = cv::Rect(0, 0, mask.cols, mask.rows);
        }

        return info;
    }

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
        }

        drawText(debug, 20, 30, "stage: " + stageName(stage_));
        drawText(debug, 20, 60, format("target: %d", active_target));
        drawText(debug, 20, 90, format("err: %.2f", filtered_pos_error_));
        drawText(
            debug,
            20,
            120,
            format(
                "angle: %.2f",
                right_line.found ? right_line.angle_deg : 0.0));
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

    void enterStage(Stage stage, const char* message)
    {
        stage_ = stage;
        stage_start_time_ = ros::Time::now();
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