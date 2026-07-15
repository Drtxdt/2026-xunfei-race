#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <xmlrpcpp/XmlRpcValue.h>

namespace
{
constexpr int kImageRows = 480;
constexpr int kImageCols = 640;

double clampDouble(double value, double low, double high)
{
  return std::max(low, std::min(high, value));
}

int clampInt(int value, int low, int high)
{
  return std::max(low, std::min(high, value));
}

std::string boolText(bool value)
{
  return value ? "1" : "0";
}
}  // namespace

class RightTrackEndStopNode
{
public:
  RightTrackEndStopNode() : private_nh_("~")
  {
    loadParams();

    cmd_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_vel_topic_, 1);
    status_pub_ = nh_.advertise<std_msgs::String>(status_topic_, 1);
    debug_info_pub_ = nh_.advertise<std_msgs::String>(debug_info_topic_, 1);
    debug_image_pub_ = nh_.advertise<sensor_msgs::Image>(debug_image_topic_, 1);
    image_sub_ = nh_.subscribe(image_topic_, 1, &RightTrackEndStopNode::imageCallback, this);

    start_time_ = ros::Time::now();
    last_image_time_ = start_time_;
    state_ = auto_start_ ? State::StartupForward : State::Idle;
    setStatus(auto_start_ ? "right_startup_forward" : "idle");

    ROS_INFO("right_track_end_stop_node started: image=%s cmd_vel=%s debug=%s",
             image_topic_.c_str(), cmd_vel_topic_.c_str(), debug_image_topic_.c_str());
  }

private:
  enum class State
  {
    Idle,
    StartupForward,
    StartupTurn,
    StartupEnter,
    Follow,
    EndDetected,
    TurnRight,
    Forward50cm,
    FinalStop,
    Finish
  };

  struct FollowResult
  {
    bool found = false;
    double target_x = 0.0;
    double target_y = 0.0;
    double error = 0.0;
    double linear = 0.0;
    double angular = 0.0;
    std::vector<cv::Point2f> raw_line;
    std::vector<cv::Point2f> center_path;
  };

  struct Segment
  {
    int left = 0;
    int right = 0;
    double center = 0.0;
    int width = 0;
  };

  struct EndOfTrackResult
  {
    bool detected = false;
    double best_width_ratio = 0.0;
    double best_y_ratio = 0.0;
  };

  void loadParams()
  {
    private_nh_.param<std::string>("image_topic", image_topic_, "/usb_cam/image_raw");
    private_nh_.param<std::string>("cmd_vel_topic", cmd_vel_topic_, "/cmd_vel");
    private_nh_.param<std::string>("status_topic", status_topic_, "/right_track_end_stop/status");
    private_nh_.param<std::string>("debug_image_topic", debug_image_topic_, "/right_track_end_stop/debug_image");
    private_nh_.param<std::string>("debug_info_topic", debug_info_topic_, "/right_track_end_stop/debug_info");

    private_nh_.param("auto_start", auto_start_, true);
    private_nh_.param("startup_forward_duration", startup_forward_duration_, 2.2);
    private_nh_.param("startup_forward_speed", startup_forward_speed_, 0.16);
    private_nh_.param("startup_turn_duration", startup_turn_duration_, 3.85);
    private_nh_.param("startup_turn_angular_speed", startup_turn_angular_speed_, -0.34);
    private_nh_.param("startup_enter_duration", startup_enter_duration_, 1.8);
    private_nh_.param("startup_enter_speed", startup_enter_speed_, 0.10);

    private_nh_.param("white_s_max", white_s_max_, 85);
    private_nh_.param("white_v_min", white_v_min_, 155);
    private_nh_.param("gray_white_threshold", gray_white_threshold_, 175);
    private_nh_.param("morph_kernel_size", morph_kernel_size_, 5);
    private_nh_.param("min_component_area", min_component_area_, 80.0);
    private_nh_.param("max_component_area_ratio", max_component_area_ratio_, 0.70);
    private_nh_.param("min_white_mask_ratio", min_white_mask_ratio_, 0.002);
    private_nh_.param("max_white_mask_ratio", max_white_mask_ratio_, 0.32);
    private_nh_.param("adaptive_threshold_delta", adaptive_threshold_delta_, 28);

    loadDoubleList("right_scan_rows", right_scan_rows_, {0.95, 0.92, 0.88, 0.84, 0.80, 0.75, 0.70});
    private_nh_.param("right_offset_px", right_offset_px_, 200.0);
    private_nh_.param("right_scan_bottom_weight", right_scan_bottom_weight_, 1.8);
    private_nh_.param("min_line_width_px", min_line_width_px_, 5);
    private_nh_.param("max_line_segment_width_px", max_line_segment_width_px_, 90);
    private_nh_.param("min_segment_gap_px", min_segment_gap_px_, 10);
    private_nh_.param("max_target_jump_px", max_target_jump_px_, 160.0);
    private_nh_.param("kp", kp_, 0.0042);
    private_nh_.param("kd", kd_, 0.0014);

    private_nh_.param("base_speed", base_speed_, 0.22);
    private_nh_.param("min_speed", min_speed_, 0.10);
    private_nh_.param("fast_base_speed", fast_base_speed_, 0.25);
    private_nh_.param("fast_error_px", fast_error_px_, 60.0);
    private_nh_.param("medium_error_px", medium_error_px_, 130.0);
    private_nh_.param("hard_error_px", hard_error_px_, 210.0);
    private_nh_.param("medium_speed", medium_speed_, 0.17);
    private_nh_.param("max_angular_speed", max_angular_speed_, 0.65);
    private_nh_.param("angular_alpha", angular_alpha_, 0.55);
    private_nh_.param("error_alpha", error_alpha_, 0.65);
    private_nh_.param("lost_timeout", lost_timeout_, 0.8);
    private_nh_.param("lost_speed", lost_speed_, 0.06);
    private_nh_.param("lost_angular_speed", lost_angular_speed_, 0.0);
    private_nh_.param("stop_on_lost", stop_on_lost_, true);

    private_nh_.param("end_enable_delay", end_enable_delay_, 3.0);
    private_nh_.param("end_roi_y_start_ratio", end_roi_y_start_ratio_, 0.87);
    private_nh_.param("end_min_width_ratio", end_min_width_ratio_, 0.45);
    private_nh_.param("end_stop_hold", end_stop_hold_, 1.0);
    private_nh_.param("end_forward_distance_m", end_forward_distance_m_, 0.65);
    private_nh_.param("end_forward_speed", end_forward_speed_, 0.17);
    private_nh_.param("end_turn_left_angle_deg", end_turn_left_angle_deg_, 10.0);
    private_nh_.param("end_turn_left_angular_speed", end_turn_left_angular_speed_, 0.50);

    // Keep the existing launch/YAML values compatible while shortening only the
    // two requested straight segments by 5 cm.
    startup_forward_duration_ = std::max(
        0.0, startup_forward_duration_ - 0.05 / std::max(startup_forward_speed_, 1e-6));
    end_forward_distance_m_ = std::max(0.0, end_forward_distance_m_ - 0.05);

    if (morph_kernel_size_ % 2 == 0)
      ++morph_kernel_size_;
  }

  void loadDoubleList(const std::string& name, std::vector<double>& out, const std::vector<double>& fallback)
  {
    XmlRpc::XmlRpcValue values;
    if (!private_nh_.getParam(name, values) || values.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      out = fallback;
      return;
    }
    out.clear();
    for (int i = 0; i < values.size(); ++i)
    {
      if (values[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
        out.push_back(static_cast<int>(values[i]));
      else if (values[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        out.push_back(static_cast<double>(values[i]));
    }
    if (out.empty())
      out = fallback;
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    cv::Mat frame;
    try
    {
      frame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    }
    catch (const cv_bridge::Exception& exc)
    {
      ROS_WARN_THROTTLE(2.0, "right_track_end_stop cv_bridge failed: %s", exc.what());
      return;
    }

    if (frame.empty())
      return;
    if (frame.cols != kImageCols || frame.rows != kImageRows)
      cv::resize(frame, frame, cv::Size(kImageCols, kImageRows), 0, 0, cv::INTER_AREA);

    ros::Time now = ros::Time::now();
    last_image_time_ = now;

    cv::Mat mask = extractWhiteMask(frame);
    EndOfTrackResult end_result = detectEndOfTrack(mask, now);
    FollowResult follow = computeFollow(mask, now);
    geometry_msgs::Twist cmd;

    const double elapsed = (now - start_time_).toSec();

    switch (state_)
    {
      case State::Idle:
        setStatus("idle");
        publishStop();
        break;

      case State::StartupForward:
        if (elapsed < startup_forward_duration_)
        {
          setStatus("right_startup_forward");
          cmd.linear.x = startup_forward_speed_;
          publishCmd(cmd);
        }
        else
        {
          state_start_time_ = now;
          state_ = State::StartupTurn;
        }
        break;

      case State::StartupTurn:
        if ((now - state_start_time_).toSec() < startup_turn_duration_)
        {
          setStatus("right_startup_turn");
          cmd.angular.z = startup_turn_angular_speed_;
          publishCmd(cmd);
        }
        else
        {
          state_start_time_ = now;
          state_ = State::StartupEnter;
        }
        break;

      case State::StartupEnter:
        if ((now - state_start_time_).toSec() < startup_enter_duration_)
        {
          setStatus("right_startup_enter");
          cmd.linear.x = startup_enter_speed_;
          publishCmd(cmd);
        }
        else
        {
          state_start_time_ = now;
          state_ = State::Follow;
          last_detection_time_ = now;
          resetPid();
        }
        break;

      case State::Follow:
        if (end_result.detected)
        {
          ROS_INFO("right track end detected! width_ratio=%.2f y_ratio=%.2f",
                   end_result.best_width_ratio, end_result.best_y_ratio);
          state_ = State::EndDetected;
          state_start_time_ = now;
          hardStop();
        }
        else
        {
          publishFollowCommand(follow, now);
        }
        break;

      case State::EndDetected:
        setStatus("right_end_detected");
        hardStop();
        if ((now - state_start_time_).toSec() >= end_stop_hold_)
        {
          state_ = State::TurnRight;
          state_start_time_ = now;
          ROS_INFO("turning left %.1f deg at %.2f rad/s",
                   end_turn_left_angle_deg_, end_turn_left_angular_speed_);
        }
        break;

      case State::TurnRight:
      {
        const double turn_duration = (end_turn_left_angle_deg_ * M_PI / 180.0) /
                                     std::max(end_turn_left_angular_speed_, 1e-6);
        if ((now - state_start_time_).toSec() < turn_duration)
        {
          setStatus("right_turn_left_align");
          cmd.angular.z = end_turn_left_angular_speed_;
          publishCmd(cmd);
        }
        else
        {
          state_ = State::Forward50cm;
          state_start_time_ = now;
          hardStop();
          ROS_INFO("driving forward %.2f m at %.2f m/s",
                   end_forward_distance_m_, end_forward_speed_);
        }
        break;
      }

      case State::Forward50cm:
      {
        const double forward_duration = end_forward_distance_m_ / std::max(end_forward_speed_, 1e-6);
        if ((now - state_start_time_).toSec() < forward_duration)
        {
          setStatus("right_fast_forward");
          cmd.linear.x = end_forward_speed_;
          publishCmd(cmd);
        }
        else
        {
          state_ = State::FinalStop;
          state_start_time_ = now;
          hardStop();
        }
        break;
      }

      case State::FinalStop:
        hardStop();
        setStatus((now - state_start_time_).toSec() >= end_stop_hold_ ? "right_finish" : "right_final_stop");
        if ((now - state_start_time_).toSec() >= end_stop_hold_)
          state_ = State::Finish;
        break;

      case State::Finish:
        setStatus("right_finish");
        hardStop();
        break;
    }

    publishDebug(frame, mask, follow, end_result, now);
    publishDebugInfo(follow, end_result, now);
    publishStatus();
  }

  cv::Mat extractWhiteMask(const cv::Mat& frame)
  {
    cv::Mat blur;
    cv::GaussianBlur(frame, blur, cv::Size(5, 5), 0);

    cv::Mat hsv;
    cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
    cv::Mat gray;
    cv::cvtColor(blur, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask = buildWhiteMask(hsv, gray, white_s_max_, white_v_min_, gray_white_threshold_, false);
    last_mask_ratio_ = static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(std::max(1, mask.rows * mask.cols));
    last_mask_mode_ = "normal";

    if (last_mask_ratio_ > max_white_mask_ratio_)
    {
      const int strict_s = std::max(20, white_s_max_ - adaptive_threshold_delta_);
      const int strict_v = std::min(245, white_v_min_ + adaptive_threshold_delta_);
      const int strict_gray = std::min(245, gray_white_threshold_ + adaptive_threshold_delta_);
      mask = buildWhiteMask(hsv, gray, strict_s, strict_v, strict_gray, true);
      last_mask_ratio_ = static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(std::max(1, mask.rows * mask.cols));
      last_mask_mode_ = "strict";
    }
    else if (last_mask_ratio_ < min_white_mask_ratio_)
    {
      const int loose_s = std::min(140, white_s_max_ + adaptive_threshold_delta_);
      const int loose_v = std::max(80, white_v_min_ - adaptive_threshold_delta_);
      const int loose_gray = std::max(100, gray_white_threshold_ - adaptive_threshold_delta_);
      mask = buildWhiteMask(hsv, gray, loose_s, loose_v, loose_gray, false);
      last_mask_ratio_ = static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(std::max(1, mask.rows * mask.cols));
      last_mask_mode_ = "loose";
    }

    mask = filterComponents(mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_kernel_size_, morph_kernel_size_));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::medianBlur(mask, mask, 5);
    return mask;
  }

  cv::Mat buildWhiteMask(const cv::Mat& hsv, const cv::Mat& gray, int s_max, int v_min, int gray_threshold,
                         bool require_both) const
  {
    cv::Mat hsv_mask;
    cv::inRange(hsv, cv::Scalar(0, 0, v_min), cv::Scalar(179, s_max, 255), hsv_mask);
    cv::Mat gray_mask;
    cv::threshold(gray, gray_mask, gray_threshold, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    if (require_both)
      cv::bitwise_and(hsv_mask, gray_mask, mask);
    else
      cv::bitwise_or(hsv_mask, gray_mask, mask);
    return mask;
  }

  cv::Mat filterComponents(const cv::Mat& mask)
  {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8UC1);
    const double max_area = mask.rows * mask.cols * max_component_area_ratio_;
    for (const auto& contour : contours)
    {
      double area = cv::contourArea(contour);
      if (area >= min_component_area_ && area <= max_area)
        cv::drawContours(filtered, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
    }
    return filtered;
  }

  EndOfTrackResult detectEndOfTrack(const cv::Mat& mask, const ros::Time& now)
  {
    EndOfTrackResult result;
    const double elapsed = (now - start_time_).toSec();
    if (elapsed < end_enable_delay_)
      return result;

    const int bottom_y0 = clampInt(static_cast<int>(mask.rows * end_roi_y_start_ratio_), 0, mask.rows - 1);
    const int bottom_height = mask.rows - bottom_y0;
    const int min_segment_width = static_cast<int>(mask.cols * end_min_width_ratio_);
    const int min_r = static_cast<int>(bottom_height * 0.45);

    for (int y = mask.rows - 1; y >= bottom_y0; --y)
    {
      int r = y - bottom_y0;
      if (r <= min_r)
        continue;

      std::vector<Segment> segments = findSegments(mask.row(y));
      for (const auto& seg : segments)
      {
        if (seg.width >= min_segment_width)
        {
          result.detected = true;
          result.best_width_ratio = static_cast<double>(seg.width) / static_cast<double>(mask.cols);
          result.best_y_ratio = static_cast<double>(y) / static_cast<double>(mask.rows);
          return result;
        }
      }
    }
    return result;
  }

  FollowResult computeFollow(const cv::Mat& mask, const ros::Time& now)
  {
    FollowResult result = computeRightmostLineFollow(mask);

    if (result.found)
    {
      last_detection_time_ = now;
      last_error_ = result.error;
      last_target_x_ = result.target_x;
    }
    return result;
  }

  FollowResult computeRightmostLineFollow(const cv::Mat& mask)
  {
    FollowResult result;
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> weights;

    for (size_t i = 0; i < right_scan_rows_.size(); ++i)
    {
      double ratio = right_scan_rows_[i];
      int y = clampInt(static_cast<int>(mask.rows * ratio), 0, mask.rows - 1);
      std::vector<Segment> segments = findSegments(mask.row(y));
      segments.erase(
          std::remove_if(segments.begin(), segments.end(),
                         [this](const Segment& segment) {
                           return segment.width > max_line_segment_width_px_;
                         }),
          segments.end());
      if (segments.empty())
        continue;

      const Segment& rightmost = segments.back();
      double weight = 1.0 + (static_cast<double>(right_scan_rows_.size() - i) /
                             std::max(1.0, static_cast<double>(right_scan_rows_.size() - 1))) *
                                (right_scan_bottom_weight_ - 1.0);
      xs.push_back(rightmost.center);
      ys.push_back(y);
      weights.push_back(weight);
      result.raw_line.emplace_back(static_cast<float>(rightmost.center), static_cast<float>(y));
    }

    if (xs.empty())
      return result;

    double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    double right_x = 0.0;
    double target_y = 0.0;
    for (size_t i = 0; i < xs.size(); ++i)
    {
      right_x += xs[i] * weights[i];
      target_y += ys[i] * weights[i];
    }
    right_x /= std::max(weight_sum, 1e-6);
    target_y /= std::max(weight_sum, 1e-6);

    double target_x = right_x - right_offset_px_;
    if (has_last_target_)
      target_x = clampDouble(target_x, last_target_x_ - max_target_jump_px_, last_target_x_ + max_target_jump_px_);
    target_x = clampDouble(target_x, 0.0, static_cast<double>(kImageCols - 1));

    result.found = true;
    result.target_x = target_x;
    result.target_y = target_y;
    result.error = target_x - kImageCols / 2.0;
    result.center_path.emplace_back(static_cast<float>(target_x), static_cast<float>(target_y));
    has_last_target_ = true;
    return result;
  }

  std::vector<Segment> findSegments(const cv::Mat& row) const
  {
    std::vector<Segment> segments;
    int start = -1;
    for (int x = 0; x < row.cols; ++x)
    {
      bool active = row.at<uchar>(0, x) > 0;
      if (active && start < 0)
        start = x;
      else if (!active && start >= 0)
      {
        appendSegment(segments, start, x - 1);
        start = -1;
      }
    }
    if (start >= 0)
      appendSegment(segments, start, row.cols - 1);
    return mergeCloseSegments(segments);
  }

  void appendSegment(std::vector<Segment>& segments, int left, int right) const
  {
    int width = right - left + 1;
    if (width >= min_line_width_px_)
      segments.push_back(Segment{left, right, (left + right) / 2.0, width});
  }

  std::vector<Segment> mergeCloseSegments(const std::vector<Segment>& segments) const
  {
    if (segments.empty())
      return {};
    std::vector<Segment> merged;
    merged.push_back(segments.front());
    for (size_t i = 1; i < segments.size(); ++i)
    {
      Segment& previous = merged.back();
      const Segment& current = segments[i];
      if (current.left - previous.right <= min_segment_gap_px_)
      {
        previous.right = current.right;
        previous.width = previous.right - previous.left + 1;
        previous.center = (previous.left + previous.right) / 2.0;
      }
      else
      {
        merged.push_back(current);
      }
    }
    return merged;
  }

  void publishFollowCommand(const FollowResult& follow, const ros::Time& now,
                            double speed_limit = -1.0, const char* status = "right_tracking")
  {
    if (!follow.found)
    {
      const bool timed_out = (now - last_detection_time_).toSec() > lost_timeout_;
      if (stop_on_lost_ && timed_out)
      {
        setStatus("right_lost_stop");
        resetPid();
        hardStop();
        return;
      }

      setStatus("right_line_wait");
      geometry_msgs::Twist cmd;
      cmd.linear.x = lost_speed_;
      cmd.angular.z = lost_angular_speed_;
      publishCmd(cmd);
      return;
    }

    const double now_sec = now.toSec();
    const double dt = last_pid_time_ > 0.0 ? std::max(1e-3, now_sec - last_pid_time_) : 0.0;
    last_pid_time_ = now_sec;
    filtered_error_px_ = error_alpha_ * filtered_error_px_ + (1.0 - error_alpha_) * follow.error;
    const double derivative = dt > 0.0 ? (filtered_error_px_ - last_error_px_) / dt : 0.0;
    last_error_px_ = filtered_error_px_;

    double angular = -(kp_ * filtered_error_px_ + kd_ * derivative);
    angular = clampDouble(angular, -max_angular_speed_, max_angular_speed_);

    const double effective_base_speed = std::max(base_speed_, fast_base_speed_);
    const double error_abs = std::abs(filtered_error_px_);
    double linear = min_speed_;
    if (error_abs <= fast_error_px_)
      linear = effective_base_speed;
    else if (error_abs <= medium_error_px_)
    {
      const double t = (error_abs - fast_error_px_) / std::max(1e-6, medium_error_px_ - fast_error_px_);
      linear = effective_base_speed + t * (medium_speed_ - effective_base_speed);
    }
    else
    {
      const double t = std::min((error_abs - medium_error_px_) /
                                  std::max(1e-6, hard_error_px_ - medium_error_px_),
                              1.0);
      linear = medium_speed_ + t * (min_speed_ - medium_speed_);
    }
    linear = clampDouble(linear, min_speed_, effective_base_speed);
    if (speed_limit > 0.0)
      linear = std::min(linear, speed_limit);
    filtered_angular_ = angular_alpha_ * filtered_angular_ + (1.0 - angular_alpha_) * angular;

    geometry_msgs::Twist cmd;
    cmd.linear.x = linear;
    cmd.angular.z = filtered_angular_;
    setStatus(status);
    publishCmd(cmd);
  }

  void publishCmd(const geometry_msgs::Twist& cmd)
  {
    last_linear_ = cmd.linear.x;
    last_angular_ = cmd.angular.z;
    cmd_pub_.publish(cmd);
  }

  void publishStop()
  {
    geometry_msgs::Twist stop;
    publishCmd(stop);
  }

  void hardStop()
  {
    geometry_msgs::Twist stop;
    last_linear_ = 0.0;
    last_angular_ = 0.0;
    for (int i = 0; i < 4; ++i)
      cmd_pub_.publish(stop);
  }

  void resetPid()
  {
    filtered_angular_ = 0.0;
    filtered_error_px_ = 0.0;
    last_error_px_ = 0.0;
    last_pid_time_ = 0.0;
    has_last_target_ = false;
  }

  void publishDebug(const cv::Mat& frame, const cv::Mat& mask, const FollowResult& follow,
                    const EndOfTrackResult& end_result, const ros::Time& now)
  {
    cv::Mat debug = frame.clone();

    cv::Mat mask_small;
    cv::cvtColor(mask, mask_small, cv::COLOR_GRAY2BGR);
    cv::resize(mask_small, mask_small, cv::Size(kImageCols / 3, kImageRows / 3), 0, 0, cv::INTER_AREA);
    mask_small.copyTo(debug(cv::Rect(0, 0, mask_small.cols, mask_small.rows)));

    cv::line(debug, cv::Point(kImageCols / 2, 0), cv::Point(kImageCols / 2, kImageRows - 1), cv::Scalar(255, 0, 0), 1);

    for (const auto& point : follow.raw_line)
      cv::circle(debug, toPixel(point), 1, cv::Scalar(0, 255, 255), -1);

    for (size_t i = 1; i < follow.center_path.size(); i += 2)
      cv::line(debug, toPixel(follow.center_path[i - 1]), toPixel(follow.center_path[i]), cv::Scalar(255, 0, 255), 2);

    if (follow.found)
      cv::circle(debug, cv::Point(static_cast<int>(follow.target_x), static_cast<int>(follow.target_y)), 6,
                 cv::Scalar(0, 0, 255), -1);

    const int end_roi_y0 = clampInt(static_cast<int>(kImageRows * end_roi_y_start_ratio_), 0, kImageRows - 1);
    cv::Rect end_roi(0, end_roi_y0, kImageCols, kImageRows - end_roi_y0);
    cv::Scalar end_roi_color = end_result.detected ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 200, 0);
    cv::rectangle(debug, end_roi, end_roi_color, 1);

    if (end_result.best_width_ratio > 0.0)
    {
      int detect_y = static_cast<int>(end_result.best_y_ratio * kImageRows);
      cv::line(debug, cv::Point(0, detect_y), cv::Point(kImageCols - 1, detect_y),
               cv::Scalar(0, 255, 255), 1);
    }

    if (end_result.detected)
    {
      cv::putText(debug, "END DETECTED", cv::Point(kImageCols / 2 - 90, end_roi_y0 - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }

    std::ostringstream line1;
    line1 << "R state=" << status_ << " cmd=(" << std::fixed << std::setprecision(2) << last_linear_ << ","
          << last_angular_ << ") found=" << boolText(follow.found);
    cv::putText(debug, line1.str(), cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 255, 0), 2);

    std::ostringstream line2;
    line2 << "target=(" << std::fixed << std::setprecision(1) << follow.target_x << "," << follow.target_y
          << ") err=" << follow.error
          << " end_w=" << std::fixed << std::setprecision(2) << end_result.best_width_ratio
          << " end_y=" << end_result.best_y_ratio;
    cv::putText(debug, line2.str(), cv::Point(10, 215), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 220, 255), 2);

    try
    {
      sensor_msgs::ImagePtr out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug).toImageMsg();
      out->header.stamp = now;
      debug_image_pub_.publish(out);
    }
    catch (const cv_bridge::Exception& exc)
    {
      ROS_WARN_THROTTLE(2.0, "right_track_end_stop debug publish failed: %s", exc.what());
    }
  }

  void publishDebugInfo(const FollowResult& follow, const EndOfTrackResult& end_result, const ros::Time& now)
  {
    std::ostringstream ss;
    ss << "status=" << status_
       << " elapsed=" << std::fixed << std::setprecision(2) << (now - start_time_).toSec()
       << " mask_mode=" << last_mask_mode_
       << " mask_ratio=" << last_mask_ratio_
       << " found=" << boolText(follow.found)
       << " target_x=" << follow.target_x
       << " target_y=" << follow.target_y
       << " error=" << follow.error
       << " cmd_linear=" << last_linear_
       << " cmd_angular=" << last_angular_
       << " raw_points=" << follow.raw_line.size()
       << " end_detected=" << boolText(end_result.detected)
       << " end_width_ratio=" << end_result.best_width_ratio
       << " end_y_ratio=" << end_result.best_y_ratio;
    std_msgs::String msg;
    msg.data = ss.str();
    debug_info_pub_.publish(msg);
    ROS_INFO_THROTTLE(0.5, "right_track_end_stop: %s", msg.data.c_str());
  }

  void setStatus(const std::string& status)
  {
    status_ = status;
  }

  void publishStatus()
  {
    std_msgs::String msg;
    msg.data = status_;
    status_pub_.publish(msg);
  }

  cv::Point toPixel(const cv::Point2f& point) const
  {
    return cv::Point(clampInt(static_cast<int>(std::round(point.x)), 0, kImageCols - 1),
                     clampInt(static_cast<int>(std::round(point.y)), 0, kImageRows - 1));
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber image_sub_;
  ros::Publisher cmd_pub_;
  ros::Publisher status_pub_;
  ros::Publisher debug_info_pub_;
  ros::Publisher debug_image_pub_;

  std::string image_topic_;
  std::string cmd_vel_topic_;
  std::string status_topic_;
  std::string debug_image_topic_;
  std::string debug_info_topic_;

  bool auto_start_ = true;
  double startup_forward_duration_ = 2.2;
  double startup_forward_speed_ = 0.16;
  double startup_turn_duration_ = 3.85;
  double startup_turn_angular_speed_ = -0.34;
  double startup_enter_duration_ = 1.8;
  double startup_enter_speed_ = 0.10;

  int white_s_max_ = 85;
  int white_v_min_ = 155;
  int gray_white_threshold_ = 175;
  int morph_kernel_size_ = 5;
  double min_component_area_ = 80.0;
  double max_component_area_ratio_ = 0.70;
  double min_white_mask_ratio_ = 0.002;
  double max_white_mask_ratio_ = 0.32;
  int adaptive_threshold_delta_ = 28;

  std::vector<double> right_scan_rows_;
  double right_offset_px_ = 200.0;
  double right_scan_bottom_weight_ = 1.8;
  int min_line_width_px_ = 5;
  int max_line_segment_width_px_ = 90;
  int min_segment_gap_px_ = 10;
  double max_target_jump_px_ = 160.0;
  double kp_ = 0.0042;
  double kd_ = 0.0014;

  double base_speed_ = 0.22;
  double min_speed_ = 0.10;
  double fast_base_speed_ = 0.25;
  double fast_error_px_ = 60.0;
  double medium_error_px_ = 130.0;
  double hard_error_px_ = 210.0;
  double medium_speed_ = 0.17;
  double max_angular_speed_ = 0.65;
  double angular_alpha_ = 0.55;
  double error_alpha_ = 0.65;
  double lost_timeout_ = 0.8;
  double lost_speed_ = 0.06;
  double lost_angular_speed_ = 0.0;
  bool stop_on_lost_ = true;

  double end_enable_delay_ = 3.0;
  double end_roi_y_start_ratio_ = 0.87;
  double end_min_width_ratio_ = 0.45;
  double end_stop_hold_ = 1.0;
  double end_forward_distance_m_ = 0.65;
  double end_forward_speed_ = 0.17;
  double end_turn_left_angle_deg_ = 10.0;
  double end_turn_left_angular_speed_ = 0.50;

  State state_ = State::Idle;
  ros::Time start_time_;
  ros::Time state_start_time_;
  ros::Time last_detection_time_;
  ros::Time last_image_time_;
  std::string status_ = "idle";

  double filtered_angular_ = 0.0;
  double filtered_error_px_ = 0.0;
  double last_error_px_ = 0.0;
  double last_pid_time_ = 0.0;
  double last_error_ = 0.0;
  double last_target_x_ = 0.0;
  bool has_last_target_ = false;
  double last_linear_ = 0.0;
  double last_angular_ = 0.0;
  double last_mask_ratio_ = 0.0;
  std::string last_mask_mode_ = "normal";
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "right_track_end_stop_node");
  RightTrackEndStopNode node;
  ros::spin();
  return 0;
}