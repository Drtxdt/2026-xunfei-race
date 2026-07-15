#include <algorithm>
#include <cmath>
#include <iomanip>
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

namespace
{
constexpr int kImageRows = 480;
constexpr int kImageCols = 640;

int clampInt(int value, int low, int high)
{
  return std::max(low, std::min(high, value));
}

double clampDouble(double value, double low, double high)
{
  return std::max(low, std::min(high, value));
}

std::string boolText(bool value)
{
  return value ? "1" : "0";
}
}  // namespace

class StableRightTrackEndStopNode
{
public:
  StableRightTrackEndStopNode() : private_nh_("~")
  {
    loadParams();

    cmd_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_vel_topic_, 1);
    status_pub_ = nh_.advertise<std_msgs::String>(status_topic_, 1);
    debug_info_pub_ = nh_.advertise<std_msgs::String>(debug_info_topic_, 1);
    debug_image_pub_ = nh_.advertise<sensor_msgs::Image>(debug_image_topic_, 1);
    image_sub_ = nh_.subscribe(image_topic_, 1, &StableRightTrackEndStopNode::imageCallback, this);

    start_time_ = ros::Time::now();
    state_start_time_ = start_time_;
    last_detection_time_ = start_time_;
    state_ = auto_start_ ? State::StartupForward : State::Idle;
    setStatus(auto_start_ ? "stable_right_startup_forward" : "idle");

    ROS_INFO("stable_right_track_end_stop_node started: image=%s cmd_vel=%s debug=%s",
             image_topic_.c_str(), cmd_vel_topic_.c_str(), debug_image_topic_.c_str());
  }

private:
  enum class State
  {
    Idle,
    StartupForward,
    SearchRightLine,
    Follow,
    EndDetected,
    Forward50cm,
    FinalStop,
    Finish
  };

  struct FollowResult
  {
    bool found = false;
    int right_x = -1;
    double error = 0.0;
    double filtered_error = 0.0;
    double linear = 0.0;
    double angular = 0.0;
    int valid_rows = 0;
  };

  struct Segment
  {
    int left = 0;
    int right = 0;
    int width = 0;
    double center = 0.0;
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
    private_nh_.param<std::string>("status_topic", status_topic_, "/stable_right_track_end_stop/status");
    private_nh_.param<std::string>("debug_image_topic", debug_image_topic_, "/stable_right_track_end_stop/debug_image");
    private_nh_.param<std::string>("debug_info_topic", debug_info_topic_, "/stable_right_track_end_stop/debug_info");

    private_nh_.param("auto_start", auto_start_, true);
    private_nh_.param("startup_distance_m", startup_distance_m_, 1.0);
    private_nh_.param("startup_speed", startup_speed_, 0.45);

    private_nh_.param("target_right_x", target_right_x_, 200);
    private_nh_.param("base_speed", base_speed_, 0.32);
    private_nh_.param("curve_speed", curve_speed_, 0.28);
    private_nh_.param("search_speed", search_speed_, 0.10);
    private_nh_.param("search_angular_speed", search_angular_speed_, -0.26);
    private_nh_.param("lost_linear_speed", lost_linear_speed_, 0.14);
    private_nh_.param("lost_angular_speed", lost_angular_speed_, -0.22);
    private_nh_.param("kp", kp_, 0.0052);
    private_nh_.param("kd", kd_, 0.0018);
    private_nh_.param("error_alpha", error_alpha_, 0.22);
    private_nh_.param("curve_error_threshold", curve_error_threshold_, 38.0);
    private_nh_.param("curve_angular_gain", curve_angular_gain_, 1.18);
    private_nh_.param("max_angular_speed", max_angular_speed_, 0.55);
    private_nh_.param("max_tracking_speed", max_tracking_speed_, 0.24);
    private_nh_.param("min_tracking_speed", min_tracking_speed_, 0.08);
    private_nh_.param("angular_alpha", angular_alpha_, 0.70);
    private_nh_.param("lost_grace_time", lost_grace_time_, 0.30);
    private_nh_.param("lost_timeout", lost_timeout_, 0.90);

    private_nh_.param("roi_y_start_ratio", roi_y_start_ratio_, 0.60);
    private_nh_.param("white_s_max", white_s_max_, 45);
    private_nh_.param("white_v_min", white_v_min_, 200);
    private_nh_.param("gray_white_threshold", gray_white_threshold_, 175);
    private_nh_.param("min_white_mask_ratio", min_white_mask_ratio_, 0.002);
    private_nh_.param("max_white_mask_ratio", max_white_mask_ratio_, 0.32);
    private_nh_.param("adaptive_threshold_delta", adaptive_threshold_delta_, 28);
    private_nh_.param("morph_kernel_size", morph_kernel_size_, 5);
    private_nh_.param("min_component_area", min_component_area_, 260.0);
    private_nh_.param("robust_min_component_area", robust_min_component_area_, 80.0);
    private_nh_.param("max_component_area_ratio", max_component_area_ratio_, 0.70);
    private_nh_.param("min_line_width_px", min_line_width_px_, 5);
    private_nh_.param("max_line_segment_width_px", max_line_segment_width_px_, 90);
    private_nh_.param("min_segment_gap_px", min_segment_gap_px_, 10);
    private_nh_.param("max_target_jump_px", max_target_jump_px_, 90.0);

    private_nh_.param("end_enable_delay", end_enable_delay_, 3.0);
    private_nh_.param("end_roi_y_start_ratio", end_roi_y_start_ratio_, 0.87);
    private_nh_.param("end_min_width_ratio", end_min_width_ratio_, 0.45);
    private_nh_.param("end_stop_hold", end_stop_hold_, 1.0);
    private_nh_.param("end_forward_distance_m", end_forward_distance_m_, 0.65);
    private_nh_.param("end_forward_speed", end_forward_speed_, 0.17);

    // Match the shortened, precise final approach used by the left/right nodes.
    end_forward_distance_m_ = std::max(0.0, end_forward_distance_m_ - 0.05);

    if (morph_kernel_size_ % 2 == 0)
      ++morph_kernel_size_;
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
      ROS_WARN_THROTTLE(2.0, "stable_right_track_end_stop cv_bridge failed: %s", exc.what());
      return;
    }

    if (frame.empty())
      return;
    if (frame.cols != kImageCols || frame.rows != kImageRows)
      cv::resize(frame, frame, cv::Size(kImageCols, kImageRows), 0, 0, cv::INTER_AREA);

    const ros::Time now = ros::Time::now();
    cv::Mat roi = frame(cv::Range(clampInt(static_cast<int>(frame.rows * roi_y_start_ratio_), 0, frame.rows - 1),
                                  frame.rows),
                        cv::Range(0, frame.cols));
    cv::Mat mask = extractWhiteMask(roi);
    EndOfTrackResult end_result = detectEndOfTrack(mask, now);
    FollowResult follow;
    geometry_msgs::Twist cmd;

    switch (state_)
    {
      case State::Idle:
        setStatus("idle");
        publishStop();
        break;

      case State::StartupForward:
        if ((now - start_time_).toSec() <
            startup_distance_m_ / std::max(startup_speed_, 1e-6))
        {
          setStatus("stable_right_startup_forward");
          cmd.linear.x = startup_speed_;
          publishCmd(cmd);
        }
        else
        {
          ROS_INFO("stable right follow entering search mode");
          state_ = State::SearchRightLine;
          state_start_time_ = now;
        }
        break;

      case State::SearchRightLine:
        follow = computeFollow(mask, now);
        if (!follow.found)
        {
          setStatus("stable_right_search");
          cmd.linear.x = std::min(search_speed_, 0.05);
          cmd.angular.z = last_right_x_ >= 0
                              ? recoveryAngular()
                              : clampDouble(search_angular_speed_, -0.10, 0.10);
          publishCmd(cmd);
        }
        else
        {
          ROS_INFO("stable right line found");
          last_right_x_ = follow.right_x;
          last_detection_time_ = now;
          state_ = State::Follow;
          state_start_time_ = now;
          publishFollowCommand(follow, now);
        }
        break;

      case State::Follow:
        follow = computeFollow(mask, now);
        if (end_result.detected)
        {
          ROS_INFO("stable right track end detected! width_ratio=%.2f y_ratio=%.2f",
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
        setStatus("stable_right_end_detected");
        hardStop();
        if ((now - state_start_time_).toSec() >= end_stop_hold_)
        {
          state_ = State::Forward50cm;
          state_start_time_ = now;
          ROS_INFO("driving straight forward %.2f m at %.2f m/s after end detection",
                   end_forward_distance_m_, end_forward_speed_);
        }
        break;

      case State::Forward50cm:
      {
        const double forward_duration = end_forward_distance_m_ / std::max(end_forward_speed_, 1e-6);
        if ((now - state_start_time_).toSec() < forward_duration)
        {
          setStatus("stable_right_fast_forward");
          cmd.linear.x = end_forward_speed_;
          cmd.angular.z = 0.0;
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
        setStatus((now - state_start_time_).toSec() >= end_stop_hold_ ? "stable_right_finish" : "stable_right_final_stop");
        if ((now - state_start_time_).toSec() >= end_stop_hold_)
          state_ = State::Finish;
        break;

      case State::Finish:
        setStatus("stable_right_finish");
        hardStop();
        break;
    }

    publishDebug(frame, mask, follow, end_result, now);
    publishDebugInfo(follow, end_result, now);
    publishStatus();
  }

  cv::Mat extractWhiteMask(const cv::Mat& roi)
  {
    cv::Mat blur;
    cv::GaussianBlur(roi, blur, cv::Size(5, 5), 0);

    cv::Mat hsv;
    cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
    cv::Mat gray;
    cv::cvtColor(blur, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask = buildWhiteMask(hsv, gray, white_s_max_, white_v_min_, gray_white_threshold_, false);
    last_mask_ratio_ = static_cast<double>(cv::countNonZero(mask)) /
                       static_cast<double>(std::max(1, mask.rows * mask.cols));
    last_mask_mode_ = "normal";

    if (last_mask_ratio_ > max_white_mask_ratio_)
    {
      mask = buildWhiteMask(hsv, gray,
                            std::max(20, white_s_max_ - adaptive_threshold_delta_),
                            std::min(245, white_v_min_ + adaptive_threshold_delta_),
                            std::min(245, gray_white_threshold_ + adaptive_threshold_delta_), true);
      last_mask_mode_ = "strict";
    }
    else if (last_mask_ratio_ < min_white_mask_ratio_)
    {
      mask = buildWhiteMask(hsv, gray,
                            std::min(140, white_s_max_ + adaptive_threshold_delta_),
                            std::max(80, white_v_min_ - adaptive_threshold_delta_),
                            std::max(100, gray_white_threshold_ - adaptive_threshold_delta_), false);
      last_mask_mode_ = "loose";
    }

    last_mask_ratio_ = static_cast<double>(cv::countNonZero(mask)) /
                       static_cast<double>(std::max(1, mask.rows * mask.cols));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat clean_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    const double max_area = mask.rows * mask.cols * max_component_area_ratio_;
    const double effective_min_area = std::min(min_component_area_, robust_min_component_area_);
    for (const auto& contour : contours)
    {
      const double area = cv::contourArea(contour);
      if (area >= effective_min_area && area <= max_area)
        cv::drawContours(clean_mask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_kernel_size_, morph_kernel_size_));
    cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_CLOSE, kernel);
    cv::medianBlur(clean_mask, clean_mask, 5);
    return clean_mask;
  }

  cv::Mat buildWhiteMask(const cv::Mat& hsv, const cv::Mat& gray, int s_max, int v_min,
                         int gray_threshold, bool require_both) const
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

  FollowResult computeFollow(const cv::Mat& mask, const ros::Time& now)
  {
    FollowResult result;
    result.right_x = findRightLine(mask, result.valid_rows);
    result.found = result.right_x >= 0;
    if (!result.found)
      return result;

    result.error = static_cast<double>(target_right_x_ - result.right_x);
    filtered_error_ = (1.0 - error_alpha_) * filtered_error_ + error_alpha_ * result.error;
    const double now_sec = now.toSec();
    const double dt = last_pid_time_ > 0.0 ? std::max(1e-3, now_sec - last_pid_time_) : 0.0;
    last_pid_time_ = now_sec;
    const double d_error = dt > 0.0 ? clampDouble((filtered_error_ - last_error_) / dt, -100.0, 100.0) : 0.0;
    last_error_ = filtered_error_;

    double angular = kp_ * filtered_error_ + kd_ * d_error;
    double linear = std::min(base_speed_, max_tracking_speed_);
    if (std::fabs(filtered_error_) > curve_error_threshold_)
    {
      linear = std::min(curve_speed_, max_tracking_speed_ * 0.72);
      angular *= curve_angular_gain_;
    }
    if (std::fabs(filtered_error_) > curve_error_threshold_ * 2.0)
      linear = min_tracking_speed_;

    angular = clampDouble(angular, -max_angular_speed_, max_angular_speed_);
    filtered_angular_ = angular_alpha_ * filtered_angular_ + (1.0 - angular_alpha_) * angular;

    result.filtered_error = filtered_error_;
    result.linear = clampDouble(linear, min_tracking_speed_, max_tracking_speed_);
    result.angular = filtered_angular_;
    return result;
  }

  int findRightLine(const cv::Mat& mask, int& valid_rows) const
  {
    const int h = mask.rows;
    const std::vector<double> row_ratios = {0.92, 0.84, 0.76, 0.67, 0.58, 0.49, 0.40};
    std::vector<double> points;
    std::vector<double> weights;

    for (size_t i = 0; i < row_ratios.size(); ++i)
    {
      const int y = clampInt(static_cast<int>(h * row_ratios[i]), 0, h - 1);
      std::vector<Segment> segments = findSegments(mask.row(y));
      segments.erase(std::remove_if(segments.begin(), segments.end(), [this](const Segment& segment) {
                       return segment.width > max_line_segment_width_px_;
                     }), segments.end());
      if (segments.empty())
        continue;

      const Segment* selected = &segments.back();
      if (last_right_x_ >= 0)
      {
        const auto nearest = std::min_element(segments.begin(), segments.end(), [this](const Segment& a, const Segment& b) {
          return std::fabs(a.center - last_right_x_) < std::fabs(b.center - last_right_x_);
        });
        if (std::fabs(nearest->center - last_right_x_) <= max_target_jump_px_)
          selected = &(*nearest);
        else
        {
          continue;
        }
      }
      points.push_back(selected->center);
      weights.push_back(1.0 + (row_ratios.size() - i) * 0.12);
    }

    if (points.empty())
      return -1;

    std::vector<double> sorted = points;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    double weighted_x = 0.0;
    double weight_sum = 0.0;
    for (size_t i = 0; i < points.size(); ++i)
    {
      if (std::fabs(points[i] - median) > max_target_jump_px_)
        continue;
      weighted_x += points[i] * weights[i];
      weight_sum += weights[i];
      ++valid_rows;
    }

    if (valid_rows < 2 && !(valid_rows == 1 && last_right_x_ >= 0 &&
                            std::fabs(weighted_x / std::max(weight_sum, 1e-6) - last_right_x_) < 35.0))
      return -1;

    double right_x = weighted_x / std::max(weight_sum, 1e-6);
    if (last_right_x_ >= 0)
      right_x = clampDouble(right_x, last_right_x_ - max_target_jump_px_, last_right_x_ + max_target_jump_px_);
    return clampInt(static_cast<int>(std::round(right_x)), 0, mask.cols - 1);
  }

  EndOfTrackResult detectEndOfTrack(const cv::Mat& mask, const ros::Time& now) const
  {
    EndOfTrackResult result;
    if ((now - start_time_).toSec() < end_enable_delay_)
      return result;

    const int y0 = clampInt(static_cast<int>(mask.rows * end_roiYInMask()), 0, mask.rows - 1);
    const int bottom_height = mask.rows - y0;
    const int min_segment_width = static_cast<int>(mask.cols * end_min_width_ratio_);
    const int min_r = static_cast<int>(bottom_height * 0.45);

    for (int y = mask.rows - 1; y >= y0; --y)
    {
      const int r = y - y0;
      if (r <= min_r)
        continue;

      std::vector<Segment> segments = findSegments(mask.row(y));
      for (const Segment& segment : segments)
      {
        if (segment.width >= min_segment_width)
        {
          result.detected = true;
          result.best_width_ratio = static_cast<double>(segment.width) / static_cast<double>(mask.cols);
          result.best_y_ratio = (roi_y_start_ratio_ + (1.0 - roi_y_start_ratio_) *
                                  (static_cast<double>(y) / std::max(1.0, static_cast<double>(mask.rows))));
          return result;
        }
      }
    }
    return result;
  }

  double end_roiYInMask() const
  {
    if (end_roi_y_start_ratio_ <= roi_y_start_ratio_)
      return 0.0;
    return clampDouble((end_roi_y_start_ratio_ - roi_y_start_ratio_) / std::max(1e-6, 1.0 - roi_y_start_ratio_), 0.0, 1.0);
  }

  std::vector<Segment> findSegments(const cv::Mat& row) const
  {
    std::vector<Segment> segments;
    int start = -1;
    for (int x = 0; x < row.cols; ++x)
    {
      const bool active = row.at<uchar>(0, x) > 0;
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
    const int width = right - left + 1;
    if (width >= min_line_width_px_)
      segments.push_back(Segment{left, right, width, (left + right) / 2.0});
  }

  std::vector<Segment> mergeCloseSegments(const std::vector<Segment>& segments) const
  {
    if (segments.empty())
      return {};
    std::vector<Segment> merged(1, segments.front());
    for (size_t i = 1; i < segments.size(); ++i)
    {
      Segment& previous = merged.back();
      if (segments[i].left - previous.right <= min_segment_gap_px_)
      {
        previous.right = segments[i].right;
        previous.width = previous.right - previous.left + 1;
        previous.center = (previous.left + previous.right) / 2.0;
      }
      else
      {
        merged.push_back(segments[i]);
      }
    }
    return merged;
  }

  double recoveryAngular() const
  {
    if (std::fabs(last_valid_angular_) >= 0.03)
      return clampDouble(last_valid_angular_ * 0.65, -0.14, 0.14);
    return last_error_ >= 0.0 ? 0.08 : -0.08;
  }

  void publishFollowCommand(const FollowResult& follow, const ros::Time& now)
  {
    geometry_msgs::Twist cmd;
    if (!follow.found)
    {
      const double lost_age = (now - last_detection_time_).toSec();
      if (lost_age <= lost_grace_time_)
      {
        setStatus("stable_right_lost_hold");
        cmd.linear.x = std::min(lost_linear_speed_, 0.06);
        cmd.angular.z = clampDouble(last_valid_angular_ * 0.55, -0.12, 0.12);
      }
      else if (lost_age <= lost_timeout_)
      {
        setStatus("stable_right_recovering");
        cmd.linear.x = 0.03;
        cmd.angular.z = recoveryAngular();
      }
      else
      {
        setStatus("stable_right_lost_stop");
        hardStop();
        return;
      }
      publishCmd(cmd);
      return;
    }

    last_right_x_ = follow.right_x;
    last_detection_time_ = now;
    last_valid_angular_ = follow.angular;
    cmd.linear.x = follow.linear;
    cmd.angular.z = follow.angular;
    setStatus("stable_right_tracking");
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

  void publishDebug(const cv::Mat& frame, const cv::Mat& mask, const FollowResult& follow,
                    const EndOfTrackResult& end_result, const ros::Time& now)
  {
    cv::Mat debug = frame.clone();
    cv::Mat mask_bgr;
    cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
    cv::resize(mask_bgr, mask_bgr, cv::Size(kImageCols / 3, kImageRows / 3), 0, 0, cv::INTER_AREA);
    mask_bgr.copyTo(debug(cv::Rect(0, 0, mask_bgr.cols, mask_bgr.rows)));

    const int roi_y0 = clampInt(static_cast<int>(kImageRows * roi_y_start_ratio_), 0, kImageRows - 1);
    cv::rectangle(debug, cv::Rect(0, roi_y0, kImageCols, kImageRows - roi_y0), cv::Scalar(255, 200, 0), 1);
    cv::line(debug, cv::Point(target_right_x_, roi_y0), cv::Point(target_right_x_, kImageRows - 1), cv::Scalar(255, 0, 0), 2);

    if (follow.found)
      cv::circle(debug, cv::Point(follow.right_x, roi_y0 + (kImageRows - roi_y0) / 2), 6, cv::Scalar(0, 0, 255), -1);

    const int end_y0 = clampInt(static_cast<int>(kImageRows * end_roi_y_start_ratio_), 0, kImageRows - 1);
    cv::rectangle(debug, cv::Rect(0, end_y0, kImageCols, kImageRows - end_y0),
                  end_result.detected ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 220, 255), 1);
    if (end_result.detected)
      cv::putText(debug, "END DETECTED", cv::Point(kImageCols / 2 - 90, end_y0 - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    std::ostringstream line1;
    line1 << "state=" << status_ << " cmd=(" << std::fixed << std::setprecision(2)
          << last_linear_ << "," << last_angular_ << ") found=" << boolText(follow.found);
    cv::putText(debug, line1.str(), cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 255, 0), 2);

    std::ostringstream line2;
    line2 << "right_x=" << follow.right_x << " err=" << std::fixed << std::setprecision(1)
          << follow.filtered_error << " rows=" << follow.valid_rows
          << " mask=" << last_mask_mode_ << " end_w=" << std::setprecision(2) << end_result.best_width_ratio;
    cv::putText(debug, line2.str(), cv::Point(10, 215), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 220, 255), 2);

    try
    {
      sensor_msgs::ImagePtr out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug).toImageMsg();
      out->header.stamp = now;
      debug_image_pub_.publish(out);
    }
    catch (const cv_bridge::Exception& exc)
    {
      ROS_WARN_THROTTLE(2.0, "stable_right_track_end_stop debug publish failed: %s", exc.what());
    }
  }

  void publishDebugInfo(const FollowResult& follow, const EndOfTrackResult& end_result, const ros::Time& now)
  {
    std::ostringstream ss;
    ss << "status=" << status_
       << " elapsed=" << std::fixed << std::setprecision(2) << (now - start_time_).toSec()
       << " found=" << boolText(follow.found)
       << " right_x=" << follow.right_x
       << " error=" << follow.error
       << " filtered_error=" << follow.filtered_error
       << " valid_rows=" << follow.valid_rows
       << " mask_mode=" << last_mask_mode_
       << " mask_ratio=" << last_mask_ratio_
       << " cmd_linear=" << last_linear_
       << " cmd_angular=" << last_angular_
       << " end_detected=" << boolText(end_result.detected)
       << " end_width_ratio=" << end_result.best_width_ratio
       << " end_y_ratio=" << end_result.best_y_ratio;
    std_msgs::String msg;
    msg.data = ss.str();
    debug_info_pub_.publish(msg);
    ROS_INFO_THROTTLE(0.5, "stable_right_track_end_stop: %s", msg.data.c_str());
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
  double startup_distance_m_ = 1.0;
  double startup_speed_ = 0.45;

  int target_right_x_ = 200;
  double base_speed_ = 0.32;
  double curve_speed_ = 0.28;
  double search_speed_ = 0.10;
  double search_angular_speed_ = -0.26;
  double lost_linear_speed_ = 0.14;
  double lost_angular_speed_ = -0.22;
  double kp_ = 0.0052;
  double kd_ = 0.0018;
  double error_alpha_ = 0.22;
  double curve_error_threshold_ = 38.0;
  double curve_angular_gain_ = 1.18;
  double max_angular_speed_ = 0.55;
  double max_tracking_speed_ = 0.24;
  double min_tracking_speed_ = 0.08;
  double angular_alpha_ = 0.70;
  double lost_grace_time_ = 0.30;
  double lost_timeout_ = 0.90;

  double roi_y_start_ratio_ = 0.60;
  int white_s_max_ = 45;
  int white_v_min_ = 200;
  int gray_white_threshold_ = 175;
  double min_white_mask_ratio_ = 0.002;
  double max_white_mask_ratio_ = 0.32;
  int adaptive_threshold_delta_ = 28;
  int morph_kernel_size_ = 5;
  double min_component_area_ = 260.0;
  double robust_min_component_area_ = 80.0;
  double max_component_area_ratio_ = 0.70;
  int min_line_width_px_ = 5;
  int max_line_segment_width_px_ = 90;
  int min_segment_gap_px_ = 10;
  double max_target_jump_px_ = 90.0;

  double end_enable_delay_ = 3.0;
  double end_roi_y_start_ratio_ = 0.87;
  double end_min_width_ratio_ = 0.45;
  double end_stop_hold_ = 1.0;
  double end_forward_distance_m_ = 0.65;
  double end_forward_speed_ = 0.17;

  State state_ = State::Idle;
  ros::Time start_time_;
  ros::Time state_start_time_;
  ros::Time last_detection_time_;
  std::string status_ = "idle";

  double last_error_ = 0.0;
  double filtered_error_ = 0.0;
  double filtered_angular_ = 0.0;
  double last_pid_time_ = 0.0;
  double last_valid_angular_ = 0.0;
  int last_right_x_ = -1;
  double last_linear_ = 0.0;
  double last_angular_ = 0.0;
  double last_mask_ratio_ = 0.0;
  std::string last_mask_mode_ = "normal";
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "stable_right_track_end_stop_node");
  StableRightTrackEndStopNode node;
  ros::spin();
  return 0;
}