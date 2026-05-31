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

class RightLineFollowCppNode
{
public:
  RightLineFollowCppNode() : private_nh_("~")
  {
    loadParams();

    cmd_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_vel_topic_, 1);
    status_pub_ = nh_.advertise<std_msgs::String>(status_topic_, 1);
    debug_info_pub_ = nh_.advertise<std_msgs::String>(debug_info_topic_, 1);
    debug_image_pub_ = nh_.advertise<sensor_msgs::Image>(debug_image_topic_, 1);
    image_sub_ = nh_.subscribe(image_topic_, 1, &RightLineFollowCppNode::imageCallback, this);

    cv::Mat perspective = (cv::Mat_<double>(3, 3) << -2.897018, 2.446196, -388.368977,
                           -0.061836, 1.194630, -756.140464,
                           -0.000272, 0.008324, -4.335235);
    inv_perspective_ = perspective.inv();

    start_time_ = ros::Time::now();
    last_image_time_ = start_time_;
    state_ = auto_start_ ? State::StartupForward : State::Idle;
    setStatus(auto_start_ ? "startup_forward" : "idle");

    ROS_INFO("right_line_follow_cpp_node started: image=%s cmd_vel=%s debug=%s",
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
    FinishApproach,
    FinishForward,
    FinishStop,
    Finish
  };

  struct FollowResult
  {
    bool found = false;
    bool used_fallback = false;
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

  struct FinishResult
  {
    bool detected = false;
    cv::Rect box;
    double width_ratio = 0.0;
    double height_ratio = 0.0;
    double bottom_ratio = 0.0;
    double horizontal_presence = 0.0;
    double vertical_presence = 0.0;
  };

  void loadParams()
  {
    private_nh_.param<std::string>("image_topic", image_topic_, "/usb_cam/image_raw");
    private_nh_.param<std::string>("cmd_vel_topic", cmd_vel_topic_, "/cmd_vel");
    private_nh_.param<std::string>("status_topic", status_topic_, "/right_line_follow_cpp/status");
    private_nh_.param<std::string>("debug_image_topic", debug_image_topic_, "/right_line_follow_cpp/debug_image");
    private_nh_.param<std::string>("debug_info_topic", debug_info_topic_, "/right_line_follow_cpp/debug_info");

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

    private_nh_.param("begin_x", begin_x_, 25.0);
    private_nh_.param("begin_y", begin_y_, 400.0);
    private_nh_.param("seed_search_rows", seed_search_rows_, 6);
    private_nh_.param("seed_row_step_px", seed_row_step_px_, 25);
    private_nh_.param("edge_probe_px", edge_probe_px_, 6);
    private_nh_.param("edge_threshold", edge_threshold_, 90.0);
    private_nh_.param("trace_max_points", trace_max_points_, 300);
    private_nh_.param("trace_step_guard", trace_step_guard_, 4);
    private_nh_.param("line_blur_kernel", line_blur_kernel_, 7);
    private_nh_.param("sample_dist_m", sample_dist_m_, 0.01);
    private_nh_.param("aim_dist_m", aim_dist_m_, 0.12);
    private_nh_.param("forward_bias_m", forward_bias_m_, 0.20);
    private_nh_.param("pixel_per_meter", pixel_per_meter_, 500.0);
    private_nh_.param("lane_width_m", lane_width_m_, 0.42);
    private_nh_.param("right_center_bias_m", right_center_bias_m_, 0.035);

    private_nh_.param("fallback_roi_y_start_ratio", fallback_roi_y_start_ratio_, 0.48);
    private_nh_.param("right_offset_px", right_offset_px_, 180.0);
    private_nh_.param("right_scan_bottom_weight", right_scan_bottom_weight_, 1.8);
    private_nh_.param("min_line_width_px", min_line_width_px_, 5);
    private_nh_.param("max_line_segment_width_px", max_line_segment_width_px_, 90);
    private_nh_.param("min_segment_gap_px", min_segment_gap_px_, 10);
    private_nh_.param("max_target_jump_px", max_target_jump_px_, 120.0);
    private_nh_.param("kp", kp_, 0.0042);
    private_nh_.param("kd", kd_, 0.0014);
    loadDoubleList("right_scan_rows", right_scan_rows_, {0.95, 0.92, 0.88, 0.84, 0.80, 0.75, 0.70});

    private_nh_.param("base_speed", base_speed_, 0.16);
    private_nh_.param("min_speed", min_speed_, 0.07);
    private_nh_.param("right_fast_base_speed", right_fast_base_speed_, 0.22);
    private_nh_.param("right_fast_error_px", right_fast_error_px_, 45.0);
    private_nh_.param("right_medium_error_px", right_medium_error_px_, 130.0);
    private_nh_.param("right_hard_error_px", right_hard_error_px_, 210.0);
    private_nh_.param("right_medium_speed", right_medium_speed_, 0.12);
    private_nh_.param("curve_speed_error_scale", curve_speed_error_scale_, 0.18);
    private_nh_.param("max_angular_speed", max_angular_speed_, 0.55);
    private_nh_.param("angular_alpha", angular_alpha_, 0.65);
    private_nh_.param("lost_timeout", lost_timeout_, 0.8);
    private_nh_.param("lost_speed", lost_speed_, 0.06);
    private_nh_.param("lost_angular_speed", lost_angular_speed_, 0.0);
    private_nh_.param("stop_on_lost", stop_on_lost_, true);

    private_nh_.param("finish_enable_delay", finish_enable_delay_, 18.0);
    private_nh_.param("finish_min_width_ratio", finish_min_width_ratio_, 0.48);
    private_nh_.param("finish_max_width_ratio", finish_max_width_ratio_, 0.92);
    private_nh_.param("finish_min_height_ratio", finish_min_height_ratio_, 0.18);
    private_nh_.param("finish_min_bottom_y_ratio", finish_min_bottom_y_ratio_, 0.70);
    private_nh_.param("finish_roi_y_start_ratio", finish_roi_y_start_ratio_, 0.52);
    private_nh_.param("finish_min_horizontal_presence", finish_min_horizontal_presence_, 0.55);
    private_nh_.param("finish_min_vertical_presence", finish_min_vertical_presence_, 0.42);
    private_nh_.param("finish_confirm_frames", finish_confirm_frames_, 4);
    private_nh_.param("finish_release_frames", finish_release_frames_, 2);
    private_nh_.param("finish_stop_on_box_count", finish_stop_on_box_count_, 2);
    private_nh_.param("finish_box_cooldown_sec", finish_box_cooldown_sec_, 2.0);
    private_nh_.param("finish_approach_speed", finish_approach_speed_, 0.06);
    private_nh_.param("finish_centering_enabled", finish_centering_enabled_, true);
    private_nh_.param("finish_center_target_bottom_ratio", finish_center_target_bottom_ratio_, 0.90);
    private_nh_.param("finish_center_max_duration", finish_center_max_duration_, 1.20);
    private_nh_.param("finish_center_angular_kp", finish_center_angular_kp_, 0.0020);
    private_nh_.param("finish_center_max_angular", finish_center_max_angular_, 0.16);
    private_nh_.param("finish_forward_duration", finish_forward_duration_, 1.65);
    private_nh_.param("finish_forward_speed", finish_forward_speed_, 0.08);
    private_nh_.param("finish_stop_hold", finish_stop_hold_, 2.0);

    if (morph_kernel_size_ % 2 == 0)
      ++morph_kernel_size_;
    if (line_blur_kernel_ % 2 == 0)
      ++line_blur_kernel_;
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
      ROS_WARN_THROTTLE(2.0, "right_line_follow_cpp cv_bridge failed: %s", exc.what());
      return;
    }

    if (frame.empty())
      return;
    if (frame.cols != kImageCols || frame.rows != kImageRows)
      cv::resize(frame, frame, cv::Size(kImageCols, kImageRows), 0, 0, cv::INTER_AREA);

    ros::Time now = ros::Time::now();
    last_image_time_ = now;

    cv::Mat mask = extractWhiteMask(frame);
    FinishResult finish = detectFinishBox(mask, now);
    FollowResult follow = computeFollow(frame, mask, now);
    geometry_msgs::Twist cmd;

    const double elapsed = (now - start_time_).toSec();
    if (state_ == State::Idle)
    {
      setStatus("idle");
      publishStop();
    }
    else if (state_ == State::StartupForward)
    {
      if (elapsed < startup_forward_duration_)
      {
        setStatus("startup_forward");
        cmd.linear.x = startup_forward_speed_;
        publishCmd(cmd);
      }
      else
      {
        state_start_time_ = now;
        state_ = State::StartupTurn;
      }
    }
    else if (state_ == State::StartupTurn)
    {
      if ((now - state_start_time_).toSec() < startup_turn_duration_)
      {
        setStatus("startup_turn_right");
        cmd.angular.z = startup_turn_angular_speed_;
        publishCmd(cmd);
      }
      else
      {
        state_start_time_ = now;
        state_ = State::StartupEnter;
      }
    }
    else if (state_ == State::StartupEnter)
    {
      if ((now - state_start_time_).toSec() < startup_enter_duration_)
      {
        setStatus("startup_enter_right_lane");
        cmd.linear.x = startup_enter_speed_;
        publishCmd(cmd);
      }
      else
      {
        state_start_time_ = now;
        state_ = State::Follow;
        last_detection_time_ = now;
        resetRightLinePid();
      }
    }
    else if (state_ == State::Follow)
    {
      const bool finish_event = updateFinishEncounter(finish, now);
      if (finish_event)
      {
        ROS_INFO("finish box encounter %d/%d", finish_box_count_, finish_stop_on_box_count_);
      }

      if (finish_event && finish_box_count_ >= finish_stop_on_box_count_)
      {
        state_start_time_ = now;
        if (finish_centering_enabled_)
        {
          state_ = State::FinishApproach;
          setStatus("finish_centering");
          publishFinishCenteringCommand(finish, now);
        }
        else
        {
          state_ = State::FinishStop;
          setStatus("finish_second_box_stop");
          hardStop();
        }
      }
      else
      {
        publishFollowCommand(follow, now);
      }
    }
    else if (state_ == State::FinishApproach)
    {
      setStatus("finish_centering");
      const double elapsed_centering = (now - state_start_time_).toSec();
      const bool target_reached = finish.detected && finish.bottom_ratio >= finish_center_target_bottom_ratio_;
      const bool timed_out = elapsed_centering >= finish_center_max_duration_;
      if (target_reached || timed_out)
      {
        state_ = State::FinishStop;
        state_start_time_ = now;
        hardStop();
      }
      else
      {
        publishFinishCenteringCommand(finish, now);
      }
    }
    else if (state_ == State::FinishForward)
    {
      setStatus("finish_forward");
      if ((now - state_start_time_).toSec() < finish_forward_duration_)
      {
        cmd.linear.x = finish_forward_speed_;
        publishCmd(cmd);
      }
      else
      {
        state_ = State::FinishStop;
        state_start_time_ = now;
        hardStop();
      }
    }
    else if (state_ == State::FinishStop)
    {
      hardStop();
      setStatus((now - state_start_time_).toSec() >= finish_stop_hold_ ? "finish" : "finish_stop");
      if ((now - state_start_time_).toSec() >= finish_stop_hold_)
        state_ = State::Finish;
    }
    else if (state_ == State::Finish)
    {
      setStatus("finish");
      hardStop();
    }

    publishDebug(frame, mask, follow, finish, now);
    publishDebugInfo(follow, finish, now);
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

  FollowResult computeFollow(const cv::Mat& frame, const cv::Mat& mask, const ros::Time& now)
  {
    (void)frame;
    FollowResult result;
    result = computeRightmostLineFollow(mask);

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
    {
      target_x = clampDouble(target_x, last_target_x_ - max_target_jump_px_, last_target_x_ + max_target_jump_px_);
    }
    target_x = clampDouble(target_x, 0.0, static_cast<double>(kImageCols - 1));

    result.found = true;
    result.used_fallback = false;
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
      {
        start = x;
      }
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

  std::vector<cv::Point> traceRightLine(const cv::Mat& mask)
  {
    cv::Point seed;
    if (!findRightSeed(mask, seed))
      return {};

    static const int front[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    static const int front_right[4][2] = {{1, -1}, {1, 1}, {-1, 1}, {-1, -1}};
    int x = seed.x;
    int y = seed.y;
    int direction = 0;
    int turn = 0;
    std::vector<cv::Point> pts;
    pts.reserve(trace_max_points_);

    while (pts.size() < static_cast<size_t>(trace_max_points_) && x > 1 && x < mask.cols - 2 &&
           y > 1 && y < mask.rows - 2 && turn < trace_step_guard_)
    {
      int fx = front[direction][0];
      int fy = front[direction][1];
      int sx = front_right[direction][0];
      int sy = front_right[direction][1];
      if (pixel(mask, x + fx, y + fy) == 0)
      {
        direction = (direction + 3) % 4;
        ++turn;
      }
      else if (pixel(mask, x + sx, y + sy) == 0)
      {
        x += fx;
        y += fy;
        pts.emplace_back(x, y);
        turn = 0;
      }
      else
      {
        x += sx;
        y += sy;
        direction = (direction + 1) % 4;
        pts.emplace_back(x, y);
        turn = 0;
      }
    }
    return pts;
  }

  bool findRightSeed(const cv::Mat& mask, cv::Point& seed)
  {
    const int half = 3;
    const int d = std::max(2, edge_probe_px_);
    for (int row = 0; row < seed_search_rows_; ++row)
    {
      int y = clampInt(static_cast<int>(std::round(begin_y_)) - row * seed_row_step_px_, half + 1, mask.rows - half - 2);
      for (int x = clampInt(static_cast<int>(mask.cols / 2 + begin_x_), d + 1, mask.cols - d - 2);
           x < mask.cols - d - 1; ++x)
      {
        double local = 0.0;
        for (int dy = -half; dy <= half; ++dy)
          local += static_cast<double>(pixel(mask, x - d, y + dy)) - static_cast<double>(pixel(mask, x + d, y + dy));
        local /= static_cast<double>(2 * half + 1);
        if (local >= edge_threshold_)
        {
          seed = cv::Point(clampInt(x + d, 0, mask.cols - 1), y);
          return true;
        }
      }
    }
    return false;
  }

  FollowResult computeFallback(const cv::Mat& mask)
  {
    FollowResult result;
    std::vector<double> right_edges;
    for (double ratio : fallback_rows_)
    {
      int y = clampInt(static_cast<int>(mask.rows * ratio), 0, mask.rows - 1);
      const uchar* row = mask.ptr<uchar>(y);
      int right = -1;
      for (int x = mask.cols - 1; x >= 0; --x)
      {
        if (row[x] > 0)
        {
          right = x;
          break;
        }
      }
      if (right >= 0)
        right_edges.push_back(right);
    }

    if (right_edges.empty())
      return result;

    double right_x = std::accumulate(right_edges.begin(), right_edges.end(), 0.0) / right_edges.size();
    result.found = true;
    result.used_fallback = true;
    result.target_x = right_x - fallback_right_offset_px_;
    result.target_y = mask.rows * 0.72;
    double dx = result.target_x - kImageCols / 2.0;
    double dy = kImageRows - result.target_y + forward_bias_m_ * pixel_per_meter_;
    result.error = -std::atan2(dx, std::max(dy, 1e-3));
    result.center_path.push_back(cv::Point2f(static_cast<float>(result.target_x), static_cast<float>(result.target_y)));
    return result;
  }

  cv::Point2f mapPoint(const cv::Point& point) const
  {
    double x = point.x;
    double y = point.y;
    double denom = inv_perspective_.at<double>(2, 0) * x + inv_perspective_.at<double>(2, 1) * y +
                   inv_perspective_.at<double>(2, 2);
    if (std::abs(denom) < 1e-6)
      return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    double mx = (inv_perspective_.at<double>(0, 0) * x + inv_perspective_.at<double>(0, 1) * y +
                 inv_perspective_.at<double>(0, 2)) /
                denom;
    double my = (inv_perspective_.at<double>(1, 0) * x + inv_perspective_.at<double>(1, 1) * y +
                 inv_perspective_.at<double>(1, 2)) /
                denom;
    return cv::Point2f(static_cast<float>(mx), static_cast<float>(my));
  }

  std::vector<cv::Point2f> blurPoints(const std::vector<cv::Point2f>& pts, int kernel) const
  {
    if (pts.empty())
      return {};
    int half = std::max(1, kernel / 2);
    double denom = (2 * half + 2) * (half + 1) / 2.0;
    std::vector<cv::Point2f> out;
    out.reserve(pts.size());
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
    {
      double sx = 0.0;
      double sy = 0.0;
      for (int j = -half; j <= half; ++j)
      {
        int idx = clampInt(i + j, 0, static_cast<int>(pts.size()) - 1);
        double weight = half + 1 - std::abs(j);
        sx += pts[idx].x * weight;
        sy += pts[idx].y * weight;
      }
      out.emplace_back(static_cast<float>(sx / denom), static_cast<float>(sy / denom));
    }
    return out;
  }

  std::vector<cv::Point2f> resamplePoints(const std::vector<cv::Point2f>& pts, double dist, int max_len) const
  {
    std::vector<cv::Point2f> out;
    if (pts.size() < 2 || dist <= 1e-6)
      return out;
    out.reserve(std::min(max_len, static_cast<int>(pts.size())));
    double remain = 0.0;
    for (size_t i = 0; i + 1 < pts.size() && static_cast<int>(out.size()) < max_len; ++i)
    {
      cv::Point2f p = pts[i];
      cv::Point2f delta = pts[i + 1] - p;
      double len = std::hypot(delta.x, delta.y);
      if (len <= 1e-6)
        continue;
      cv::Point2f unit(delta.x / len, delta.y / len);
      while (remain < len && static_cast<int>(out.size()) < max_len)
      {
        p += unit * static_cast<float>(remain);
        out.push_back(p);
        len -= remain;
        remain = dist;
      }
      remain -= len;
    }
    return out;
  }

  std::vector<cv::Point2f> trackRightLine(const std::vector<cv::Point2f>& pts, int approx_num, double dist) const
  {
    std::vector<cv::Point2f> out;
    if (pts.size() < 3)
      return out;
    out.reserve(pts.size());
    out.emplace_back(kImageCols / 2.0f, kImageRows + 50.0f);
    approx_num = std::max(1, approx_num);
    for (int i = 1; i < static_cast<int>(pts.size()); ++i)
    {
      int left = clampInt(i - approx_num, 0, static_cast<int>(pts.size()) - 1);
      int right = clampInt(i + approx_num, 0, static_cast<int>(pts.size()) - 1);
      double dx = pts[right].x - pts[left].x;
      double dy = pts[right].y - pts[left].y;
      double norm = std::hypot(dx, dy);
      if (norm <= 1e-6)
        continue;
      dx /= norm;
      dy /= norm;
      out.emplace_back(static_cast<float>(pts[i].x + dy * dist), static_cast<float>(pts[i].y - dx * dist));
    }
    return out;
  }

  FinishResult detectFinishBox(const cv::Mat& mask, const ros::Time& now)
  {
    FinishResult result;
    if ((now - start_time_).toSec() < finish_enable_delay_)
      return result;

    int roi_y0 = clampInt(static_cast<int>(mask.rows * finish_roi_y_start_ratio_), 0, mask.rows - 2);
    cv::Mat finish_roi = mask(cv::Rect(0, roi_y0, mask.cols, mask.rows - roi_y0));

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
    cv::Mat closed;
    cv::morphologyEx(finish_roi, closed, cv::MORPH_CLOSE, kernel);
    cv::dilate(closed, closed, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double best_score = -1.0;
    for (const auto& contour : contours)
    {
      cv::Rect box = cv::boundingRect(contour);
      cv::Rect full_box(box.x, box.y + roi_y0, box.width, box.height);
      double width_ratio = box.width / static_cast<double>(mask.cols);
      double height_ratio = box.height / static_cast<double>(mask.rows);
      double bottom_ratio = (full_box.y + full_box.height) / static_cast<double>(mask.rows);
      if (width_ratio < finish_min_width_ratio_ || width_ratio > finish_max_width_ratio_)
        continue;
      if (height_ratio < finish_min_height_ratio_ || bottom_ratio < finish_min_bottom_y_ratio_)
        continue;

      cv::Mat roi = closed(box);
      int band_y = std::max(3, static_cast<int>(std::round(box.height * 0.14)));
      int band_x = std::max(3, static_cast<int>(std::round(box.width * 0.14)));
      double top = horizontalPresence(roi(cv::Rect(0, 0, roi.cols, std::min(band_y, roi.rows))));
      double bottom = horizontalPresence(roi(cv::Rect(0, std::max(0, roi.rows - band_y), roi.cols, std::min(band_y, roi.rows))));
      double left = verticalPresence(roi(cv::Rect(0, 0, std::min(band_x, roi.cols), roi.rows)));
      double right = verticalPresence(roi(cv::Rect(std::max(0, roi.cols - band_x), 0, std::min(band_x, roi.cols), roi.rows)));
      double horizontal = std::max(top, bottom);
      double vertical = std::max(left, right);
      if (horizontal < finish_min_horizontal_presence_ || vertical < finish_min_vertical_presence_)
        continue;

      double score = width_ratio + height_ratio + bottom_ratio + horizontal + vertical;
      if (score > best_score)
      {
        best_score = score;
        result.detected = true;
        result.box = full_box;
        result.width_ratio = width_ratio;
        result.height_ratio = height_ratio;
        result.bottom_ratio = bottom_ratio;
        result.horizontal_presence = horizontal;
        result.vertical_presence = vertical;
      }
    }
    return result;
  }

  bool updateFinishEncounter(const FinishResult& finish, const ros::Time& now)
  {
    if (now < finish_box_cooldown_until_)
      return false;

    if (finish.detected)
    {
      ++finish_frames_;
      finish_lost_frames_ = 0;
    }
    else
    {
      ++finish_lost_frames_;
      if (finish_lost_frames_ >= finish_release_frames_)
      {
        finish_frames_ = 0;
        finish_lost_frames_ = 0;
        finish_box_armed_ = true;
      }
      return false;
    }

    if (finish_box_armed_ && finish_frames_ >= finish_confirm_frames_)
    {
      ++finish_box_count_;
      finish_box_armed_ = false;
      finish_frames_ = 0;
      finish_lost_frames_ = 0;
      finish_box_cooldown_until_ = now + ros::Duration(finish_box_cooldown_sec_);
      return true;
    }
    return false;
  }

  void publishFinishCenteringCommand(const FinishResult& finish, const ros::Time& now)
  {
    (void)now;
    geometry_msgs::Twist cmd;
    cmd.linear.x = finish_approach_speed_;
    if (finish.detected && finish.box.area() > 0)
    {
      const double box_center_x = finish.box.x + finish.box.width * 0.5;
      const double center_error = box_center_x - kImageCols * 0.5;
      cmd.angular.z = clampDouble(-finish_center_angular_kp_ * center_error,
                                  -finish_center_max_angular_, finish_center_max_angular_);
      last_finish_center_error_px_ = center_error;
    }
    else
    {
      cmd.angular.z = 0.0;
      last_finish_center_error_px_ = 0.0;
    }
    publishCmd(cmd);
  }

  double horizontalPresence(const cv::Mat& image) const
  {
    if (image.empty())
      return 0.0;
    int left = image.cols;
    int right = -1;
    for (int y = 0; y < image.rows; ++y)
    {
      const uchar* row = image.ptr<uchar>(y);
      for (int x = 0; x < image.cols; ++x)
      {
        if (row[x] > 0)
        {
          left = std::min(left, x);
          right = std::max(right, x);
        }
      }
    }
    if (right < left)
      return 0.0;
    return (right - left + 1) / static_cast<double>(image.cols);
  }

  double verticalPresence(const cv::Mat& image) const
  {
    if (image.empty())
      return 0.0;
    int top = image.rows;
    int bottom = -1;
    for (int y = 0; y < image.rows; ++y)
    {
      const uchar* row = image.ptr<uchar>(y);
      for (int x = 0; x < image.cols; ++x)
      {
        if (row[x] > 0)
        {
          top = std::min(top, y);
          bottom = std::max(bottom, y);
        }
      }
    }
    if (bottom < top)
      return 0.0;
    return (bottom - top + 1) / static_cast<double>(image.rows);
  }

  void publishFollowCommand(const FollowResult& follow, const ros::Time& now)
  {
    if (!follow.found)
    {
      const bool timed_out = (now - last_detection_time_).toSec() > lost_timeout_;
      if (stop_on_lost_ || timed_out)
      {
        setStatus(timed_out ? "lost_stop" : "line_wait");
        resetRightLinePid();
        hardStop();
        return;
      }

      setStatus("line_wait_slow");
      geometry_msgs::Twist cmd;
      cmd.linear.x = lost_speed_;
      cmd.angular.z = lost_angular_speed_;
      publishCmd(cmd);
      return;
    }

    const double now_sec = now.toSec();
    const double dt = last_pid_time_ > 0.0 ? std::max(1e-3, now_sec - last_pid_time_) : 0.0;
    last_pid_time_ = now_sec;
    filtered_error_px_ = 0.78 * filtered_error_px_ + 0.22 * follow.error;
    const double derivative = dt > 0.0 ? (filtered_error_px_ - last_error_px_) / dt : 0.0;
    last_error_px_ = filtered_error_px_;

    double angular = -(kp_ * filtered_error_px_ + kd_ * derivative);
    angular = clampDouble(angular, -max_angular_speed_, max_angular_speed_);

    const double effective_base_speed = std::max(base_speed_, right_fast_base_speed_);
    const double error_abs = std::abs(filtered_error_px_);
    double linear = min_speed_;
    if (error_abs <= right_fast_error_px_)
    {
      linear = effective_base_speed;
    }
    else if (error_abs <= right_medium_error_px_)
    {
      const double t = (error_abs - right_fast_error_px_) /
                       std::max(1e-6, right_medium_error_px_ - right_fast_error_px_);
      linear = effective_base_speed + t * (right_medium_speed_ - effective_base_speed);
    }
    else
    {
      const double t = std::min((error_abs - right_medium_error_px_) /
                                    std::max(1e-6, right_hard_error_px_ - right_medium_error_px_),
                                1.0);
      linear = right_medium_speed_ + t * (min_speed_ - right_medium_speed_);
    }
    linear = clampDouble(linear, min_speed_, effective_base_speed);
    filtered_angular_ = angular_alpha_ * filtered_angular_ + (1.0 - angular_alpha_) * angular;

    geometry_msgs::Twist cmd;
    cmd.linear.x = linear;
    cmd.angular.z = filtered_angular_;
    setStatus("tracking_rightmost");
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

  void resetRightLinePid()
  {
    filtered_angular_ = 0.0;
    filtered_error_px_ = 0.0;
    last_error_px_ = 0.0;
    last_pid_time_ = 0.0;
    has_last_target_ = false;
  }

  void publishDebug(const cv::Mat& frame, const cv::Mat& mask, const FollowResult& follow,
                    const FinishResult& finish, const ros::Time& now)
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
    if (finish.box.area() > 0)
    {
      cv::rectangle(debug, finish.box, finish.detected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 160, 255), 2);
      cv::Point box_center(finish.box.x + finish.box.width / 2, finish.box.y + finish.box.height / 2);
      cv::circle(debug, box_center, 5, cv::Scalar(255, 255, 0), -1);
      cv::line(debug, cv::Point(kImageCols / 2, box_center.y), box_center, cv::Scalar(255, 255, 0), 1);
    }

    std::ostringstream line1;
    line1 << "state=" << status_ << " cmd=(" << std::fixed << std::setprecision(2) << last_linear_ << ","
          << last_angular_ << ") finish_frames=" << finish_frames_ << " found=" << boolText(follow.found);
    cv::putText(debug, line1.str(), cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 255, 0), 2);

    std::ostringstream line2;
    line2 << "target=(" << std::fixed << std::setprecision(1) << follow.target_x << "," << follow.target_y
          << ") err=" << follow.error << " c_err=" << last_finish_center_error_px_
          << " bottom=" << finish.bottom_ratio;
    cv::putText(debug, line2.str(), cv::Point(10, 215), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 220, 255), 2);

    try
    {
      sensor_msgs::ImagePtr out = cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug).toImageMsg();
      out->header.stamp = now;
      debug_image_pub_.publish(out);
    }
    catch (const cv_bridge::Exception& exc)
    {
      ROS_WARN_THROTTLE(2.0, "right_line_follow_cpp debug publish failed: %s", exc.what());
    }
  }

  void publishDebugInfo(const FollowResult& follow, const FinishResult& finish, const ros::Time& now)
  {
    std::ostringstream ss;
    ss << "status=" << status_
       << " elapsed=" << std::fixed << std::setprecision(2) << (now - start_time_).toSec()
       << " mask_mode=" << last_mask_mode_
       << " mask_ratio=" << last_mask_ratio_
       << " found=" << boolText(follow.found)
       << " fallback=" << boolText(follow.used_fallback)
       << " target_x=" << follow.target_x
       << " target_y=" << follow.target_y
       << " error=" << follow.error
       << " cmd_linear=" << last_linear_
       << " cmd_angular=" << last_angular_
       << " raw_points=" << follow.raw_line.size()
       << " path_points=" << follow.center_path.size()
       << " finish_detected=" << boolText(finish.detected)
       << " finish_frames=" << finish_frames_
       << " finish_box_count=" << finish_box_count_
       << " finish_box_armed=" << boolText(finish_box_armed_)
       << " finish_center_err=" << last_finish_center_error_px_
       << " finish_center_target_bottom=" << finish_center_target_bottom_ratio_
       << " box_w=" << finish.width_ratio
       << " box_h=" << finish.height_ratio
       << " box_bottom=" << finish.bottom_ratio
       << " box_hp=" << finish.horizontal_presence
       << " box_vp=" << finish.vertical_presence;
    std_msgs::String msg;
    msg.data = ss.str();
    debug_info_pub_.publish(msg);
    ROS_INFO_THROTTLE(0.5, "right_line_follow_cpp: %s", msg.data.c_str());
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

  int pixel(const cv::Mat& image, int x, int y) const
  {
    x = clampInt(x, 0, image.cols - 1);
    y = clampInt(y, 0, image.rows - 1);
    return static_cast<int>(image.at<uchar>(y, x));
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

  double begin_x_ = 25.0;
  double begin_y_ = 400.0;
  int seed_search_rows_ = 6;
  int seed_row_step_px_ = 25;
  int edge_probe_px_ = 6;
  double edge_threshold_ = 90.0;
  int trace_max_points_ = 300;
  int trace_step_guard_ = 4;
  int line_blur_kernel_ = 7;
  double sample_dist_m_ = 0.01;
  double aim_dist_m_ = 0.12;
  double forward_bias_m_ = 0.20;
  double pixel_per_meter_ = 500.0;
  double lane_width_m_ = 0.42;
  double right_center_bias_m_ = 0.035;

  double fallback_roi_y_start_ratio_ = 0.48;
  std::vector<double> right_scan_rows_;
  std::vector<double> fallback_rows_;
  double right_offset_px_ = 180.0;
  double fallback_right_offset_px_ = 125.0;
  double right_scan_bottom_weight_ = 1.8;
  int min_line_width_px_ = 5;
  int max_line_segment_width_px_ = 90;
  int min_segment_gap_px_ = 10;
  double max_target_jump_px_ = 120.0;
  double kp_ = 0.0042;
  double kd_ = 0.0014;

  double base_speed_ = 0.16;
  double min_speed_ = 0.07;
  double right_fast_base_speed_ = 0.22;
  double right_fast_error_px_ = 45.0;
  double right_medium_error_px_ = 130.0;
  double right_hard_error_px_ = 210.0;
  double right_medium_speed_ = 0.12;
  double curve_speed_error_scale_ = 0.18;
  double max_angular_speed_ = 0.55;
  double angular_alpha_ = 0.65;
  double lost_timeout_ = 0.8;
  double lost_speed_ = 0.06;
  double lost_angular_speed_ = 0.0;
  bool stop_on_lost_ = true;

  double finish_enable_delay_ = 18.0;
  double finish_min_width_ratio_ = 0.48;
  double finish_max_width_ratio_ = 0.92;
  double finish_min_height_ratio_ = 0.18;
  double finish_min_bottom_y_ratio_ = 0.70;
  double finish_roi_y_start_ratio_ = 0.52;
  double finish_min_horizontal_presence_ = 0.55;
  double finish_min_vertical_presence_ = 0.42;
  int finish_confirm_frames_ = 4;
  int finish_release_frames_ = 2;
  int finish_stop_on_box_count_ = 2;
  double finish_box_cooldown_sec_ = 2.0;
  double finish_approach_speed_ = 0.06;
  bool finish_centering_enabled_ = true;
  double finish_center_target_bottom_ratio_ = 0.90;
  double finish_center_max_duration_ = 1.20;
  double finish_center_angular_kp_ = 0.0020;
  double finish_center_max_angular_ = 0.16;
  double finish_forward_duration_ = 1.65;
  double finish_forward_speed_ = 0.08;
  double finish_stop_hold_ = 2.0;

  State state_ = State::Idle;
  ros::Time start_time_;
  ros::Time state_start_time_;
  ros::Time last_detection_time_;
  ros::Time last_image_time_;
  cv::Mat inv_perspective_;
  std::string status_ = "idle";

  int finish_frames_ = 0;
  int finish_lost_frames_ = 0;
  int finish_box_count_ = 0;
  bool finish_box_armed_ = true;
  ros::Time finish_box_cooldown_until_;
  double last_finish_center_error_px_ = 0.0;
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
  ros::init(argc, argv, "right_line_follow_cpp_node");
  RightLineFollowCppNode node;
  ros::spin();
  return 0;
}
