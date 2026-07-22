#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
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
constexpr double kPi = 3.14159265358979323846;

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
    TurnRight,
    Forward50cm,
    FinalStop,
    Finish
  };

  struct FollowResult
  {
    bool found = false;
    int right_x = -1;
    int line_support = 0;
    double line_confidence = 0.0;
    double line_span_ratio = 0.0;
    double error = 0.0;
    double filtered_error = 0.0;
    double linear = 0.0;
    double angular = 0.0;
    int guard_level = 0;
  };

  struct RightLineResult
  {
    bool found = false;
    int x = -1;
    int support = 0;
    double confidence = 0.0;
    double span_ratio = 0.0;
  };

  struct LineCandidate
  {
    int support = 0;
    double weighted_x = 0.0;
    double weight_sum = 0.0;
    double min_y = 0.0;
    double max_y = 0.0;
  };

  struct Segment
  {
    int left = 0;
    int right = 0;
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
    private_nh_.param<std::string>("status_topic", status_topic_, "/stable_right_track_end_stop/status");
    private_nh_.param<std::string>("debug_image_topic", debug_image_topic_, "/stable_right_track_end_stop/debug_image");
    private_nh_.param<std::string>("debug_info_topic", debug_info_topic_, "/stable_right_track_end_stop/debug_info");

    private_nh_.param("auto_start", auto_start_, true);
    private_nh_.param("startup_time", startup_time_, 2.0);
    private_nh_.param("startup_speed", startup_speed_, 0.45);

    private_nh_.param("right_line_offset_px", right_line_offset_px_, 170.0);
    target_right_x_ = clampInt(
        static_cast<int>(std::round(kImageCols * 0.5 +
                                    right_line_offset_px_)),
        0, kImageCols - 1);
    private_nh_.param("base_speed", base_speed_, 0.24);
    private_nh_.param("curve_speed", curve_speed_, 0.15);
    private_nh_.param("search_speed", search_speed_, 0.0);
    private_nh_.param("search_angular_speed", search_angular_speed_, -0.08);
    private_nh_.param("lost_linear_speed", lost_linear_speed_, 0.0);
    private_nh_.param("lost_angular_speed", lost_angular_speed_, -0.08);
    private_nh_.param("lost_guard_frames", lost_guard_frames_, 3);
    private_nh_.param("reacquire_confirm_frames", reacquire_confirm_frames_, 3);
    private_nh_.param("kp", kp_, 0.0037);
    private_nh_.param("kd", kd_, 0.0006);
    private_nh_.param("error_alpha", error_alpha_, 0.15);
    private_nh_.param("curve_error_threshold", curve_error_threshold_, 38.0);
    private_nh_.param("curve_angular_gain", curve_angular_gain_, 1.05);
    private_nh_.param("max_angular_speed", max_angular_speed_, 0.40);
    private_nh_.param("steering_deadband_px", steering_deadband_px_, 7.0);
    private_nh_.param("max_straight_angular_speed", max_straight_angular_speed_, 0.15);
    private_nh_.param("max_right_angular_speed", max_right_angular_speed_, 0.34);
    private_nh_.param("straight_angular_alpha", straight_angular_alpha_, 0.82);
    private_nh_.param("curve_angular_alpha", curve_angular_alpha_, 0.58);
    private_nh_.param("straight_angular_step", straight_angular_step_, 0.03);
    private_nh_.param("curve_angular_step", curve_angular_step_, 0.06);
    private_nh_.param("right_warning_error_px", right_warning_error_px_, 28.0);
    private_nh_.param("right_hard_error_px", right_hard_error_px_, 52.0);
    private_nh_.param("right_guard_speed", right_guard_speed_, 0.11);
    private_nh_.param("right_guard_away_angular", right_guard_away_angular_, 0.10);
    private_nh_.param("right_hard_away_angular", right_hard_away_angular_, 0.24);
    private_nh_.param("deadband_angular_decay", deadband_angular_decay_, 0.45);

    private_nh_.param("roi_y_start_ratio", roi_y_start_ratio_, 0.60);
    private_nh_.param("white_s_max", white_s_max_, 45);
    private_nh_.param("white_v_min", white_v_min_, 200);
    private_nh_.param("morph_kernel_size", morph_kernel_size_, 5);
    private_nh_.param("min_component_area", min_component_area_, 260.0);
    right_scan_rows_ = {
        0.95, 0.92, 0.88, 0.84, 0.80,
        0.75, 0.70, 0.64, 0.58};
    private_nh_.param("right_scan_bottom_weight",
                      right_scan_bottom_weight_, 1.8);
    private_nh_.param("min_line_width_px", min_line_width_px_, 5);
    private_nh_.param("max_line_segment_width_px",
                      max_line_segment_width_px_, 90);
    private_nh_.param("min_segment_gap_px", min_segment_gap_px_, 10);
    private_nh_.param("right_min_scan_support",
                      right_min_scan_support_, 3);
    private_nh_.param("max_target_jump_px", max_target_jump_px_, 160.0);

    private_nh_.param("end_enable_delay", end_enable_delay_, 3.0);
    private_nh_.param("end_roi_y_start_ratio", end_roi_y_start_ratio_, 0.87);
    private_nh_.param("end_min_width_ratio", end_min_width_ratio_, 0.45);
    private_nh_.param("end_stop_hold", end_stop_hold_, 1.0);
    private_nh_.param("end_forward_distance_m", end_forward_distance_m_, 0.65);
    private_nh_.param("end_forward_speed", end_forward_speed_, 0.17);
    private_nh_.param("end_turn_left_angle_deg", end_turn_left_angle_deg_, 10.0);
    private_nh_.param("end_turn_left_angular_speed", end_turn_left_angular_speed_, 0.50);

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
        if ((now - start_time_).toSec() < startup_time_)
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
        follow = computeFollow(mask);
        if (!follow.found)
        {
          reacquire_count_ = 0;
          setStatus("stable_right_search");
          cmd.linear.x = search_speed_;
          cmd.angular.z = search_angular_speed_;
          publishCmd(cmd);
        }
        else
        {
          ++reacquire_count_;
          last_right_x_ = follow.right_x;
          if (reacquire_count_ < reacquire_confirm_frames_)
          {
            setStatus("stable_right_search_confirming_continuous_line");
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
            publishCmd(cmd);
          }
          else
          {
            ROS_INFO("stable right continuous line found: support=%d confidence=%.2f",
                     follow.line_support, follow.line_confidence);
            last_detection_time_ = now;
            line_was_lost_ = false;
            state_ = State::Follow;
            state_start_time_ = now;
            publishFollowCommand(follow);
          }
        }
        break;

      case State::Follow:
        follow = computeFollow(mask);
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
          publishFollowCommand(follow);
        }
        break;

      case State::EndDetected:
        setStatus("stable_right_end_detected");
        hardStop();
        if ((now - state_start_time_).toSec() >= end_stop_hold_)
        {
          state_ = State::TurnRight;
          state_start_time_ = now;
          ROS_INFO("turning left %.1f deg at %.2f rad/s before parking",
                   end_turn_left_angle_deg_, end_turn_left_angular_speed_);
        }
        break;

      case State::TurnRight:
      {
        const double turn_duration = (end_turn_left_angle_deg_ * kPi / 180.0) /
                                     std::max(end_turn_left_angular_speed_, 1e-6);
        if ((now - state_start_time_).toSec() < turn_duration)
        {
          setStatus("stable_right_turn_left_align");
          cmd.angular.z = end_turn_left_angular_speed_;
          publishCmd(cmd);
        }
        else
        {
          state_ = State::Forward50cm;
          state_start_time_ = now;
          hardStop();
          ROS_INFO("driving straight forward %.2f m at %.2f m/s after alignment",
                   end_forward_distance_m_, end_forward_speed_);
        }
        break;
      }

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

  cv::Mat extractWhiteMask(const cv::Mat& roi) const
  {
    cv::Mat blur;
    cv::GaussianBlur(roi, blur, cv::Size(5, 5), 0);

    cv::Mat hsv;
    cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(0, 0, white_v_min_), cv::Scalar(180, white_s_max_, 255), mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_kernel_size_, morph_kernel_size_));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::medianBlur(mask, mask, 5);
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat clean_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (const auto& contour : contours)
    {
      if (cv::contourArea(contour) > min_component_area_)
        cv::drawContours(clean_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
    }
    return clean_mask;
  }

  FollowResult computeFollow(const cv::Mat& mask)
  {
    FollowResult result;
    const RightLineResult line = findRightLine(mask);
    result.right_x = line.x;
    result.line_support = line.support;
    result.line_confidence = line.confidence;
    result.line_span_ratio = line.span_ratio;
    result.found = line.found;
    if (!result.found)
      return result;

    result.error = static_cast<double>(target_right_x_ - result.right_x);
    filtered_error_ = (1.0 - error_alpha_) * filtered_error_ + error_alpha_ * result.error;
    const double d_error = filtered_error_ - last_error_;
    last_error_ = filtered_error_;

    const double abs_error = std::fabs(filtered_error_);
    const bool inside_deadband = abs_error <= steering_deadband_px_;
    const double control_error = inside_deadband ? 0.0 : filtered_error_;
    const double control_derivative = inside_deadband ? 0.0 : d_error;
    double angular = kp_ * control_error + kd_ * control_derivative;
    double linear = base_speed_;
    const bool in_curve = abs_error > curve_error_threshold_;
    if (in_curve)
    {
      linear = curve_speed_;
      angular *= curve_angular_gain_;
    }

    // error = target - detected.  A positive error means that the right line
    // has moved left in the image and the car is getting too close to it.
    if (filtered_error_ >= right_warning_error_px_)
    {
      result.guard_level = 1;
      linear = std::min(linear, right_guard_speed_);
      angular = std::max(angular, right_guard_away_angular_);
    }
    if (filtered_error_ >= right_hard_error_px_)
    {
      result.guard_level = 2;
      linear = 0.0;
      angular = std::max(angular, right_hard_away_angular_);
    }

    const double positive_limit = in_curve ? max_angular_speed_ : max_straight_angular_speed_;
    const double negative_limit = std::min(positive_limit, max_right_angular_speed_);
    angular = clampDouble(angular, -negative_limit, positive_limit);

    // Smooth small straight-line corrections heavily.  In a real bend use a
    // lighter filter and a larger step so the car still turns in time.
    const double angular_alpha = in_curve ? curve_angular_alpha_ : straight_angular_alpha_;
    const double filtered_target = angular_alpha * filtered_angular_ +
                                   (1.0 - angular_alpha) * angular;
    const double angular_step = in_curve ? curve_angular_step_ : straight_angular_step_;
    filtered_angular_ += clampDouble(filtered_target - filtered_angular_,
                                     -angular_step, angular_step);
    if (inside_deadband)
    {
      filtered_angular_ *= deadband_angular_decay_;
      if (std::fabs(filtered_angular_) < 0.015)
        filtered_angular_ = 0.0;
    }

    // Enforce the active straight/curve limit on the final command as well.
    // Without this second clamp, a large bend command stored in the filter can
    // leak into the straight section and keep steering toward the line.
    filtered_angular_ = clampDouble(filtered_angular_, -negative_limit, positive_limit);
    if (result.guard_level == 1)
      filtered_angular_ = std::max(filtered_angular_, right_guard_away_angular_);
    else if (result.guard_level >= 2)
      filtered_angular_ = std::max(filtered_angular_, right_hard_away_angular_);

    result.filtered_error = filtered_error_;
    result.linear = linear;
    result.angular = filtered_angular_;
    return result;
  }

  RightLineResult findRightLine(const cv::Mat& mask) const
  {
    RightLineResult result;

    // Keep every disconnected white marking separate.  The former
    // row-by-row "rightmost segment" method could take low scan points from
    // the old road and high scan points from the road after the right turn,
    // then average them into a fictitious line in the middle of the image.
    cv::Mat labels;
    const int component_count =
        cv::connectedComponents(mask, labels, 8, CV_32S);
    if (component_count <= 1)
      return result;

    std::vector<LineCandidate> candidates(component_count);

    for (std::size_t i = 0; i < right_scan_rows_.size(); ++i)
    {
      const int y = clampInt(
          static_cast<int>(mask.rows * right_scan_rows_[i]),
          0, mask.rows - 1);
      const double bottom_factor =
          static_cast<double>(right_scan_rows_.size() - i) /
          std::max(1.0,
                   static_cast<double>(right_scan_rows_.size() - 1));
      const double weight =
          1.0 + bottom_factor * (right_scan_bottom_weight_ - 1.0);

      const int* label_row = labels.ptr<int>(y);
      int x = 0;
      while (x < labels.cols)
      {
        const int label = label_row[x];
        if (label == 0)
        {
          ++x;
          continue;
        }

        const int left = x;
        while (x < labels.cols && label_row[x] == label)
          ++x;
        const int right = x - 1;
        const int width = right - left + 1;
        if (width < min_line_width_px_ ||
            width > max_line_segment_width_px_)
          continue;

        LineCandidate& candidate = candidates[label];
        const double center =
            0.5 * static_cast<double>(left + right);
        candidate.weighted_x += center * weight;
        candidate.weight_sum += weight;
        if (candidate.support == 0)
        {
          candidate.min_y = static_cast<double>(y);
          candidate.max_y = static_cast<double>(y);
        }
        else
        {
          candidate.min_y =
              std::min(candidate.min_y, static_cast<double>(y));
          candidate.max_y =
              std::max(candidate.max_y, static_cast<double>(y));
        }
        ++candidate.support;
      }
    }

    const LineCandidate* best = nullptr;
    double best_x = -1.0;
    double best_distance = std::numeric_limits<double>::max();
    for (int label = 1; label < component_count; ++label)
    {
      const LineCandidate& candidate = candidates[label];
      if (candidate.support < right_min_scan_support_ ||
          candidate.weight_sum <= 0.0)
        continue;

      const double candidate_x =
          candidate.weighted_x / candidate.weight_sum;

      // A large jump is not "fixed" by clamping: clamping manufactures a
      // plausible coordinate for the wrong line.  Reject it and let the
      // existing lost-line right-search behaviour find the true boundary.
      if (last_right_x_ >= 0 &&
          std::fabs(candidate_x - last_right_x_) >
              max_target_jump_px_)
        continue;

      // Preserve the original tracking pixel: once a right boundary has been
      // acquired, choose the connected component nearest to its previous raw
      // position.  Only at startup, when no previous pixel exists, retain the
      // original rightmost-line preference.
      const double distance =
          last_right_x_ >= 0
              ? std::fabs(candidate_x - last_right_x_)
              : -candidate_x;
      if (distance < best_distance)
      {
        best_distance = distance;
        best_x = candidate_x;
        best = &candidate;
      }
    }

    if (best == nullptr)
      return result;

    result.found = true;
    result.x = clampInt(static_cast<int>(std::round(best_x)),
                        0, mask.cols - 1);
    result.support = best->support;
    result.confidence =
        static_cast<double>(best->support) /
        std::max(1.0, static_cast<double>(right_scan_rows_.size()));
    result.span_ratio =
        (best->max_y - best->min_y) /
        std::max(1.0, static_cast<double>(mask.rows));
    return result;
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
        const int width = x - start;
        if (width >= min_line_width_px_)
          segments.push_back(Segment{start, x - 1, width});
        start = -1;
      }
    }
    if (start >= 0)
    {
      const int width = row.cols - start;
      if (width >= min_line_width_px_)
        segments.push_back(
            Segment{start, row.cols - 1, width});
    }

    if (segments.empty())
      return segments;
    std::vector<Segment> merged;
    merged.push_back(segments.front());
    for (std::size_t i = 1; i < segments.size(); ++i)
    {
      Segment& previous = merged.back();
      const Segment& current = segments[i];
      if (current.left - previous.right <= min_segment_gap_px_)
      {
        previous.right = current.right;
        previous.width = previous.right - previous.left + 1;
      }
      else
      {
        merged.push_back(current);
      }
    }
    return merged;
  }

  void publishFollowCommand(const FollowResult& follow)
  {
    geometry_msgs::Twist cmd;
    if (!follow.found)
    {
      line_was_lost_ = true;
      reacquire_count_ = 0;
      ++lost_frame_count_;
      if (filtered_error_ >= right_warning_error_px_ &&
          lost_frame_count_ <= lost_guard_frames_)
      {
        // If the last trustworthy observation already showed that the car was
        // close to the right boundary, briefly move away.  This guard must be
        // bounded: while the line is absent filtered_error_ cannot update, so
        // an unbounded condition here locks the vehicle into a left spin.
        setStatus("stable_right_lost_brief_left_guard");
        cmd.linear.x = 0.0;
        cmd.angular.z = right_hard_away_angular_;
      }
      else
      {
        setStatus("stable_right_lost_rotate_right_in_place");
        cmd.linear.x = lost_linear_speed_;
        cmd.angular.z = lost_angular_speed_;
      }
      publishCmd(cmd);
      return;
    }

    lost_frame_count_ = 0;
    if (line_was_lost_)
    {
      ++reacquire_count_;
      last_right_x_ = follow.right_x;
      setStatus("stable_right_reacquire_confirming_continuous_line");
      if (reacquire_count_ < reacquire_confirm_frames_)
      {
        publishStop();
        return;
      }
      line_was_lost_ = false;
      reacquire_count_ = 0;
      filtered_error_ = follow.error;
      last_error_ = follow.error;
      filtered_angular_ = 0.0;
      setStatus("stable_right_reacquired_continuous_line");
      publishStop();
      return;
    }

    last_right_x_ = follow.right_x;
    last_detection_time_ = ros::Time::now();
    cmd.linear.x = follow.linear;
    cmd.angular.z = follow.angular;
    if (follow.guard_level >= 2)
      setStatus("stable_right_tracking_right_hard_guard");
    else if (follow.guard_level == 1)
      setStatus("stable_right_tracking_right_guard");
    else if (std::fabs(follow.filtered_error) > curve_error_threshold_)
      setStatus("stable_right_tracking_curve");
    else
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
          << last_linear_ << "," << last_angular_ << ") found="
          << boolText(follow.found) << " conf=" << follow.line_confidence;
    cv::putText(debug, line1.str(), cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 255, 0), 2);

    std::ostringstream line2;
    line2 << "right_x=" << follow.right_x << " err=" << std::fixed << std::setprecision(1)
          << follow.filtered_error << " support=" << follow.line_support
          << " span=" << std::setprecision(2) << follow.line_span_ratio
          << " guard=" << follow.guard_level;
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
       << " line_support=" << follow.line_support
       << " line_confidence=" << follow.line_confidence
       << " line_span_ratio=" << follow.line_span_ratio
       << " error=" << follow.error
       << " filtered_error=" << follow.filtered_error
       << " guard_level=" << follow.guard_level
       << " lost_frames=" << lost_frame_count_
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
  double startup_time_ = 2.0;
  double startup_speed_ = 0.45;

  double right_line_offset_px_ = 170.0;
  int target_right_x_ = 490;
  double base_speed_ = 0.24;
  double curve_speed_ = 0.15;
  double search_speed_ = 0.0;
  double search_angular_speed_ = -0.08;
  double lost_linear_speed_ = 0.0;
  double lost_angular_speed_ = -0.08;
  int lost_guard_frames_ = 3;
  int reacquire_confirm_frames_ = 3;
  double kp_ = 0.0037;
  double kd_ = 0.0006;
  double error_alpha_ = 0.15;
  double curve_error_threshold_ = 38.0;
  double curve_angular_gain_ = 1.05;
  double max_angular_speed_ = 0.40;
  double steering_deadband_px_ = 7.0;
  double max_straight_angular_speed_ = 0.15;
  double max_right_angular_speed_ = 0.34;
  double straight_angular_alpha_ = 0.82;
  double curve_angular_alpha_ = 0.58;
  double straight_angular_step_ = 0.03;
  double curve_angular_step_ = 0.06;
  double right_warning_error_px_ = 28.0;
  double right_hard_error_px_ = 52.0;
  double right_guard_speed_ = 0.11;
  double right_guard_away_angular_ = 0.10;
  double right_hard_away_angular_ = 0.24;
  double deadband_angular_decay_ = 0.45;

  double roi_y_start_ratio_ = 0.60;
  int white_s_max_ = 45;
  int white_v_min_ = 200;
  int morph_kernel_size_ = 5;
  double min_component_area_ = 260.0;
  std::vector<double> right_scan_rows_;
  double right_scan_bottom_weight_ = 1.8;
  int min_line_width_px_ = 5;
  int max_line_segment_width_px_ = 90;
  int min_segment_gap_px_ = 10;
  int right_min_scan_support_ = 3;
  double max_target_jump_px_ = 160.0;

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
  std::string status_ = "idle";

  double last_error_ = 0.0;
  double filtered_error_ = 0.0;
  double filtered_angular_ = 0.0;
  int last_right_x_ = -1;
  bool line_was_lost_ = false;
  int lost_frame_count_ = 0;
  int reacquire_count_ = 0;
  double last_linear_ = 0.0;
  double last_angular_ = 0.0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "stable_right_track_end_stop_node");
  StableRightTrackEndStopNode node;
  ros::spin();
  return 0;
}