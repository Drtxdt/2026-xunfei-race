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
    TurnRight,
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
    private_nh_.param("startup_time", startup_time_, 2.8);
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

    private_nh_.param("roi_y_start_ratio", roi_y_start_ratio_, 0.60);
    private_nh_.param("white_s_max", white_s_max_, 45);
    private_nh_.param("white_v_min", white_v_min_, 200);
    private_nh_.param("morph_kernel_size", morph_kernel_size_, 5);
    private_nh_.param("min_component_area", min_component_area_, 260.0);

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
          setStatus("stable_right_search");
          cmd.linear.x = search_speed_;
          cmd.angular.z = search_angular_speed_;
          publishCmd(cmd);
        }
        else
        {
          ROS_INFO("stable right line found");
          last_right_x_ = follow.right_x;
          last_detection_time_ = now;
          state_ = State::Follow;
          state_start_time_ = now;
          publishFollowCommand(follow);
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
        const double turn_duration = (end_turn_left_angle_deg_ * M_PI / 180.0) /
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
    result.right_x = findRightLine(mask);
    result.found = result.right_x >= 0;
    if (!result.found)
      return result;

    result.error = static_cast<double>(target_right_x_ - result.right_x);
    filtered_error_ = (1.0 - error_alpha_) * filtered_error_ + error_alpha_ * result.error;
    const double d_error = filtered_error_ - last_error_;
    last_error_ = filtered_error_;

    double angular = kp_ * filtered_error_ + kd_ * d_error;

if (filtered_error_ < -15.0)
{
  angular *= 1.35;
}
    double linear = base_speed_;
    if (std::fabs(filtered_error_) > curve_error_threshold_)
    {
      linear = curve_speed_;
      angular *= curve_angular_gain_;
    }

    result.filtered_error = filtered_error_;
    result.linear = linear;
    result.angular = clampDouble(angular, -max_angular_speed_, max_angular_speed_);
    return result;
  }

  int findRightLine(const cv::Mat& mask) const
  {
    const int h = mask.rows;
    std::vector<int> rows = {
    static_cast<int>(h * 0.35),
    static_cast<int>(h * 0.50),
    static_cast<int>(h * 0.65),
    static_cast<int>(h * 0.80)
    };
    std::vector<int> points;

    for (int y : rows)
    {
      y = clampInt(y, 0, mask.rows - 1);
      const uchar* ptr = mask.ptr<uchar>(y);
      for (int x = mask.cols - 1; x >= 0; --x)
      {
        if (ptr[x] > 0)
        {
          points.push_back(x);
          break;
        }
      }
    }

    if (points.empty())
      return -1;

    std::sort(points.begin(), points.end());

    return points[points.size() / 2];
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
        segments.push_back(Segment{start, x - 1, x - start});
        start = -1;
      }
    }
    if (start >= 0)
      segments.push_back(Segment{start, row.cols - 1, row.cols - start});
    return segments;
  }

  void publishFollowCommand(const FollowResult& follow)
  {
    geometry_msgs::Twist cmd;
    if (!follow.found)
    {
      setStatus("stable_right_lost");
      cmd.linear.x = lost_linear_speed_;
      cmd.angular.z = last_right_x_ >= 0 ? lost_angular_speed_ : search_angular_speed_;
      publishCmd(cmd);
      return;
    }

    last_right_x_ = follow.right_x;
    last_detection_time_ = ros::Time::now();
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
          << follow.filtered_error << " end_w=" << std::setprecision(2) << end_result.best_width_ratio;
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
  double startup_time_ = 2.8;
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

  double roi_y_start_ratio_ = 0.60;
  int white_s_max_ = 45;
  int white_v_min_ = 200;
  int morph_kernel_size_ = 5;
  double min_component_area_ = 260.0;

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
  int last_right_x_ = -1;
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