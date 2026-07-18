#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stable_track
{
constexpr double kPi = 3.14159265358979323846;

struct SamplePoint
{
  double x = 0.0;
  double y = 0.0;
};

struct LineModel
{
  bool valid = false;
  double dx_dy = 0.0;
  double intercept = 0.0;
  double rmse = 1e9;
  int point_count = 0;
};

struct KinkResult
{
  bool detected = false;
  int split_index = -1;
  double angle_deg = 0.0;
  double combined_rmse = 1e9;
  SamplePoint vertex;
  LineModel near_arm;
  LineModel far_arm;
};

struct RightObservation
{
  bool valid = false;
  double detected_x = 0.0;
  double lateral_error_px = 0.0;
  double heading_error_px = 0.0;
  double heading_angle_deg = 0.0;
  double confidence = 0.0;
};

class EndLineDebouncer
{
public:
  explicit EndLineDebouncer(int required_frames = 3)
      : required_frames_(std::max(1, required_frames))
  {
  }

  bool update(bool detected, bool enabled)
  {
    if (!enabled || !detected)
    {
      count_ = 0;
      return false;
    }
    count_ = std::min(required_frames_, count_ + 1);
    return count_ >= required_frames_;
  }

  void reset()
  {
    count_ = 0;
  }

  int count() const
  {
    return count_;
  }

private:
  int required_frames_ = 3;
  int count_ = 0;
};

double xAt(const LineModel& line, double y)
{
  return line.dx_dy * y + line.intercept;
}

LineModel fitLineModel(const std::vector<SamplePoint>& points)
{
  LineModel result;
  if (points.size() < 2)
    return result;

  double mean_x = 0.0;
  double mean_y = 0.0;
  for (const SamplePoint& point : points)
  {
    mean_x += point.x;
    mean_y += point.y;
  }
  mean_x /= static_cast<double>(points.size());
  mean_y /= static_cast<double>(points.size());

  double numerator = 0.0;
  double denominator = 0.0;
  for (const SamplePoint& point : points)
  {
    numerator += (point.y - mean_y) * (point.x - mean_x);
    denominator += (point.y - mean_y) * (point.y - mean_y);
  }
  if (denominator < 1e-9)
    return result;

  result.valid = true;
  result.dx_dy = numerator / denominator;
  result.intercept = mean_x - result.dx_dy * mean_y;
  result.point_count = static_cast<int>(points.size());
  double squared_error = 0.0;
  for (const SamplePoint& point : points)
  {
    const double residual = point.x - xAt(result, point.y);
    squared_error += residual * residual;
  }
  result.rmse = std::sqrt(squared_error / static_cast<double>(points.size()));
  return result;
}

KinkResult detectKink(const std::vector<SamplePoint>& points, int min_arm_points,
                      double min_angle_deg, double max_angle_deg)
{
  KinkResult best;
  if (min_arm_points < 2 ||
      static_cast<int>(points.size()) < min_arm_points * 2)
    return best;

  const LineModel whole = fitLineModel(points);
  for (int split = min_arm_points - 1;
       split <= static_cast<int>(points.size()) - min_arm_points; ++split)
  {
    const std::vector<SamplePoint> first(points.begin(), points.begin() + split + 1);
    const std::vector<SamplePoint> second(points.begin() + split, points.end());
    const LineModel a = fitLineModel(first);
    const LineModel b = fitLineModel(second);
    if (!a.valid || !b.valid)
      continue;

    const double angle_deg =
        std::fabs(std::atan(a.dx_dy) - std::atan(b.dx_dy)) * 180.0 / kPi;
    const double combined_rmse =
        (a.rmse * static_cast<double>(first.size()) +
         b.rmse * static_cast<double>(second.size())) /
        static_cast<double>(first.size() + second.size());
    if (angle_deg < min_angle_deg || angle_deg > max_angle_deg)
      continue;
    if (whole.valid && whole.rmse > 1e-6 &&
        combined_rmse >= whole.rmse * 0.70)
      continue;
    if (combined_rmse >= best.combined_rmse)
      continue;

    best.detected = true;
    best.split_index = split;
    best.angle_deg = angle_deg;
    best.combined_rmse = combined_rmse;
    best.vertex = points[split];
    best.far_arm = a;
    best.near_arm = b;
  }
  return best;
}

RightObservation observeRightBoundary(const LineModel& right,
                                      double control_y,
                                      double far_y,
                                      double target_right_x)
{
  RightObservation result;
  if (!right.valid || control_y <= far_y)
    return result;
  result.valid = true;
  result.detected_x = xAt(right, control_y);
  const double far_x = xAt(right, far_y);
  result.lateral_error_px = target_right_x - result.detected_x;
  result.heading_error_px = far_x - result.detected_x;
  result.heading_angle_deg =
      std::atan2(result.heading_error_px, control_y - far_y) *
      180.0 / kPi;
  return result;
}

bool parkingEnabled(int completed_corners, int required_corners)
{
  return completed_corners >= required_corners;
}

double advanceDuration(double distance_m, double speed_mps)
{
  return speed_mps > 1e-9 ? distance_m / speed_mps : 0.0;
}

}  // namespace stable_track

#ifdef STABLE_TRACK_SELF_TEST

using stable_track::KinkResult;
using stable_track::LineModel;
using stable_track::RightObservation;
using stable_track::SamplePoint;
using stable_track::EndLineDebouncer;
using stable_track::advanceDuration;
using stable_track::detectKink;
using stable_track::fitLineModel;
using stable_track::observeRightBoundary;
using stable_track::parkingEnabled;

static void expect(bool value, const char* message)
{
  if (!value)
    throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
  const std::vector<SamplePoint> straight{{100, 0}, {110, 10}, {120, 20}, {130, 30}};
  const LineModel line = fitLineModel(straight);
  expect(line.valid, "straight line must fit");
  expect(std::fabs(line.dx_dy - 1.0) < 1e-6, "slope must be one");

  const std::vector<SamplePoint> kink{
      {90, 0}, {100, 10}, {110, 20}, {120, 30},
      {120, 40}, {110, 50}, {100, 60}, {90, 70}};
  const KinkResult corner = detectKink(kink, 3, 20.0, 85.0);
  expect(corner.detected, "two stable arms must form a corner");
  expect(!detectKink(straight, 2, 20.0, 85.0).detected,
         "one straight line must not form a corner");

  expect(!parkingEnabled(1, 2), "parking must stay disabled before two corners");
  expect(parkingEnabled(2, 2), "parking must enable after two corners");
  expect(std::fabs(advanceDuration(0.20, 0.10) - 2.0) < 1e-9,
         "twenty centimetres at 0.10 m/s must take two seconds");

  EndLineDebouncer end_line(3);
  expect(!end_line.update(true, false),
         "end line must be ignored while parking is gated");
  expect(!end_line.update(true, true),
         "one enabled end-line frame must be insufficient");
  expect(!end_line.update(true, true),
         "two enabled end-line frames must be insufficient");
  expect(end_line.update(true, true),
         "three enabled end-line frames must confirm parking");
  expect(!end_line.update(false, true),
         "one negative frame must clear end-line confirmation");

  LineModel tracked_right;
  tracked_right.valid = true;
  tracked_right.intercept = 210.0;
  tracked_right.rmse = 1.0;
  tracked_right.point_count = 8;
  const RightObservation right_observation =
      observeRightBoundary(tracked_right, 160.0, 30.0, 200.0);
  expect(right_observation.valid,
         "a fitted right boundary must produce a right-only observation");
  expect(std::fabs(right_observation.lateral_error_px + 10.0) < 1e-9,
         "right-only error must preserve target minus detected position");
  expect(std::fabs(right_observation.heading_angle_deg) < 1e-9,
         "vertical right boundary must have zero heading angle");

  if (argc > 1)
  {
    std::ifstream input(argv[1], std::ios::binary);
    const std::string source((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
    const std::string corner_advance = std::string("Corner") + "Advance";
    const std::string reacquire = std::string("Reacquire") + "RightLine";
    const std::string align_right = std::string("Align") + "RightLine";
    const std::string lost_timeout = std::string("lost_stop") + "_timeout";
    const std::string corner_counter = std::string("completed_corner") + "_count_";
    const std::string blind_lost_speed =
        std::string("lost_hold") + "_speed_";
    const std::string active_left_line =
        std::string("vision.") + "left_line";
    const std::string active_left_points =
        std::string("vision.") + "left_points";
    const std::string permanent_reacquire_fault =
        std::string("enterFault(\"") + "reacquire_timeout";
    const std::string right_target =
        std::string("target_right_x_") + ", 200.0";
    expect(source.find(corner_advance) != std::string::npos,
           "corner advance state must exist");
    expect(source.find(reacquire) != std::string::npos,
           "right-line reacquisition state must exist");
    expect(source.find(align_right) != std::string::npos,
           "right-only alignment state must exist");
    expect(source.find(lost_timeout) != std::string::npos,
           "lost-line stop timeout parameter must exist");
    expect(source.find(corner_counter) != std::string::npos,
           "completed corner counter must exist");
    expect(source.find(blind_lost_speed) == std::string::npos,
           "lost line must never keep a blind forward speed");
    expect(source.find(active_left_line) == std::string::npos,
           "production control must never consume a left line");
    expect(source.find(active_left_points) == std::string::npos,
           "production vision must never extract left-line points");
    expect(source.find(permanent_reacquire_fault) == std::string::npos,
           "reacquire timeout must not enter a permanent fault stop");
    expect(source.find(right_target) != std::string::npos,
           "right-only tracking must retain the calibrated 200px target");
  }
  return 0;
}

#else

#include <iomanip>
#include <sstream>

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

const char* boolText(bool value)
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
    end_line_debouncer_ =
        stable_track::EndLineDebouncer(end_confirm_frames_);

    cmd_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_vel_topic_, 1);
    status_pub_ = nh_.advertise<std_msgs::String>(status_topic_, 1);
    debug_info_pub_ = nh_.advertise<std_msgs::String>(debug_info_topic_, 1);
    debug_image_pub_ = nh_.advertise<sensor_msgs::Image>(debug_image_topic_, 1);
    image_sub_ = nh_.subscribe(
        image_topic_, 1, &StableRightTrackEndStopNode::imageCallback, this);

    const ros::Time now = ros::Time::now();
    start_time_ = now;
    state_start_time_ = now;
    last_confident_line_time_ = now;
    corner_cooldown_until_ = now;
    state_ = auto_start_ ? State::StartupForward : State::Idle;
    status_ = auto_start_ ? "stable_right_startup_forward" : "idle";

    ROS_INFO("stable_right_track_end_stop_node started: image=%s cmd_vel=%s",
             image_topic_.c_str(), cmd_vel_topic_.c_str());
  }

private:
  enum class State
  {
    Idle,
    StartupForward,
    SearchRightLine,
    Follow,
    CornerConfirm,
    CornerAdvance,
    CornerStop,
    TurnRight,
    ReacquireRightLine,
    AlignRightLine,
    EndDetected,
    ParkingTurnLeft,
    ParkingForward,
    FinalStop,
    Finish
  };

  struct Segment
  {
    int left = 0;
    int right = 0;
    int width = 0;

    double center() const
    {
      return 0.5 * static_cast<double>(left + right);
    }
  };

  struct EndLineResult
  {
    bool detected = false;
    double width_ratio = 0.0;
    double y_ratio = 0.0;
  };

  struct VisionResult
  {
    stable_track::LineModel right_line;
    stable_track::RightObservation right;
    stable_track::KinkResult corner;
    std::vector<stable_track::SamplePoint> right_points;
    EndLineResult end_line;
    double right_confidence = 0.0;
  };

  void loadParams()
  {
    private_nh_.param<std::string>("image_topic", image_topic_,
                                   "/usb_cam/image_raw");
    private_nh_.param<std::string>("cmd_vel_topic", cmd_vel_topic_, "/cmd_vel");
    private_nh_.param<std::string>(
        "status_topic", status_topic_, "/stable_right_track_end_stop/status");
    private_nh_.param<std::string>(
        "debug_image_topic", debug_image_topic_,
        "/stable_right_track_end_stop/debug_image");
    private_nh_.param<std::string>(
        "debug_info_topic", debug_info_topic_,
        "/stable_right_track_end_stop/debug_info");

    private_nh_.param("auto_start", auto_start_, true);
    private_nh_.param("startup_time", startup_time_, 2.0);
    private_nh_.param("startup_speed", startup_speed_, 0.45);
    private_nh_.param("startup_max_angular", startup_max_angular_, 0.10);

    private_nh_.param("roi_y_start_ratio", roi_y_start_ratio_, 0.42);
    private_nh_.param("white_s_max", white_s_max_, 80);
    private_nh_.param("white_v_min", white_v_min_, 170);
    private_nh_.param("morph_kernel_size", morph_kernel_size_, 3);
    private_nh_.param("min_component_area", min_component_area_, 80);

    private_nh_.param("sample_row_count", sample_row_count_, 16);
    private_nh_.param("sample_y_min_ratio", sample_y_min_ratio_, 0.12);
    private_nh_.param("sample_y_max_ratio", sample_y_max_ratio_, 0.94);
    private_nh_.param("min_line_segment_width_px",
                      min_line_segment_width_px_, 2);
    private_nh_.param("max_line_segment_width_px",
                      max_line_segment_width_px_, 85);
    private_nh_.param("candidate_search_radius_px",
                      candidate_search_radius_px_, 170.0);
    private_nh_.param("min_line_points", min_line_points_, 5);
    private_nh_.param("fit_inlier_threshold_px",
                      fit_inlier_threshold_px_, 18.0);
    private_nh_.param("max_fit_rmse_px", max_fit_rmse_px_, 16.0);
    private_nh_.param("control_fit_y_min_ratio",
                      control_fit_y_min_ratio_, 0.42);

    private_nh_.param("target_right_x", target_right_x_, 200.0);
    private_nh_.param("right_control_y_ratio",
                      right_control_y_ratio_, 0.65);
    private_nh_.param("right_far_y_ratio", right_far_y_ratio_, 0.28);
    private_nh_.param("reference_heading_alpha",
                      reference_heading_alpha_, 0.06);
    private_nh_.param("min_tracking_confidence",
                      min_tracking_confidence_, 0.32);
    private_nh_.param("line_lock_frames", line_lock_frames_, 4);

    private_nh_.param("base_speed", base_speed_, 0.20);
    private_nh_.param("curve_speed", curve_speed_, 0.12);
    private_nh_.param("low_confidence_speed", low_confidence_speed_, 0.08);
    private_nh_.param("lateral_kp", lateral_kp_, 0.0030);
    private_nh_.param("lateral_kd", lateral_kd_, 0.0007);
    private_nh_.param("heading_kp", heading_kp_, 0.0060);
    private_nh_.param("lateral_deadband_px", lateral_deadband_px_, 5.0);
    private_nh_.param("heading_deadband_deg", heading_deadband_deg_, 2.0);
    private_nh_.param("curve_lateral_threshold_px",
                      curve_lateral_threshold_px_, 32.0);
    private_nh_.param("curve_heading_threshold_deg",
                      curve_heading_threshold_deg_, 7.0);
    private_nh_.param("max_left_angular_speed",
                      max_left_angular_speed_, 0.28);
    private_nh_.param("max_right_angular_speed",
                      max_right_angular_speed_, 0.32);
    private_nh_.param("angular_filter_old_weight",
                      angular_filter_old_weight_, 0.65);
    private_nh_.param("max_angular_step", max_angular_step_, 0.05);

    private_nh_.param("right_warning_error_px",
                      right_warning_error_px_, -48.0);
    private_nh_.param("right_hard_error_px",
                      right_hard_error_px_, -82.0);
    private_nh_.param("right_guard_speed", right_guard_speed_, 0.06);
    private_nh_.param("right_guard_away_angular",
                      right_guard_away_angular_, 0.10);

    private_nh_.param("lost_hold_timeout", lost_hold_timeout_, 0.10);
    private_nh_.param("lost_stop_timeout", lost_stop_timeout_, 0.30);
    private_nh_.param("search_rotate_time", search_rotate_time_, 0.80);
    private_nh_.param("search_right_angular",
                      search_right_angular_, -0.10);

    private_nh_.param("corner_confirm_frames", corner_confirm_frames_, 5);
    private_nh_.param("corner_confirm_timeout", corner_confirm_timeout_, 0.80);
    private_nh_.param("corner_min_arm_points", corner_min_arm_points_, 4);
    private_nh_.param("corner_min_angle_deg", corner_min_angle_deg_, 20.0);
    private_nh_.param("corner_max_angle_deg", corner_max_angle_deg_, 75.0);
    private_nh_.param("corner_vertex_y_min_ratio",
                      corner_vertex_y_min_ratio_, 0.15);
    private_nh_.param("corner_vertex_y_max_ratio",
                      corner_vertex_y_max_ratio_, 0.80);
    private_nh_.param("corner_forward_distance_m",
                      corner_forward_distance_m_, 0.20);
    private_nh_.param("corner_forward_speed", corner_forward_speed_, 0.10);
    private_nh_.param("corner_forward_max_angular",
                      corner_forward_max_angular_, 0.07);
    private_nh_.param("corner_advance_lost_timeout",
                      corner_advance_lost_timeout_, 0.15);
    private_nh_.param("corner_stop_hold", corner_stop_hold_, 0.50);
    private_nh_.param("turn_right_angular_speed",
                      turn_right_angular_speed_, -0.34);
    private_nh_.param("turn_right_min_time", turn_right_min_time_, 1.80);
    private_nh_.param("reacquire_angular_speed",
                      reacquire_angular_speed_, -0.16);
    private_nh_.param("reacquire_timeout", reacquire_timeout_, 2.5);
    private_nh_.param("reacquire_frames", reacquire_frames_, 3);
    private_nh_.param("align_speed", align_speed_, 0.08);
    private_nh_.param("align_timeout", align_timeout_, 4.0);
    private_nh_.param("align_lateral_tolerance_px",
                      align_lateral_tolerance_px_, 12.0);
    private_nh_.param("align_heading_tolerance_deg",
                      align_heading_tolerance_deg_, 6.0);
    private_nh_.param("align_stable_frames", align_stable_frames_, 4);
    private_nh_.param("corner_cooldown_seconds",
                      corner_cooldown_seconds_, 1.5);
    private_nh_.param("required_corner_count", required_corner_count_, 2);

    private_nh_.param("end_roi_y_start_ratio",
                      end_roi_y_start_ratio_, 0.84);
    private_nh_.param("end_min_width_ratio", end_min_width_ratio_, 0.45);
    private_nh_.param("end_confirm_frames", end_confirm_frames_, 3);
    private_nh_.param("end_stop_hold", end_stop_hold_, 1.0);
    private_nh_.param("end_turn_left_angle_deg",
                      end_turn_left_angle_deg_, 10.0);
    private_nh_.param("end_turn_left_angular_speed",
                      end_turn_left_angular_speed_, 0.50);
    private_nh_.param("end_forward_distance_m",
                      end_forward_distance_m_, 0.65);
    private_nh_.param("end_forward_speed", end_forward_speed_, 0.17);

    sample_row_count_ = std::max(8, sample_row_count_);
    min_line_points_ = std::max(3, min_line_points_);
    line_lock_frames_ = std::max(2, line_lock_frames_);
    corner_confirm_frames_ = std::max(2, corner_confirm_frames_);
    reacquire_frames_ = std::max(2, reacquire_frames_);
    align_stable_frames_ = std::max(2, align_stable_frames_);
    end_confirm_frames_ = std::max(2, end_confirm_frames_);
    search_rotate_time_ = std::max(0.0, search_rotate_time_);
    morph_kernel_size_ = std::max(1, morph_kernel_size_);
    if (morph_kernel_size_ % 2 == 0)
      ++morph_kernel_size_;
    right_control_y_ratio_ =
        clampDouble(right_control_y_ratio_, 0.45, 0.85);
    right_far_y_ratio_ =
        clampDouble(right_far_y_ratio_, 0.10,
                    right_control_y_ratio_ - 0.10);
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
      ROS_WARN_THROTTLE(2.0, "cv_bridge failed: %s", exc.what());
      return;
    }
    if (frame.empty())
      return;
    if (frame.rows != kImageRows || frame.cols != kImageCols)
      cv::resize(frame, frame, cv::Size(kImageCols, kImageRows),
                 0.0, 0.0, cv::INTER_AREA);

    const ros::Time now = ros::Time::now();
    const int roi_y0 =
        clampInt(static_cast<int>(frame.rows * roi_y_start_ratio_),
                 0, frame.rows - 2);
    const cv::Mat roi = frame(cv::Range(roi_y0, frame.rows), cv::Range::all());
    const cv::Mat mask = extractWhiteMask(roi);
    VisionResult vision = analyzeMask(mask);
    vision.end_line = detectEndLine(mask);

    switch (state_)
    {
      case State::Idle:
        setStatus("idle");
        publishStop();
        break;

      case State::StartupForward:
        handleStartup(vision, now);
        break;

      case State::SearchRightLine:
        handleSearch(vision, now);
        break;

      case State::Follow:
        handleFollow(vision, now);
        break;

      case State::CornerConfirm:
        handleCornerConfirm(vision, now);
        break;

      case State::CornerAdvance:
        handleCornerAdvance(vision, now);
        break;

      case State::CornerStop:
        handleCornerStop(now);
        break;

      case State::TurnRight:
        handleTurnRight(now);
        break;

      case State::ReacquireRightLine:
        handleReacquire(vision, now);
        break;

      case State::AlignRightLine:
        handleAlign(vision, now);
        break;

      case State::EndDetected:
        handleEndDetected(now);
        break;

      case State::ParkingTurnLeft:
        handleParkingTurnLeft(now);
        break;

      case State::ParkingForward:
        handleParkingForward(now);
        break;

      case State::FinalStop:
        publishStop();
        setStatus("stable_right_final_stop");
        if ((now - state_start_time_).toSec() >= end_stop_hold_)
          enterState(State::Finish, now);
        break;

      case State::Finish:
        setStatus("stable_right_finish");
        publishStop();
        break;
    }

    publishDebug(frame, roi_y0, vision, now, msg->header);
    publishDebugInfo(vision, now);
    publishStatus();
  }

  cv::Mat extractWhiteMask(const cv::Mat& roi) const
  {
    cv::Mat blurred;
    cv::GaussianBlur(roi, blurred, cv::Size(5, 5), 0.0);
    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(0, 0, white_v_min_),
                cv::Scalar(180, white_s_max_, 255), mask);

    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(morph_kernel_size_, morph_kernel_size_));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int label = 1; label < component_count; ++label)
    {
      if (stats.at<int>(label, cv::CC_STAT_AREA) < min_component_area_)
        continue;
      filtered.setTo(255, labels == label);
    }
    return filtered;
  }

  std::vector<Segment> findSegments(const cv::Mat& row) const
  {
    std::vector<Segment> segments;
    int start = -1;
    for (int x = 0; x < row.cols; ++x)
    {
      const bool active = row.at<uchar>(0, x) != 0;
      if (active && start < 0)
        start = x;
      else if (!active && start >= 0)
      {
        segments.push_back(Segment{start, x - 1, x - start});
        start = -1;
      }
    }
    if (start >= 0)
      segments.push_back(
          Segment{start, row.cols - 1, row.cols - start});
    return segments;
  }

  std::vector<stable_track::SamplePoint> extractRightBoundarySamples(
      const cv::Mat& mask) const
  {
    std::vector<stable_track::SamplePoint> right_points;
    for (int index = 0; index < sample_row_count_; ++index)
    {
      const double t = sample_row_count_ <= 1
                           ? 0.0
                           : static_cast<double>(index) /
                                 static_cast<double>(sample_row_count_ - 1);
      const double y_ratio =
          sample_y_min_ratio_ +
          t * (sample_y_max_ratio_ - sample_y_min_ratio_);
      const int y = clampInt(
          static_cast<int>(std::round(y_ratio * (mask.rows - 1))),
          0, mask.rows - 1);

      std::vector<double> candidates;
      for (const Segment& segment : findSegments(mask.row(y)))
      {
        if (segment.width < min_line_segment_width_px_ ||
            segment.width > max_line_segment_width_px_)
          continue;
        candidates.push_back(segment.center());
      }
      if (candidates.empty())
        continue;
      std::sort(candidates.begin(), candidates.end());

      int selected_index = static_cast<int>(candidates.size()) - 1;
      if (previous_right_model_.valid)
      {
        const double predicted =
            stable_track::xAt(previous_right_model_, y);
        double best_distance = candidate_search_radius_px_;
        int predicted_index = -1;
        for (int candidate_index = 0;
             candidate_index < static_cast<int>(candidates.size());
             ++candidate_index)
        {
          const double distance =
              std::fabs(candidates[candidate_index] - predicted);
          if (distance < best_distance)
          {
            best_distance = distance;
            predicted_index = candidate_index;
          }
        }
        if (predicted_index >= 0)
          selected_index = predicted_index;
      }
      right_points.push_back(
          stable_track::SamplePoint{candidates[selected_index],
                                    static_cast<double>(y)});
    }
    return right_points;
  }

  stable_track::LineModel robustControlFit(
      const std::vector<stable_track::SamplePoint>& all_points,
      int mask_height) const
  {
    std::vector<stable_track::SamplePoint> control_points;
    const double y_min = control_fit_y_min_ratio_ * mask_height;
    for (const stable_track::SamplePoint& point : all_points)
    {
      if (point.y >= y_min)
        control_points.push_back(point);
    }
    if (static_cast<int>(control_points.size()) < min_line_points_)
      control_points = all_points;
    stable_track::LineModel first =
        stable_track::fitLineModel(control_points);
    if (!first.valid)
      return first;

    std::vector<stable_track::SamplePoint> inliers;
    for (const stable_track::SamplePoint& point : control_points)
    {
      if (std::fabs(point.x - stable_track::xAt(first, point.y)) <=
          fit_inlier_threshold_px_)
        inliers.push_back(point);
    }
    if (static_cast<int>(inliers.size()) < min_line_points_)
      return first;
    return stable_track::fitLineModel(inliers);
  }

  double lineConfidence(const stable_track::LineModel& line,
                        int raw_point_count) const
  {
    if (!line.valid || line.point_count < min_line_points_ ||
        line.rmse > max_fit_rmse_px_)
      return 0.0;
    const double coverage = clampDouble(
        static_cast<double>(raw_point_count) /
            static_cast<double>(sample_row_count_),
        0.0, 1.0);
    const double residual =
        clampDouble(1.0 - line.rmse / max_fit_rmse_px_, 0.0, 1.0);
    return coverage * residual;
  }

  VisionResult analyzeMask(const cv::Mat& mask)
  {
    VisionResult result;
    result.right_points = extractRightBoundarySamples(mask);
    result.right_line =
        robustControlFit(result.right_points, mask.rows);
    result.right_confidence =
        lineConfidence(result.right_line,
                       static_cast<int>(result.right_points.size()));
    if (result.right_confidence <= 0.0)
      result.right_line.valid = false;

    result.right = stable_track::observeRightBoundary(
        result.right_line,
        mask.rows * right_control_y_ratio_,
        mask.rows * right_far_y_ratio_,
        target_right_x_);
    result.right.confidence = result.right_confidence;
    if (result.right_line.valid)
    {
      previous_right_model_ = result.right_line;
    }

    result.corner = stable_track::detectKink(
        result.right_points, corner_min_arm_points_,
        corner_min_angle_deg_, corner_max_angle_deg_);
    if (result.corner.detected)
    {
      const double vertex_ratio =
          result.corner.vertex.y /
          std::max(1.0, static_cast<double>(mask.rows - 1));
      if (vertex_ratio < corner_vertex_y_min_ratio_ ||
          vertex_ratio > corner_vertex_y_max_ratio_)
        result.corner.detected = false;
    }
    return result;
  }

  EndLineResult detectEndLine(const cv::Mat& mask) const
  {
    EndLineResult result;
    const double mask_start_ratio = clampDouble(
        (end_roi_y_start_ratio_ - roi_y_start_ratio_) /
            std::max(1e-6, 1.0 - roi_y_start_ratio_),
        0.0, 1.0);
    const int y0 = clampInt(
        static_cast<int>(mask.rows * mask_start_ratio),
        0, mask.rows - 1);
    const int minimum_width =
        static_cast<int>(mask.cols * end_min_width_ratio_);
    int supporting_rows = 0;
    for (int y = y0; y < mask.rows; y += 2)
    {
      for (const Segment& segment : findSegments(mask.row(y)))
      {
        if (segment.width < minimum_width)
          continue;
        ++supporting_rows;
        const double ratio =
            static_cast<double>(segment.width) /
            static_cast<double>(mask.cols);
        if (ratio > result.width_ratio)
        {
          result.width_ratio = ratio;
          result.y_ratio =
              roi_y_start_ratio_ +
              (1.0 - roi_y_start_ratio_) *
                  static_cast<double>(y) /
                  std::max(1.0, static_cast<double>(mask.rows));
        }
      }
    }
    result.detected = supporting_rows >= 3;
    return result;
  }

  bool trackingUsable(const VisionResult& vision) const
  {
    return vision.right.valid &&
           vision.right_line.valid &&
           vision.right_confidence >= min_tracking_confidence_;
  }

  geometry_msgs::Twist makeRightCommand(const VisionResult& vision,
                                        double speed_limit)
  {
    geometry_msgs::Twist cmd;
    const stable_track::RightObservation& right = vision.right;
    double lateral = right.lateral_error_px;
    double heading =
        right.heading_angle_deg - reference_right_heading_deg_;
    if (std::fabs(lateral) <= lateral_deadband_px_)
      lateral = 0.0;
    if (std::fabs(heading) <= heading_deadband_deg_)
      heading = 0.0;

    const double derivative = lateral - last_lateral_error_;
    last_lateral_error_ = lateral;
    double target_angular =
        lateral_kp_ * lateral +
        lateral_kd_ * derivative +
        heading_kp_ * heading;
    target_angular = clampDouble(
        target_angular, -max_right_angular_speed_,
        max_left_angular_speed_);
    const double filtered_target =
        angular_filter_old_weight_ * filtered_angular_ +
        (1.0 - angular_filter_old_weight_) * target_angular;
    filtered_angular_ += clampDouble(
        filtered_target - filtered_angular_,
        -max_angular_step_, max_angular_step_);
    filtered_angular_ = clampDouble(
        filtered_angular_, -max_right_angular_speed_,
        max_left_angular_speed_);

    const bool curve =
        std::fabs(lateral) > curve_lateral_threshold_px_ ||
        std::fabs(heading) >
            curve_heading_threshold_deg_;
    double speed = curve ? curve_speed_ : base_speed_;
    if (right.confidence < 0.55)
      speed = std::min(speed, low_confidence_speed_);
    speed = std::min(speed, speed_limit);

    if (right.lateral_error_px <= right_hard_error_px_)
    {
      cmd.linear.x = 0.0;
      cmd.angular.z = std::max(
          right_guard_away_angular_, filtered_angular_);
      setStatus("stable_right_hard_guard_recover");
      return cmd;
    }
    if (right.lateral_error_px <= right_warning_error_px_)
    {
      cmd.linear.x = std::min(speed, right_guard_speed_);
      cmd.angular.z = std::max(
          right_guard_away_angular_, filtered_angular_);
      setStatus("stable_right_warning_guard");
      return cmd;
    }
    cmd.linear.x = speed;
    cmd.angular.z = filtered_angular_;
    return cmd;
  }

  bool publishTrackedCommand(const VisionResult& vision,
                             const ros::Time& now,
                             double speed_limit,
                             const std::string& normal_status)
  {
    if (trackingUsable(vision))
    {
      last_confident_line_time_ = now;
      setStatus(normal_status);
      publishCmd(makeRightCommand(vision, speed_limit));
      return true;
    }

    const double lost_time =
        (now - last_confident_line_time_).toSec();
    if (lost_time < lost_hold_timeout_)
    {
      geometry_msgs::Twist cmd;
      cmd.linear.x = 0.0;
      cmd.angular.z = 0.0;
      setStatus("stable_right_line_weak_stop");
      publishCmd(cmd);
    }
    else
    {
      setStatus("stable_right_line_lost_stop");
      publishStop();
    }

    if (lost_time >= lost_stop_timeout_)
    {
      enterState(State::SearchRightLine, now);
      setStatus("stable_right_search_after_loss");
      publishStop();
    }
    return false;
  }

  void handleStartup(const VisionResult& vision, const ros::Time& now)
  {
    const double elapsed = (now - start_time_).toSec();
    if (elapsed >= startup_time_)
    {
      if (trackingUsable(vision))
      {
        last_confident_line_time_ = now;
        updateReferenceHeading(vision);
        enterState(State::Follow, now);
        setStatus("stable_right_tracking_after_startup");
      }
      else
      {
        enterState(State::SearchRightLine, now);
        setStatus("stable_right_search_after_startup");
      }
      publishStop();
      return;
    }

    geometry_msgs::Twist cmd;
    cmd.linear.x = startup_speed_;
    if (trackingUsable(vision))
    {
      last_confident_line_time_ = now;
      updateReferenceHeading(vision);
      const double target =
          0.0012 * vision.right.lateral_error_px +
          0.0020 *
              (vision.right.heading_angle_deg -
               reference_right_heading_deg_);
      cmd.angular.z = clampDouble(
          target, -startup_max_angular_, startup_max_angular_);
      if (vision.right.lateral_error_px <= right_hard_error_px_)
      {
        cmd.linear.x = 0.0;
        cmd.angular.z = right_guard_away_angular_;
        setStatus("stable_right_startup_guard");
      }
      else
      {
        setStatus("stable_right_startup_forward");
      }
    }
    else
    {
      setStatus("stable_right_startup_forward_vision_search");
    }
    publishCmd(cmd);
  }

  void handleSearch(const VisionResult& vision, const ros::Time& now)
  {
    if (trackingUsable(vision))
    {
      ++line_lock_count_;
      setStatus("stable_right_search_locking");
      publishStop();
      if (line_lock_count_ >= line_lock_frames_)
      {
        last_confident_line_time_ = now;
        updateReferenceHeading(vision);
        resetController();
        enterState(State::Follow, now);
        setStatus("stable_right_tracking");
      }
      return;
    }
    line_lock_count_ = 0;

    const double elapsed = (now - state_start_time_).toSec();
    geometry_msgs::Twist cmd;
    if (elapsed < search_rotate_time_)
    {
      cmd.angular.z = search_right_angular_;
      setStatus("stable_right_search_single_right_pulse");
    }
    else
    {
      cmd.angular.z = 0.0;
      setStatus("stable_right_search_waiting_stopped");
    }
    publishCmd(cmd);
  }

  void handleFollow(const VisionResult& vision, const ros::Time& now)
  {
    if (trackingUsable(vision) && !vision.corner.detected)
      updateReferenceHeading(vision);

    const bool parking_enabled = stable_track::parkingEnabled(
        completed_corner_count_, required_corner_count_);
    const bool end_confirmed = end_line_debouncer_.update(
        vision.end_line.detected, parking_enabled);
    if (end_confirmed)
    {
      enterState(State::EndDetected, now);
      setStatus("stable_right_end_detected");
      hardStop();
      return;
    }

    if (now >= corner_cooldown_until_ &&
        completed_corner_count_ < required_corner_count_ &&
        vision.corner.detected)
    {
      corner_confirm_count_ = 1;
      enterState(State::CornerConfirm, now);
      setStatus("stable_right_corner_confirm_1");
      publishTrackedCommand(
          vision, now, curve_speed_,
          "stable_right_corner_confirm_1");
      return;
    }
    publishTrackedCommand(
        vision, now, base_speed_, "stable_right_tracking");
  }

  void handleCornerConfirm(const VisionResult& vision,
                           const ros::Time& now)
  {
    const double elapsed = (now - state_start_time_).toSec();
    if (!vision.corner.detected)
    {
      corner_confirm_count_ = 0;
      enterState(State::Follow, now);
      publishTrackedCommand(
          vision, now, curve_speed_,
          "stable_right_corner_rejected");
      return;
    }

    ++corner_confirm_count_;
    std::ostringstream status;
    status << "stable_right_corner_confirm_"
           << corner_confirm_count_;
    if (!publishTrackedCommand(
            vision, now, curve_speed_, status.str()))
      return;

    if (corner_confirm_count_ >= corner_confirm_frames_)
    {
      enterState(State::CornerAdvance, now);
      setStatus("stable_right_corner_advance_20cm");
      handleCornerAdvance(vision, now);
      return;
    }
    if (elapsed >= corner_confirm_timeout_)
    {
      corner_confirm_count_ = 0;
      enterState(State::Follow, now);
    }
  }

  void handleCornerAdvance(const VisionResult& vision,
                           const ros::Time& now)
  {
    const double duration = stable_track::advanceDuration(
        corner_forward_distance_m_, corner_forward_speed_);
    if ((now - state_start_time_).toSec() >= duration)
    {
      enterState(State::CornerStop, now);
      setStatus("stable_right_corner_stop");
      hardStop();
      return;
    }

    geometry_msgs::Twist cmd;
    cmd.linear.x = corner_forward_speed_;
    if (trackingUsable(vision))
    {
      last_confident_line_time_ = now;
      const double correction =
          0.0010 * vision.right.lateral_error_px +
          0.0015 *
              (vision.right.heading_angle_deg -
               reference_right_heading_deg_);
      cmd.angular.z = clampDouble(
          correction, -corner_forward_max_angular_,
          corner_forward_max_angular_);
      if (vision.right.lateral_error_px <= right_hard_error_px_)
      {
        enterState(State::CornerStop, now);
        setStatus("stable_right_corner_advance_guard_stop");
        hardStop();
        return;
      }
    }
    else if ((now - last_confident_line_time_).toSec() >=
             corner_advance_lost_timeout_)
    {
      enterState(State::CornerStop, now);
      setStatus("stable_right_corner_advance_line_lost_stop");
      hardStop();
      return;
    }
    setStatus("stable_right_corner_advance_20cm");
    publishCmd(cmd);
  }

  void handleCornerStop(const ros::Time& now)
  {
    setStatus("stable_right_corner_stop");
    hardStop();
    if ((now - state_start_time_).toSec() >= corner_stop_hold_)
    {
      enterState(State::TurnRight, now);
      setStatus("stable_right_turn_right");
    }
  }

  void handleTurnRight(const ros::Time& now)
  {
    geometry_msgs::Twist cmd;
    cmd.angular.z = turn_right_angular_speed_;
    setStatus("stable_right_turn_right");
    publishCmd(cmd);
    if ((now - state_start_time_).toSec() >= turn_right_min_time_)
    {
      enterState(State::ReacquireRightLine, now);
      setStatus("stable_right_reacquire_right_line");
    }
  }

  void handleReacquire(const VisionResult& vision,
                       const ros::Time& now)
  {
    const bool candidate = trackingUsable(vision);
    if (candidate)
      ++reacquire_stable_count_;
    else
      reacquire_stable_count_ = 0;

    if (reacquire_stable_count_ >= reacquire_frames_)
    {
      enterState(State::AlignRightLine, now);
      setStatus("stable_right_align_right_line");
      hardStop();
      return;
    }

    geometry_msgs::Twist cmd;
    if ((now - state_start_time_).toSec() < reacquire_timeout_)
    {
      cmd.angular.z = reacquire_angular_speed_;
      setStatus("stable_right_reacquire_turn_right");
    }
    else
    {
      cmd.angular.z = 0.0;
      setStatus("stable_right_reacquire_waiting_stopped");
    }
    publishCmd(cmd);
  }

  void handleAlign(const VisionResult& vision, const ros::Time& now)
  {
    if (!trackingUsable(vision))
    {
      align_stable_count_ = 0;
      setStatus("stable_right_align_right_line_lost_stop");
      publishStop();
      if ((now - state_start_time_).toSec() >= align_timeout_)
      {
        enterState(State::ReacquireRightLine, now);
        setStatus("stable_right_align_retry_reacquire");
      }
      return;
    }

    const double heading_delta =
        vision.right.heading_angle_deg -
        reference_right_heading_deg_;
    const bool aligned =
        std::fabs(vision.right.lateral_error_px) <=
            align_lateral_tolerance_px_ &&
        std::fabs(heading_delta) <=
            align_heading_tolerance_deg_;
    if (aligned)
      ++align_stable_count_;
    else
      align_stable_count_ = 0;

    publishTrackedCommand(
        vision, now, align_speed_, "stable_right_align_right_line");
    const bool alignment_timed_out =
        (now - state_start_time_).toSec() >= align_timeout_;
    if (align_stable_count_ < align_stable_frames_ &&
        !alignment_timed_out)
      return;

    ++completed_corner_count_;
    corner_confirm_count_ = 0;
    corner_cooldown_until_ =
        now + ros::Duration(corner_cooldown_seconds_);
    resetController();
    enterState(State::Follow, now);
    setStatus(alignment_timed_out
                  ? "stable_right_align_timeout_follow_right"
                  : "stable_right_aligned_right_follow");
    ROS_INFO("right corner completed: %d/%d",
             completed_corner_count_, required_corner_count_);
  }

  void handleEndDetected(const ros::Time& now)
  {
    setStatus("stable_right_end_detected");
    hardStop();
    if ((now - state_start_time_).toSec() >= end_stop_hold_)
    {
      enterState(State::ParkingTurnLeft, now);
      setStatus("stable_right_parking_turn_left");
    }
  }

  void handleParkingTurnLeft(const ros::Time& now)
  {
    const double duration =
        (end_turn_left_angle_deg_ * stable_track::kPi / 180.0) /
        std::max(1e-6, end_turn_left_angular_speed_);
    if ((now - state_start_time_).toSec() >= duration)
    {
      enterState(State::ParkingForward, now);
      hardStop();
      return;
    }
    geometry_msgs::Twist cmd;
    cmd.angular.z = end_turn_left_angular_speed_;
    setStatus("stable_right_parking_turn_left");
    publishCmd(cmd);
  }

  void handleParkingForward(const ros::Time& now)
  {
    const double duration = stable_track::advanceDuration(
        end_forward_distance_m_, end_forward_speed_);
    if ((now - state_start_time_).toSec() >= duration)
    {
      enterState(State::FinalStop, now);
      hardStop();
      return;
    }
    geometry_msgs::Twist cmd;
    cmd.linear.x = end_forward_speed_;
    setStatus("stable_right_parking_forward");
    publishCmd(cmd);
  }

  void enterState(State state, const ros::Time& now)
  {
    state_ = state;
    state_start_time_ = now;
    if (state == State::SearchRightLine)
      line_lock_count_ = 0;
    if (state == State::ReacquireRightLine)
      reacquire_stable_count_ = 0;
    if (state == State::AlignRightLine)
      align_stable_count_ = 0;
  }

  void resetController()
  {
    last_lateral_error_ = 0.0;
    filtered_angular_ = 0.0;
    last_angular_ = 0.0;
  }

  void updateReferenceHeading(const VisionResult& vision)
  {
    if (!trackingUsable(vision))
      return;
    if (!reference_heading_initialized_)
    {
      reference_right_heading_deg_ =
          vision.right.heading_angle_deg;
      reference_heading_initialized_ = true;
      return;
    }
    reference_right_heading_deg_ =
        (1.0 - reference_heading_alpha_) *
            reference_right_heading_deg_ +
        reference_heading_alpha_ *
            vision.right.heading_angle_deg;
  }

  void publishCmd(const geometry_msgs::Twist& cmd)
  {
    last_linear_ = cmd.linear.x;
    last_angular_ = cmd.angular.z;
    cmd_pub_.publish(cmd);
  }

  void publishStop()
  {
    geometry_msgs::Twist cmd;
    publishCmd(cmd);
  }

  void hardStop()
  {
    geometry_msgs::Twist cmd;
    last_linear_ = 0.0;
    last_angular_ = 0.0;
    filtered_angular_ = 0.0;
    for (int index = 0; index < 4; ++index)
      cmd_pub_.publish(cmd);
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

  void drawLine(cv::Mat& debug,
                const stable_track::LineModel& line,
                int roi_y0, int roi_height,
                const cv::Scalar& color) const
  {
    if (!line.valid)
      return;
    const int y1 = static_cast<int>(roi_height * 0.18);
    const int y2 = static_cast<int>(roi_height * 0.95);
    const int x1 = clampInt(
        static_cast<int>(std::round(stable_track::xAt(line, y1))),
        0, debug.cols - 1);
    const int x2 = clampInt(
        static_cast<int>(std::round(stable_track::xAt(line, y2))),
        0, debug.cols - 1);
    cv::line(debug, cv::Point(x1, roi_y0 + y1),
             cv::Point(x2, roi_y0 + y2), color, 3);
  }

  void publishDebug(const cv::Mat& frame, int roi_y0,
                    const VisionResult& vision,
                    const ros::Time& now,
                    const std_msgs::Header& header)
  {
    cv::Mat debug = frame.clone();
    const int roi_height = frame.rows - roi_y0;
    cv::rectangle(debug, cv::Rect(0, roi_y0, frame.cols, roi_height),
                  cv::Scalar(255, 180, 0), 1);
    cv::line(debug, cv::Point(static_cast<int>(target_right_x_), roi_y0),
             cv::Point(static_cast<int>(target_right_x_), frame.rows - 1),
             cv::Scalar(255, 0, 255), 1);

    for (const stable_track::SamplePoint& point : vision.right_points)
      cv::circle(debug,
                 cv::Point(static_cast<int>(point.x),
                           roi_y0 + static_cast<int>(point.y)),
                 3, cv::Scalar(0, 0, 255), -1);
    drawLine(debug, vision.right_line, roi_y0, roi_height,
             cv::Scalar(0, 0, 255));

    if (vision.corner.detected)
    {
      cv::circle(
          debug,
          cv::Point(static_cast<int>(vision.corner.vertex.x),
                    roi_y0 +
                        static_cast<int>(vision.corner.vertex.y)),
          10, cv::Scalar(0, 165, 255), 3);
      cv::putText(
          debug, "RIGHT CORNER", cv::Point(360, 30),
          cv::FONT_HERSHEY_SIMPLEX, 0.7,
          cv::Scalar(0, 165, 255), 2);
    }
    if (vision.end_line.detected)
      cv::putText(debug, "END RAW", cv::Point(480, 60),
                  cv::FONT_HERSHEY_SIMPLEX, 0.65,
                  cv::Scalar(0, 0, 255), 2);

    std::ostringstream line1;
    line1 << "state=" << status_ << " cmd=("
          << std::fixed << std::setprecision(2)
          << last_linear_ << "," << last_angular_
          << ") corners=" << completed_corner_count_
          << "/" << required_corner_count_;
    cv::putText(debug, line1.str(), cv::Point(10, 205),
                cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(0, 255, 0), 2);

    std::ostringstream line2;
    line2 << "RIGHT ONLY conf=" << std::fixed
          << std::setprecision(2)
          << vision.right_confidence
          << " x=" << std::setprecision(1)
          << vision.right.detected_x
          << " target=" << target_right_x_;
    cv::putText(debug, line2.str(), cv::Point(10, 230),
                cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(0, 220, 255), 2);

    std::ostringstream line3;
    line3 << "lat=" << std::fixed << std::setprecision(1)
          << vision.right.lateral_error_px
          << " head=" << vision.right.heading_angle_deg
          << "deg ref=" << reference_right_heading_deg_
          << " corner=" << boolText(vision.corner.detected)
          << "(" << vision.corner.angle_deg << "deg)";
    cv::putText(debug, line3.str(), cv::Point(10, 255),
                cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(255, 255, 255), 2);

    try
    {
      sensor_msgs::ImagePtr output =
          cv_bridge::CvImage(header, "bgr8", debug).toImageMsg();
      output->header.stamp = now;
      debug_image_pub_.publish(output);
    }
    catch (const cv_bridge::Exception& exc)
    {
      ROS_WARN_THROTTLE(2.0, "debug image failed: %s", exc.what());
    }
  }

  void publishDebugInfo(const VisionResult& vision,
                        const ros::Time& now)
  {
    std::ostringstream stream;
    stream << "state=" << status_
           << " right_found=" << boolText(vision.right_line.valid)
           << " right_only=1"
           << " right_conf=" << std::fixed
           << std::setprecision(3) << vision.right_confidence
           << " right_x=" << vision.right.detected_x
           << " lateral_px=" << std::setprecision(1)
           << vision.right.lateral_error_px
           << " heading_deg=" << vision.right.heading_angle_deg
           << " reference_heading_deg="
           << reference_right_heading_deg_
           << " corner=" << boolText(vision.corner.detected)
           << " corner_angle_deg=" << vision.corner.angle_deg
           << " corner_confirm=" << corner_confirm_count_
           << " completed_corners=" << completed_corner_count_
           << " end_raw=" << boolText(vision.end_line.detected)
           << " end_confirm=" << end_line_debouncer_.count()
           << " end_width=" << vision.end_line.width_ratio
           << " state_elapsed="
           << (now - state_start_time_).toSec()
           << " cmd_linear=" << last_linear_
           << " cmd_angular=" << last_angular_;
    std_msgs::String msg;
    msg.data = stream.str();
    debug_info_pub_.publish(msg);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Publisher cmd_pub_;
  ros::Publisher status_pub_;
  ros::Publisher debug_info_pub_;
  ros::Publisher debug_image_pub_;
  ros::Subscriber image_sub_;

  std::string image_topic_;
  std::string cmd_vel_topic_;
  std::string status_topic_;
  std::string debug_image_topic_;
  std::string debug_info_topic_;
  std::string status_;

  State state_ = State::Idle;
  ros::Time start_time_;
  ros::Time state_start_time_;
  ros::Time last_confident_line_time_;
  ros::Time corner_cooldown_until_;

  bool auto_start_ = true;
  double startup_time_ = 2.0;
  double startup_speed_ = 0.45;
  double startup_max_angular_ = 0.10;

  double roi_y_start_ratio_ = 0.42;
  int white_s_max_ = 80;
  int white_v_min_ = 170;
  int morph_kernel_size_ = 3;
  int min_component_area_ = 80;

  int sample_row_count_ = 16;
  double sample_y_min_ratio_ = 0.12;
  double sample_y_max_ratio_ = 0.94;
  int min_line_segment_width_px_ = 2;
  int max_line_segment_width_px_ = 85;
  double candidate_search_radius_px_ = 170.0;
  int min_line_points_ = 5;
  double fit_inlier_threshold_px_ = 18.0;
  double max_fit_rmse_px_ = 16.0;
  double control_fit_y_min_ratio_ = 0.42;

  double target_right_x_ = 200.0;
  double right_control_y_ratio_ = 0.65;
  double right_far_y_ratio_ = 0.28;
  double reference_heading_alpha_ = 0.06;
  double reference_right_heading_deg_ = 0.0;
  bool reference_heading_initialized_ = false;
  double min_tracking_confidence_ = 0.32;
  int line_lock_frames_ = 4;

  double base_speed_ = 0.20;
  double curve_speed_ = 0.12;
  double low_confidence_speed_ = 0.08;
  double lateral_kp_ = 0.0030;
  double lateral_kd_ = 0.0007;
  double heading_kp_ = 0.0060;
  double lateral_deadband_px_ = 5.0;
  double heading_deadband_deg_ = 2.0;
  double curve_lateral_threshold_px_ = 32.0;
  double curve_heading_threshold_deg_ = 7.0;
  double max_left_angular_speed_ = 0.28;
  double max_right_angular_speed_ = 0.32;
  double angular_filter_old_weight_ = 0.65;
  double max_angular_step_ = 0.05;

  double right_warning_error_px_ = -48.0;
  double right_hard_error_px_ = -82.0;
  double right_guard_speed_ = 0.06;
  double right_guard_away_angular_ = 0.10;

  double lost_hold_timeout_ = 0.10;
  double lost_stop_timeout_ = 0.30;
  double search_rotate_time_ = 0.80;
  double search_right_angular_ = -0.10;

  int corner_confirm_frames_ = 5;
  double corner_confirm_timeout_ = 0.80;
  int corner_min_arm_points_ = 4;
  double corner_min_angle_deg_ = 20.0;
  double corner_max_angle_deg_ = 75.0;
  double corner_vertex_y_min_ratio_ = 0.15;
  double corner_vertex_y_max_ratio_ = 0.80;
  double corner_forward_distance_m_ = 0.20;
  double corner_forward_speed_ = 0.10;
  double corner_forward_max_angular_ = 0.07;
  double corner_advance_lost_timeout_ = 0.15;
  double corner_stop_hold_ = 0.50;
  double turn_right_angular_speed_ = -0.34;
  double turn_right_min_time_ = 1.80;
  double reacquire_angular_speed_ = -0.16;
  double reacquire_timeout_ = 2.5;
  int reacquire_frames_ = 3;
  double align_speed_ = 0.08;
  double align_timeout_ = 4.0;
  double align_lateral_tolerance_px_ = 12.0;
  double align_heading_tolerance_deg_ = 6.0;
  int align_stable_frames_ = 4;
  double corner_cooldown_seconds_ = 1.5;
  int required_corner_count_ = 2;

  double end_roi_y_start_ratio_ = 0.84;
  double end_min_width_ratio_ = 0.45;
  int end_confirm_frames_ = 3;
  double end_stop_hold_ = 1.0;
  double end_turn_left_angle_deg_ = 10.0;
  double end_turn_left_angular_speed_ = 0.50;
  double end_forward_distance_m_ = 0.65;
  double end_forward_speed_ = 0.17;

  int line_lock_count_ = 0;
  int corner_confirm_count_ = 0;
  int reacquire_stable_count_ = 0;
  int align_stable_count_ = 0;
  int completed_corner_count_ = 0;
  double last_lateral_error_ = 0.0;
  double filtered_angular_ = 0.0;
  double last_linear_ = 0.0;
  double last_angular_ = 0.0;
  stable_track::LineModel previous_right_model_;
  stable_track::EndLineDebouncer end_line_debouncer_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "stable_right_track_end_stop_node");
  StableRightTrackEndStopNode node;
  ros::spin();
  return 0;
}

#endif