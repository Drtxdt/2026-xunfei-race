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
    private_nh_.param("startup_time", startup_time_, 2.0);
    private_nh_.param("startup_speed", startup_speed_, 0.45);

    private_nh_.param("target_right_x", target_right_x_, 200);
    private_nh_.param("base_speed", base_speed_, 0.34);
    private_nh_.param("fast_straight_speed", fast_straight_speed_, 0.46);
    private_nh_.param("curve_speed", curve_speed_, 0.23);
    private_nh_.param("medium_curve_speed", medium_curve_speed_, 0.30);
    private_nh_.param("search_speed", search_speed_, 0.06);
    private_nh_.param("search_angular_speed", search_angular_speed_, -0.34);
    private_nh_.param("lost_linear_speed", lost_linear_speed_, 0.12);
    private_nh_.param("lost_angular_speed", lost_angular_speed_, -0.28);
    private_nh_.param("lost_stop_frames", lost_stop_frames_, 8);
    private_nh_.param("lost_search_frames", lost_search_frames_, 18);
    private_nh_.param("kp", kp_, 0.0042);
    private_nh_.param("kd", kd_, 0.0008);
    private_nh_.param("error_alpha", error_alpha_, 0.24);
    private_nh_.param("curve_error_threshold", curve_error_threshold_, 52.0);
    private_nh_.param("medium_curve_error_threshold", medium_curve_error_threshold_, 30.0);
    private_nh_.param("fast_straight_error_px", fast_straight_error_px_, 18.0);
    private_nh_.param("fast_straight_derivative_px", fast_straight_derivative_px_, 5.0);
    private_nh_.param("fast_straight_min_frames", fast_straight_min_frames_, 3);
    private_nh_.param("fast_straight_angular_limit", fast_straight_angular_limit_, 0.10);
    private_nh_.param("curve_angular_gain", curve_angular_gain_, 1.05);
    private_nh_.param("max_angular_speed", max_angular_speed_, 0.40);
    private_nh_.param("steering_deadband_px", steering_deadband_px_, 7.0);
    private_nh_.param("max_straight_angular_speed", max_straight_angular_speed_, 0.18);
    private_nh_.param("max_right_angular_speed", max_right_angular_speed_, 0.34);
    private_nh_.param("straight_angular_alpha", straight_angular_alpha_, 0.68);
    private_nh_.param("curve_angular_alpha", curve_angular_alpha_, 0.58);
    private_nh_.param("straight_angular_step", straight_angular_step_, 0.045);
    private_nh_.param("curve_angular_step", curve_angular_step_, 0.065);
    private_nh_.param("right_guard_error_px", right_guard_error_px_, 70.0);
    private_nh_.param("right_guard_speed", right_guard_speed_, 0.18);
    private_nh_.param("deadband_angular_decay", deadband_angular_decay_, 0.38);
    private_nh_.param("right_x_alpha", right_x_alpha_, 0.28);
    private_nh_.param("right_x_max_jump_px", right_x_max_jump_px_, 75.0);
    private_nh_.param("right_line_min_votes", right_line_min_votes_, 3);
    private_nh_.param("right_line_min_segment_width", right_line_min_segment_width_, 3);
    private_nh_.param("right_line_max_segment_width", right_line_max_segment_width_, 90);
    private_nh_.param("right_search_left_limit", right_search_left_limit_, 80);
    private_nh_.param("right_safe_min_x", right_safe_min_x_, 145);
    private_nh_.param("right_emergency_min_x", right_emergency_min_x_, 115);
    private_nh_.param("right_guard_left_angular", right_guard_left_angular_, 0.24);
    private_nh_.param("right_emergency_left_angular", right_emergency_left_angular_, 0.38);
    private_nh_.param("right_emergency_speed", right_emergency_speed_, 0.05);

    private_nh_.param("roi_y_start_ratio", roi_y_start_ratio_, 0.60);
    private_nh_.param("white_s_max", white_s_max_, 45);
    private_nh_.param("white_v_min", white_v_min_, 200);
    private_nh_.param("morph_kernel_size", morph_kernel_size_, 5);
    private_nh_.param("min_component_area", min_component_area_, 180.0);

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
    lost_frame_count_ = 0;

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
    const bool in_medium_curve = abs_error > medium_curve_error_threshold_;

    // When the car and right boundary are already aligned, do not hesitate:
    // build a few stable frames, then run at full straight speed. The fast mode
    // is disabled immediately if the line approaches the safety zone or the
    // error starts changing quickly.
    const bool fast_straight_candidate =
        abs_error <= fast_straight_error_px_ &&
        std::fabs(d_error) <= fast_straight_derivative_px_ &&
        result.right_x > right_safe_min_x_ + 18;
    if (fast_straight_candidate)
      ++straight_stable_frames_;
    else
      straight_stable_frames_ = 0;

    const bool fast_straight = straight_stable_frames_ >= fast_straight_min_frames_;
    if (fast_straight)
    {
      linear = fast_straight_speed_;
      angular = clampDouble(angular, -fast_straight_angular_limit_, fast_straight_angular_limit_);
    }
    else if (in_curve)
    {
      linear = curve_speed_;
      angular *= curve_angular_gain_;
    }
    else if (in_medium_curve)
    {
      linear = medium_curve_speed_;
    }

    // Direct pixel-distance protection: when the right line moves too close
    // to the image centre, immediately slow down and command a left correction.
    // This guard has priority over PID and is the final defence against pressing
    // or crossing the right boundary.
    if (result.right_x <= right_emergency_min_x_)
    {
      linear = std::min(linear, right_emergency_speed_);
      angular = std::max(angular, right_emergency_left_angular_);
    }
    else if (result.right_x <= right_safe_min_x_)
    {
      linear = std::min(linear, right_guard_speed_);
      const double guard_ratio = static_cast<double>(right_safe_min_x_ - result.right_x) /
                                 std::max(1, right_safe_min_x_ - right_emergency_min_x_);
      angular = std::max(angular, right_guard_left_angular_ * (0.65 + 0.35 * guard_ratio));
    }

    // A large negative error previously received two gain boosts and could
    // drive the chassis onto the right-hand line.  Keep the turn authority for
    // the bend, but slow down further while that risky correction is active.
    if (filtered_error_ < -right_guard_error_px_)
      linear = std::min(linear, right_guard_speed_);

    const bool right_protection_active = result.right_x <= right_safe_min_x_;
    const double positive_limit = right_protection_active
                                    ? max_angular_speed_
                                    : (in_curve ? max_angular_speed_ : max_straight_angular_speed_);
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

    result.filtered_error = filtered_error_;
    result.linear = linear;
    result.angular = filtered_angular_;
    return result;
  }

  int findRightLine(const cv::Mat& mask)
  {
    // Use many scan rows and segment geometry instead of taking the first white
    // pixel from only four rows. This rejects glare, isolated noise and wide
    // horizontal markings while preserving the original right-line strategy.
    const int h = mask.rows;
    const std::vector<double> row_ratios = {0.22, 0.32, 0.42, 0.52, 0.62,
                                            0.70, 0.78, 0.86, 0.93};
    std::vector<int> candidates;
    const int expected_x = filtered_right_x_ >= 0.0
                               ? static_cast<int>(filtered_right_x_)
                               : (last_right_x_ >= 0 ? last_right_x_ : target_right_x_);
    const int continuity_window = last_right_x_ >= 0
                                    ? static_cast<int>(right_x_max_jump_px_)
                                    : mask.cols;
    const int search_left = clampInt(std::min(right_search_left_limit_,
                                               expected_x - continuity_window),
                                     0, mask.cols - 1);

    for (double ratio : row_ratios)
    {
      const int y = clampInt(static_cast<int>(h * ratio), 0, h - 1);
      const std::vector<Segment> segments = findSegments(mask.row(y));
      int best_x = -1;
      double best_score = 1e9;
      for (const Segment& segment : segments)
      {
        if (segment.width < right_line_min_segment_width_ ||
            segment.width > right_line_max_segment_width_)
          continue;

        const int center_x = (segment.left + segment.right) / 2;
        if (center_x < search_left)
          continue;

        const double continuity = std::fabs(static_cast<double>(center_x - expected_x));
        if (last_right_x_ >= 0 && continuity > right_x_max_jump_px_)
          continue;

        // Prefer candidates close to the previous line position. A small bias
        // toward the right keeps the detector attached to the right boundary.
        const double score = continuity - 0.035 * center_x;
        if (score < best_score)
        {
          best_score = score;
          best_x = center_x;
        }
      }
      if (best_x >= 0)
        candidates.push_back(best_x);
    }

    if (static_cast<int>(candidates.size()) < right_line_min_votes_)
      return -1;

    std::sort(candidates.begin(), candidates.end());
    const int median_x = candidates[candidates.size() / 2];

    // Reject frames whose row observations disagree strongly; these are most
    // often reflections, crossings or fragmented masks.
    std::vector<int> deviations;
    deviations.reserve(candidates.size());
    for (int x : candidates)
      deviations.push_back(std::abs(x - median_x));
    std::sort(deviations.begin(), deviations.end());
    if (deviations[deviations.size() / 2] > 32)
      return -1;

    if (filtered_right_x_ < 0.0)
      filtered_right_x_ = median_x;
    else
      filtered_right_x_ = (1.0 - right_x_alpha_) * filtered_right_x_ +
                          right_x_alpha_ * median_x;
    return clampInt(static_cast<int>(std::lround(filtered_right_x_)), 0, mask.cols - 1);
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
      ++lost_frame_count_;
      straight_stable_frames_ = 0;
      const double target_angular = last_right_x_ >= 0 ? lost_angular_speed_ : search_angular_speed_;

      // First few missed frames: creep forward while steering right to recover.
      // Persistent loss: stop translation and rotate-search, preventing the car
      // from blindly running outside the track.
      if (lost_frame_count_ <= lost_stop_frames_)
      {
        setStatus("stable_right_lost_slow_search");
        cmd.linear.x = lost_linear_speed_;
      }
      else
      {
        setStatus("stable_right_lost_rotate_search");
        cmd.linear.x = 0.0;
      }

      double search_turn = target_angular;
      if (lost_frame_count_ > lost_search_frames_)
        search_turn = search_angular_speed_;
      cmd.angular.z = clampDouble(search_turn,
                                  last_angular_ - curve_angular_step_,
                                  last_angular_ + curve_angular_step_);
      publishCmd(cmd);
      return;
    }

    last_right_x_ = follow.right_x;
    last_detection_time_ = ros::Time::now();
    cmd.linear.x = follow.linear;
    cmd.angular.z = follow.angular;
    if (follow.filtered_error < -right_guard_error_px_)
      setStatus("stable_right_tracking_right_guard");
    else if (straight_stable_frames_ >= fast_straight_min_frames_)
      setStatus("stable_right_tracking_fast_straight");
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
       << " lost_frames=" << lost_frame_count_
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
  double startup_time_ = 2.0;
  double startup_speed_ = 0.45;

  int target_right_x_ = 200;
  double base_speed_ = 0.34;
  double fast_straight_speed_ = 0.46;
  double curve_speed_ = 0.23;
  double medium_curve_speed_ = 0.30;
  double search_speed_ = 0.06;
  double search_angular_speed_ = -0.26;
  double lost_linear_speed_ = 0.12;
  double lost_angular_speed_ = -0.28;
  int lost_stop_frames_ = 8;
  int lost_search_frames_ = 18;
  double kp_ = 0.0042;
  double kd_ = 0.0008;
  double error_alpha_ = 0.24;
  double curve_error_threshold_ = 52.0;
  double medium_curve_error_threshold_ = 30.0;
  double fast_straight_error_px_ = 18.0;
  double fast_straight_derivative_px_ = 5.0;
  int fast_straight_min_frames_ = 3;
  double fast_straight_angular_limit_ = 0.10;
  double curve_angular_gain_ = 1.05;
  double max_angular_speed_ = 0.40;
  double steering_deadband_px_ = 7.0;
  double max_straight_angular_speed_ = 0.18;
  double max_right_angular_speed_ = 0.34;
  double straight_angular_alpha_ = 0.68;
  double curve_angular_alpha_ = 0.50;
  double straight_angular_step_ = 0.045;
  double curve_angular_step_ = 0.08;
  double right_guard_error_px_ = 70.0;
  double right_guard_speed_ = 0.18;
  double deadband_angular_decay_ = 0.38;
  double right_x_alpha_ = 0.28;
  double right_x_max_jump_px_ = 75.0;
  int right_line_min_votes_ = 3;
  int right_line_min_segment_width_ = 3;
  int right_line_max_segment_width_ = 90;
  int right_search_left_limit_ = 80;
  int right_safe_min_x_ = 145;
  int right_emergency_min_x_ = 115;
  double right_guard_left_angular_ = 0.24;
  double right_emergency_left_angular_ = 0.38;
  double right_emergency_speed_ = 0.05;

  double roi_y_start_ratio_ = 0.60;
  int white_s_max_ = 45;
  int white_v_min_ = 200;
  int morph_kernel_size_ = 5;
  double min_component_area_ = 180.0;

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
  double filtered_right_x_ = -1.0;
  int lost_frame_count_ = 0;
  int straight_stable_frames_ = 0;
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