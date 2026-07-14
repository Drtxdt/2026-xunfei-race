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

        // ===== 淇敼 ===== 宸﹁浆棰勫鐞嗘爣蹇楋紙涓㈢嚎鍚庡厛鍋滀笅鍘熷湴宸﹁浆锛?
        need_left_turn_ = false;

        // ===== 淇敼 ===== PID 绉垎椤?& 寰垎闄愬箙鐩稿叧杩愯鏃跺彉閲?
        integral_pos_error_ = 0.0;

        // ===== 淇敼 ===== 鈥滄娴嬪埌鍙宠浆鈥濊Е鍙戣鏁?
        right_turn_count_ = 0.0;
        right_turn_cooldown_until_ = ros::Time::now();

        // 绗簩涓皷閿愬彸杞笓鐢ㄧ姸鎬?
        detected_corner_count_ = 0;
        second_corner_sequence_done_ = false;

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
        FINAL_STOP = 6,
        SECOND_CORNER_ADVANCE = 7,
        SECOND_CORNER_STOP = 8,
        SECOND_CORNER_RIGHT_TURN = 9
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

    // ========== 鍙傛暟 ==========
    int target_right_x_;
    double base_speed_;
    double curve_speed_;
    double search_speed_;
    double lost_line_speed_;
    double startup_speed_;

    double kp_pos_;
    double kd_pos_;
    double kp_angle_;
    // ===== 淇敼 ===== 鏂板绉垎椤癸紝娑堥櫎寮亾鍚庨暱鏈熻创鍙崇嚎鐨勭ǔ鎬佽宸?
    double ki_pos_;
    double integral_clamp_;
    // ===== 淇敼 ===== 寰垎椤归檺骞咃紝鎶戝埗鍏ュ集鐬棿璇樊璺冲彉閫犳垚鐨勨€滃井鍒嗗啿鍑烩€濆鑷寸殑杩囧害宸﹁浆
    double d_error_clamp_;
    // ===== 淇敼 ===== 纭畨鍏ㄤ笅闄愶細璺濈鍙崇嚎杩囪繎鏃跺己鍒剁殑鏈€灏忓乏杞閫熷害 & 闄愰€熸瘮渚?
    double hard_safety_margin_px_;
    double safety_min_angular_;
    double safety_speed_scale_;

    // ===== 淇敼 ===== 鐢ㄢ€滄娴嬪埌鍙宠浆鈥濅唬鏇库€滀涪澶卞彸绾库€濅綔涓哄仠杞﹀乏杞慨姝ｇ殑瑙﹀彂鏉′欢
    double right_turn_angle_threshold_deg_;
    int right_turn_trigger_count_;
    double right_turn_count_;
    // ===== 淇敼 ===== 瑙﹀彂鍚庣殑鍐峰嵈鏃堕棿锛岄槻姝㈣搴﹀櫔澹板鑷村弽澶嶅仠杞?宸﹁浆锛?涓€鎶戒竴鎶?锛?
    double right_turn_cooldown_sec_;
    ros::Time right_turn_cooldown_until_;

    // 绗簩涓皷閿愬彸杞細鍏堣秺杩囨嫄鐐逛竴娈佃窛绂伙紝鍐嶅仠杞﹀師鍦板彸杞?
    double second_corner_advance_distance_;
    double second_corner_advance_speed_;
    double second_corner_stop_time_;
    double second_corner_turn_angle_deg_;
    double second_corner_turn_angular_;

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

    // ========== 杩愯鏃跺彉閲?==========
    double last_pos_error_;
    double filtered_pos_error_;
    // ===== 淇敼 ===== 绉垎璇樊绱
    double integral_pos_error_;
    int last_right_x_;

    Stage stage_;
    ros::Time start_time_;
    ros::Time stage_start_time_;
    ros::Time forward_start_time_;

    int predicted_right_x_;
    // ===== 淇敼 ===== 鏀逛负娴偣鏁板苟鏀寔鈥滄壘鍒扮嚎鏃剁紦鎱㈣“鍑忊€濊€屼笉鏄灛闂存竻闆讹紝
    // 閬垮厤寮亾澶勫彸绾挎娴嬪伓灏旈棯鐜板鑷翠涪绾胯鏁拌寮哄埗娓呴浂銆佷粠鑰岀浜屾涓嶅啀瑙﹀彂宸﹁浆
    double lost_line_count_;
    double last_angular_;
    ros::Time last_line_time_;

    ros::Time start_moving_time_;
    double stop_line_ignore_time_ = 10.0;

    // ===== 淇敼 ===== 涓㈢嚎鍚庡厛鍋滆溅銆佸師鍦板乏杞殑鐩稿叧鍙傛暟涓庣姸鎬?
    bool need_left_turn_;
    ros::Time left_turn_start_time_;
    double lost_line_stop_duration_;   // 鍋滆溅+宸﹁浆鎬绘椂闀匡紙1~2绉掞級
    double lost_line_turn_angle_deg_;  // 鏈熸湜鍘熷湴宸﹁浆瑙掑害锛堢害12掳锛?

    // ===== 淇敼 ===== 宸﹁浆淇瀹屾垚鍚庯紝缂撴參鍚戝彸鎵弿鎵剧嚎鐨勯€熷害鍙傛暟
    double search_right_speed_;
    double search_right_angular_;

    // ===== 淇敼 ===== 宸﹁浆鏃惰嫢鍙崇嚎浠嶅彲瑙佷笖杩囪繎锛岄澶栧澶ц閫熷害锛屼富鍔ㄨ繙绂诲彸绾?
    double left_turn_safety_margin_px_;
    double left_turn_extra_gain_;
    double left_turn_max_angular_;

    int detected_corner_count_;
    bool second_corner_sequence_done_;
    ros::Time second_corner_stage_start_time_;

    // ========== 鍙傛暟鍔犺浇 ==========
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
        // ===== 淇敼 ===== 绉垎椤?& 寰垎闄愬箙
        pnh.param("ki_pos", ki_pos_, 0.00035);
        pnh.param("integral_clamp", integral_clamp_, 60.0);
        pnh.param("d_error_clamp", d_error_clamp_, 25.0);
        pnh.param("hard_safety_margin_px", hard_safety_margin_px_, 20.0);
        pnh.param("safety_min_angular", safety_min_angular_, 0.22);
        pnh.param("safety_speed_scale", safety_speed_scale_, 0.6);

        // ===== 淇敼 ===== 鐢ㄢ€滄娴嬪埌鍙宠浆鈥濅唬鏇库€滀涪澶卞彸绾库€濊Е鍙戝仠杞?宸﹁浆淇
        pnh.param("right_turn_angle_threshold_deg", right_turn_angle_threshold_deg_, 15.0);
        pnh.param("right_turn_trigger_count", right_turn_trigger_count_, 6);
        pnh.param("right_turn_cooldown_sec", right_turn_cooldown_sec_, 2.5);

        pnh.param("second_corner_advance_distance", second_corner_advance_distance_, 0.25);
        pnh.param("second_corner_advance_speed", second_corner_advance_speed_, 0.32);
        pnh.param("second_corner_stop_time", second_corner_stop_time_, 0.25);
        pnh.param("second_corner_turn_angle_deg", second_corner_turn_angle_deg_, 72.0);
        pnh.param("second_corner_turn_angular", second_corner_turn_angular_, 0.55);

        pnh.param("curve_threshold", curve_threshold_, 35.0);
        pnh.param("curve_offset", curve_offset_, 15.0);
        pnh.param("curve_gain", curve_gain_, 1.0);

        pnh.param("max_angular", max_angular_, 0.45);
        pnh.param("error_filter_alpha", error_filter_alpha_, 0.18);

        // ===== 淇敼 ===== 寮€鏈虹洿琛岃窛绂诲鍔犲埌涔嬪墠鐨?.7鍊嶏紙1.4 * 1.7 鈮?2.4锛?
        pnh.param("startup_time", startup_time_, 2.4);

        pnh.param("cross_area_threshold", cross_area_threshold_, 48000);
        pnh.param("stop_line_min_width", stop_line_min_width_, 120);
        pnh.param("stop_line_max_height", stop_line_max_height_, 40);
        pnh.param("stop_line_min_area", stop_line_min_area_, 800);
        // ===== 淇敼 ===== 鏀惧璇嗗埆闂ㄦ浠ユ彁鍗囧彫鍥炵巼锛屽悓鏃朵粛鑳借繃婊ゆ槑鏄惧櫔澹?
        pnh.param("stop_line_min_fill_ratio", stop_line_min_fill_ratio_, 0.35);
        pnh.param("stop_line_bottom_margin_ratio", stop_line_bottom_margin_ratio_, 0.40);

        pnh.param("align_speed", align_speed_, 0.18);
        pnh.param("align_angle_threshold", align_angle_threshold_, 1.0);
        // ===== 淇敼 ===== 鍋滆溅纭绛夊緟鏃堕棿锛堝緢鐭紝鍙槸闃叉姈锛夛紝瑙掑害淇鍦?ALIGN 闃舵杩涜
        pnh.param("align_stop_time", align_stop_time_, 0.3);
        pnh.param("desired_angle_deg", desired_angle_deg_, -5.0);
        // ===== 淇敼 ===== ALIGN 闃舵鐩茶浆瑙掗€熷害锛岄渶涓?desired_angle_deg_ 閰嶅悎锛?
        // 浣?鈥滃仠杞︾‘璁?+ 鍘熷湴杞?掳鈥?鎬绘椂闀胯惤鍦?1~2 绉?
        pnh.param("align_angular_speed", align_angular_speed_, 0.10);

        pnh.param("final_speed", final_speed_, 0.20);
        // ===== 淇敼 ===== 鐩磋璺濈涓庡師鏉ヤ竴鑷达紙姝ら」涓嶆槸鐢ㄦ埛瑕佹眰鍑忓崐鐨勯偅涓洿琛岃窛绂伙級
        pnh.param("final_distance", final_distance_, 0.60);

        pnh.param("show_debug", show_debug_, true);

        // ===== 淇敼 ===== 涓㈢嚎鍚庡厛鍋滆溅銆佸師鍦板乏杞弬鏁?
        pnh.param("lost_line_stop_duration", lost_line_stop_duration_, 1.5);
        // ===== 淇敼 ===== 25掳 鈫?12掳锛屽乏杞箙搴﹀噺灏?
        pnh.param("lost_line_turn_angle_deg", lost_line_turn_angle_deg_, 12.0);

        // ===== 淇敼 ===== 宸﹁浆淇瀹屾垚鍚庯紝缂撴參鍚戝彸鎵弿鎵剧嚎锛堟瘮涔嬪墠鐨?.12/-0.26鏇存參鏇寸ǔ锛?
        pnh.param("search_right_speed", search_right_speed_, 0.08);
        pnh.param("search_right_angular", search_right_angular_, 0.14);

        pnh.param("left_turn_safety_margin_px", left_turn_safety_margin_px_, 40.0);
        pnh.param("left_turn_extra_gain", left_turn_extra_gain_, 0.01);
        pnh.param("left_turn_max_angular", left_turn_max_angular_, 0.9);
    }

    // ========== 鍥惧儚鍥炶皟 ==========
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
        case SECOND_CORNER_ADVANCE:
            handleSecondCornerAdvance(twist);
            break;
        case SECOND_CORNER_STOP:
            handleSecondCornerStop(twist);
            break;
        case SECOND_CORNER_RIGHT_TURN:
            handleSecondCornerRightTurn(twist);
            break;
        case FINAL_STOP:
        default:
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            break;
        }

        // 姝ｅ父寰嚎鏃跺仛杞悜骞虫粦锛涘仠杞﹀拰鍘熷湴杞悜闃舵涓嶅钩婊戯紝閬垮厤娈嬩綑瑙掗€熷害閫犳垚婊戝姩
        if(stage_ == STARTUP || stage_ == SEARCH_RIGHT_LINE || stage_ == FOLLOW_RIGHT_LINE)
        {
            twist.angular.z = 0.55 * last_angular_ + 0.45 * twist.angular.z;
        }
        else
        {
            last_angular_ = 0.0;
        }
        last_angular_ = twist.angular.z;

        cmd_pub_.publish(twist);

        if(show_debug_)
        {
            showDebug(mask, right_line, stop_line, twist);
        }
    }

    // ========== 鐘舵€佹満澶勭悊鍑芥暟 ==========
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
        // ===== 淇敼 ===== 涓㈢嚎鎭㈠锛氬厛鍘熷湴鍋滆溅锛坙inear=0锛夊湪1~2绉掑唴瀹屾垚宸﹁浆瑙掑害淇锛?
        // 涓嶅啀涓€杈瑰墠杩涗竴杈硅浆锛岄伩鍏嶇浜屾鏇茬嚎澶勫洜瑙嗚闂儊瀵艰嚧淇琚烦杩?璺濈鍙崇嚎杩囪繎
        if(need_left_turn_)
        {
            double elapsed = (ros::Time::now() - left_turn_start_time_).toSec();

            double planned_angular =
                deg2rad(lost_line_turn_angle_deg_) / std::max(0.1, lost_line_stop_duration_);

            // 鑻ュ彸绾夸粛鍙涓旇窛绂昏繃杩戯紝閫傚綋鍔犲ぇ瑙掗€熷害锛屼富鍔ㄥ鍔犱笌鍙崇嚎鐨勮窛绂?
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
                twist.linear.x = 0.0;          // 瀹屽叏鍋滆溅锛屽彧鍋氬師鍦版棆杞慨姝?
                twist.angular.z = planned_angular;
                return;
            }
            else
            {
                need_left_turn_ = false;   // 宸﹁浆瀹屾垚锛岃繘鍏ユ甯告悳绱?
            }
        }

        // ===== 淇敼 ===== 鍘熷湴宸﹁浆淇瀹屾垚鍚庯紝鏀逛负缂撴參鍚戝彸鎵弿鎵剧嚎锛堥檷浣庤閫熷害锛岄伩鍏嶅張杞繃澶达級
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

        // ===== 淇敼 ===== 瑙﹀彂鏉′欢鐢扁€滀涪澶卞彸绾库€濇敼涓衡€滄娴嬪埌鍙宠浆鈥濓細
        // 鍙杩樿兘鐪嬪埌鍙崇嚎锛屽氨鐢ㄥ叾鎷熷悎瑙掑害鍒ゆ柇鏄惁姝ｅ湪杩涘叆鍙宠浆寮亾锛?
        // 鎻愬墠鍘熷湴鍋滆溅+宸﹁浆淇锛岃€屼笉鏄瓑鍒扮嚎瀹屽叏涓㈠け鎵嶅弽搴旓紙涓㈠け寰€寰€宸茬粡澶櫄/澶繎浜嗭級
        int current_x = right_line.found ? right_line.x : predicted_right_x_;
        double current_angle = right_line.found ? right_line.angle_deg : desired_angle_deg_;

        if(right_line.found)
        {
            last_right_x_ = right_line.x;
            predicted_right_x_ = 0.7 * predicted_right_x_ + 0.3 * right_line.x;
            last_line_time_ = ros::Time::now();

            // ===== 淇敼 ===== 蹇呴』瑙掑害鍋忓樊 涓?浣嶇疆璇樊鍚屾椂鍋忓ぇ锛屾墠绠楃湡姝ｈ繘鍏ュ彸杞集閬擄紱
            // 鍗曠函瑙掑害璇绘暟鍣０鎶栦竴涓嬶紙鐩撮亾涓婁篃鍙兘鍙戠敓锛変笉浼氱疮绉Е鍙戯紝閬垮厤鍙嶅鍋滆溅-宸﹁浆
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

        // ===== 淇敼 ===== 鍐峰嵈鏈熷唴涓嶉噸鏂拌Е鍙戯紝闃叉鍒氬仛瀹屼竴娆″仠杞?宸﹁浆銆?
        // 杞﹁繕娌＄ǔ瀹氫綇鍙堣鍣０绔嬪埢鎷夊洖鏉ワ紝瀵艰嚧鈥滀竴鎶戒竴鎶解€濊蛋涓嶅姩
        bool in_cooldown = ros::Time::now() < right_turn_cooldown_until_;

        if(!in_cooldown && right_turn_count_ >= right_turn_trigger_count_)
        {
            right_turn_count_ = 0.0;
            right_turn_cooldown_until_ = ros::Time::now() + ros::Duration(right_turn_cooldown_sec_);
            detected_corner_count_++;

            // 绗簩涓皷閿愬彸杞笉鍦ㄥ皷鐐瑰绔嬪嵆杞集锛氬厛鐩磋瓒婅繃鎷愮偣绾?5cm锛屽啀鍋滆溅鍘熷湴鍙宠浆
            if(detected_corner_count_ == 2 && !second_corner_sequence_done_)
            {
                second_corner_sequence_done_ = true;
                second_corner_stage_start_time_ = ros::Time::now();
                resetPid();
                enterStage(SECOND_CORNER_ADVANCE,
                           "SECOND CORNER: ADVANCE 25CM BEFORE RIGHT TURN");
                twist.linear.x = second_corner_advance_speed_;
                twist.angular.z = 0.0;
                return;
            }

            // 鍏朵粬寮亾缁х画閲囩敤鍘熸潵鐨勬俯鍜屼慨姝?
            need_left_turn_ = true;
            left_turn_start_time_ = ros::Time::now();
            enterStage(SEARCH_RIGHT_LINE, "RIGHT TURN DETECTED, STOP & LEFT CORRECTION");
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            return;
        }

        // 涓㈢嚎鍏滃簳锛氬嵆渚胯搴︽娴嬫病鎻愬墠鎶撳埌锛岀嚎鐪熺殑瀹屽叏涓㈠け鏃朵緷鐒惰瑙﹀彂鍚屼竴濂楀仠杞?宸﹁浆
        if(!in_cooldown && lost_line_count_ > 10.0)
        {
            right_turn_cooldown_until_ = ros::Time::now() + ros::Duration(right_turn_cooldown_sec_);
            detected_corner_count_++;

            // 灏栬澶勫彲鑳界洿鎺ヤ涪绾匡紝鍥犳鍏滃簳瑙﹀彂涔熷繀椤昏鍏モ€滅鍑犱釜鎷愮偣鈥?
            if(detected_corner_count_ == 2 && !second_corner_sequence_done_)
            {
                second_corner_sequence_done_ = true;
                resetPid();
                enterStage(SECOND_CORNER_ADVANCE,
                           "SECOND CORNER (LOST LINE): ADVANCE 25CM BEFORE RIGHT TURN");
                twist.linear.x = second_corner_advance_speed_;
                twist.angular.z = 0.0;
                return;
            }

            need_left_turn_ = true;
            left_turn_start_time_ = ros::Time::now();
            enterStage(SEARCH_RIGHT_LINE, "LINE LOST (fallback), STOP & LEFT CORRECTION");
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

        // ===== 淇敼 ===== 寰垎椤归檺骞咃細闃叉鍏ュ集鐬棿璇樊璺冲彉浜х敓鈥滃井鍒嗗啿鍑烩€濓紝
        // 杩欐槸涔嬪墠鈥滅涓€娆℃嫄寮鍚戝乏杞繃澶氥€佸帇鍒板乏杈圭嚎鈥濈殑涓昏鍘熷洜涔嬩竴
        double d_pos_error = filtered_pos_error_ - last_pos_error_;
        d_pos_error = clamp(d_pos_error, -d_error_clamp_, d_error_clamp_);
        last_pos_error_ = filtered_pos_error_;

        // ===== 淇敼 ===== 绉垎椤癸紙甯︽姉楗卞拰闄愬箙锛夛細娑堥櫎寮亾涔嬪悗闀挎湡璐村彸绾跨殑绋虫€佸亸宸?
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

        // ===== 淇敼 ===== 纭畨鍏ㄤ笅闄愶細鍙璺濈鍙崇嚎杩囪繎锛堜笉绠℃槸鍚﹀湪寮亾/PID绠楀嚭澶氬皯锛夛紝
        // 寮哄埗淇濊瘉鑷冲皯鏈夎繖涔堝ぇ鐨勫乏杞慨姝ｏ紝骞堕檷浣庣嚎閫熷害缁欏嚭鏇村鍙嶅簲鏃堕棿銆?
        // 杩欐槸閽堝鈥滀竴鐩磋创鍙崇嚎澶繎銆佸帇绾库€濊繖涓寔缁€ч棶棰樼殑鍏滃簳鎺柦銆?
        if(current_x < target_right_x_ - hard_safety_margin_px_)
        {
            angular = std::max(angular, safety_min_angular_);
            linear_speed = std::min(linear_speed, base_speed_ * safety_speed_scale_);
        }

        twist.linear.x = linear_speed;
        twist.angular.z = angular;
    }

    void handleSecondCornerAdvance(geometry_msgs::Twist& twist)
    {
        const double speed = std::max(0.05, std::fabs(second_corner_advance_speed_));
        const double duration = second_corner_advance_distance_ / speed;
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();

        if(elapsed < duration)
        {
            // 绌胯繃灏栬鏃朵繚鎸佺洿琛岋紝涓嶅啀杩介殢绐佺劧鎶樺洖鐨勫彸绾?
            twist.linear.x = second_corner_advance_speed_;
            twist.angular.z = 0.0;
            return;
        }

        enterStage(SECOND_CORNER_STOP, "SECOND CORNER: ADVANCE DONE, STOP");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleSecondCornerStop(geometry_msgs::Twist& twist)
    {
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();
        if(elapsed >= second_corner_stop_time_)
        {
            enterStage(SECOND_CORNER_RIGHT_TURN,
                       "SECOND CORNER: START IN-PLACE RIGHT TURN");
        }
    }

    void handleSecondCornerRightTurn(geometry_msgs::Twist& twist)
    {
        const double angular = std::max(0.05, std::fabs(second_corner_turn_angular_));
        const double duration =
            std::fabs(deg2rad(second_corner_turn_angle_deg_)) / angular;
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();

        if(elapsed < duration)
        {
            twist.linear.x = 0.0;
            twist.angular.z = -angular;  // ROS涓礋瑙掗€熷害涓哄彸杞?
            return;
        }

        // 杞畬鍚庨噸鏂版悳绱㈠彸绾匡紝涓嶇珛鍗冲悜鍙嶆柟鍚戣ˉ鍋?
        need_left_turn_ = false;
        lost_line_count_ = 0.0;
        right_turn_count_ = 0.0;
        resetPid();
        right_turn_cooldown_until_ =
            ros::Time::now() + ros::Duration(right_turn_cooldown_sec_);
        enterStage(SEARCH_RIGHT_LINE,
                   "SECOND CORNER: RIGHT TURN DONE, SEARCH RIGHT LINE");
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;
    }

    void handleStopLineFound(geometry_msgs::Twist& twist)
    {
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();
        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        // ===== 淇敼 ===== 杩欓噷鍙槸鐭殏闃叉姈纭鍋滄绾匡紙涓嶅仛杞悜锛夛紝
        // 鐪熸鐨勮搴︿慨姝ｅ湪绱ф帴鐫€鐨?ALIGN 闃舵锛堝悓鏍蜂繚鎸佸仠杞︾姸鎬侊級瀹屾垚锛?
        // 涓ゆ鐩稿姞鎺у埗鍦?1~2 绉掑唴
        if(elapsed >= align_stop_time_)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "ENTER ALIGN MODE (stop & correct 5 deg)");
        }
        if(elapsed > 2.0)
        {
            enterStage(ALIGN_WITH_RIGHT_LINE, "STOPLINE TIMEOUT");
        }
    }

    // ===== 淇敼 ===== 瀵归綈闃舵锛氬叏绋嬩繚鎸佸仠杞︼紙linear=0锛夛紝鍘熷湴宸﹁浆 |desired_angle_deg_|锛?掳锛夛紝
    // 涓嶄緷璧栬瑙夋寔缁窡韪紝杞姩鏃堕暱鐢?align_angular_speed_ 鍐冲畾锛?
    // 涓?STOP_LINE_FOUND 鐨勭煭鏆傜‘璁ゆ椂闂寸浉鍔狅紝鎬诲仠杞?淇鏃堕棿钀藉湪 1~2 绉?
    void handleAlign(geometry_msgs::Twist& twist, const StopLineInfo& /*stop_line*/)
    {
        twist.linear.x = 0.0;

        const double turn_duration =
            std::fabs(deg2rad(desired_angle_deg_)) / std::max(0.01, align_angular_speed_);
        const double elapsed = (ros::Time::now() - stage_start_time_).toSec();

        if(elapsed < turn_duration)
        {
            twist.angular.z = align_angular_speed_;   // 姝ｅ€?= 宸﹁浆
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

    // ========== 鍥惧儚澶勭悊锛堜繚鎸佷笉鍙橈級 ==========
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
            // ===== 淇敼 ===== 杩欓噷涓嶅啀鐩存帴淇敼 lost_line_count_锛堥伩鍏嶅拰 handleFollow 涓殑
            // 绱閫昏緫閲嶅璁℃暟锛屽鑷翠涪绾胯鏁板闀胯繃蹇垨涓嶄竴鑷达級
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

    // ===== 淇敼 ===== 鍋滆溅妯嚎璇嗗埆锛氭斁瀹藉彫鍥烇紙fill_ratio / bottom_margin 闃堝€奸檷浣庯級锛?
    // 鍚屾椂淇濈暀鍩烘湰鍑犱綍杩囨护锛屽噺灏戔€滆瘑鍒笉鍒扮涓€鏉℃í绾库€濈殑婕忔
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
                // ===== 淇敼 ===== 璇婃柇鏃ュ織锛氳褰曗€滃樊涓€鐐硅鍒ゅ畾涓哄仠杞︾嚎鈥濈殑鍊欓€夋锛?
                // 鏂逛究鍦ㄦ病鏈夌敾闈㈢殑鎯呭喌涓嬮€氳繃 rosout 鍒ゆ柇鍏蜂綋鏄摢涓槇鍊煎崱浣忎簡
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

        // ===== 淇敼 ===== 琛屽瘑搴﹀厹搴曟娴嬶細濡傛灉杞粨娉曚粛鏈瘑鍒埌锛?
        // 閫愯妫€鏌ュ簳閮ㄥ尯鍩熸槸鍚︽湁涓€鏁存潯鈥滄í鍚戝ぇ鐗囩櫧鑹测€濓紙瑕嗙洊鐜囬珮鐨勮锛夛紝
        // 杩欑鎯呭喌閫氬父灏辨槸鍋滆溅绾匡紝浣嗗洜褰㈢姸/鏂琚疆寤撹繃婊ゆ帀浜?
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

    // ========== 璋冭瘯鏄剧ず锛堜繚鎸佷笉鍙橈級 ==========
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

    // ========== 杈呭姪鍑芥暟 ==========
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
        integral_pos_error_ = 0.0;   // ===== 淇敼 ===== 姣忔閲嶆柊鎵惧埌绾挎椂娓呯┖绉垎锛岄伩鍏嶅巻鍙茶宸甫鍏?
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
        case SECOND_CORNER_ADVANCE: return "CORNER_ADVANCE";
        case SECOND_CORNER_STOP: return "CORNER_STOP";
        case SECOND_CORNER_RIGHT_TURN: return "CORNER_RIGHT";
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