#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Twist.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

class StableRightFollowNode
{
public:

    StableRightFollowNode()
    {
        //------------------------------------------
        // 参数
        //------------------------------------------

        ros::NodeHandle pnh("~");

        pnh.param("target_right_x", target_right_x_, 155);

        pnh.param("base_speed", base_speed_, 0.32);
        pnh.param("curve_speed", curve_speed_, 0.28);
        pnh.param("search_speed", search_speed_, 0.10);

        pnh.param("kp", kp_, 0.0052);
        pnh.param("kd", kd_, 0.0018);

        pnh.param("startup_time", startup_time_, 2.8);
        pnh.param("forward_time", forward_time_, 0.9);

        pnh.param("cross_area_threshold",
                  cross_area_threshold_,
                  48000);

        pnh.param("show_debug",
                  show_debug_,
                  true);

        //------------------------------------------
        // ROS
        //------------------------------------------

        cmd_pub_ =
            nh_.advertise<geometry_msgs::Twist>(
                "/cmd_vel",
                1
            );

        image_sub_ =
            nh_.subscribe(
                "/usb_cam/image_raw",
                1,
                &StableRightFollowNode::imageCallback,
                this
            );

        //------------------------------------------
        // PID
        //------------------------------------------

        last_error_ = 0.0;
        filtered_error_ = 0.0;

        //------------------------------------------
        // 状态机
        //------------------------------------------

        stage_ = 0;

        start_time_ = ros::Time::now();

        last_right_x_ = -1;

        ROS_INFO("=================================");
        ROS_INFO(" Stable Right Follow Started ");
        ROS_INFO("=================================");
    }

private:

    //------------------------------------------
    // ROS
    //------------------------------------------

    ros::NodeHandle nh_;

    ros::Publisher cmd_pub_;
    ros::Subscriber image_sub_;

    //------------------------------------------
    // 参数
    //------------------------------------------

    int target_right_x_;

    double base_speed_;
    double curve_speed_;
    double search_speed_;

    double kp_;
    double kd_;

    double startup_time_;
    double forward_time_;

    int cross_area_threshold_;

    bool show_debug_;

    //------------------------------------------
    // PID
    //------------------------------------------

    double last_error_;
    double filtered_error_;

    //------------------------------------------
    // 状态机
    //------------------------------------------

    int stage_;

    ros::Time start_time_;
    ros::Time forward_start_time_;

    int last_right_x_;

    //------------------------------------------
    // 图像回调
    //------------------------------------------

    void imageCallback(
        const sensor_msgs::ImageConstPtr& msg)
    {
        cv::Mat frame;

        try
        {
            frame =
                cv_bridge::toCvCopy(
                    msg,
                    "bgr8"
                )->image;
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("%s", e.what());
            return;
        }

        int h = frame.rows;
        int w = frame.cols;

        //------------------------------------------
        // ROI
        //------------------------------------------

        cv::Mat roi =
            frame(
                cv::Range(
                    int(h * 0.60),
                    h
                ),
                cv::Range(
                    0,
                    w
                )
            );

        //------------------------------------------
        // 白线提取
        //------------------------------------------

        cv::Mat mask =
            extractWhiteMask(roi);

        geometry_msgs::Twist twist;

        //------------------------------------------
        // STAGE 0
        //------------------------------------------

        if(stage_ == 0)
        {
            double elapsed =
                (ros::Time::now() -
                 start_time_).toSec();

            if(elapsed < startup_time_)
            {
                twist.linear.x = 0.45;
                twist.angular.z = 0.0;

                cmd_pub_.publish(twist);

                return;
            }

            ROS_INFO("ENTER SEARCH MODE");

            stage_ = 1;
        }

        //------------------------------------------
        // 找右边线
        //------------------------------------------

        int right_x =
            findRightLine(mask);

        //------------------------------------------
        // STAGE 1
        //------------------------------------------

        if(stage_ == 1)
        {
            if(right_x < 0)
            {
                twist.linear.x = 0.10;
                twist.angular.z = -0.26;

                cmd_pub_.publish(twist);

                return;
            }

            ROS_INFO("RIGHT LINE FOUND");

            last_right_x_ = right_x;

            stage_ = 2;
        }

        //------------------------------------------
        // STAGE 2
        //------------------------------------------

        if(stage_ == 2)
        {
            int cross_area =
                cv::countNonZero(mask);

            if(cross_area >
               cross_area_threshold_)
            {
                ROS_INFO("STOP LINE DETECTED");

                stage_ = 3;

                forward_start_time_ =
                    ros::Time::now();

                return;
            }

            //----------------------------------
            // 丢线
            //----------------------------------

            if(right_x < 0)
            {
                if(last_right_x_ >= 0)
                {
                    twist.linear.x = 0.14;
                    twist.angular.z = -0.22;
                }
                else
                {
                    twist.linear.x = 0.10;
                    twist.angular.z = -0.24;
                }

                cmd_pub_.publish(twist);

                return;
            }

            last_right_x_ = right_x;

            //----------------------------------
            // PID
            //----------------------------------

            double error =
                target_right_x_ -
                right_x;

            double alpha = 0.22;

            filtered_error_ =
                (1.0 - alpha) *
                filtered_error_
                +
                alpha * error;

            double d_error =
                filtered_error_
                -
                last_error_;

            last_error_ =
                filtered_error_;

            double angular =
                kp_ * filtered_error_
                +
                kd_ * d_error;

            double linear_speed;

            //----------------------------------
            // 弯道增强
            //----------------------------------

            if(std::fabs(filtered_error_) > 38)
            {
                linear_speed =
                    curve_speed_;

                angular *= 1.18;
            }
            else
            {
                linear_speed =
                    base_speed_;
            }

            //----------------------------------
            // 限幅
            //----------------------------------

            if(angular > 0.55)
                angular = 0.55;

            if(angular < -0.55)
                angular = -0.55;

            twist.linear.x =
                linear_speed;

            twist.angular.z =
                angular;

            cmd_pub_.publish(twist);

            //----------------------------------
            // Debug
            //----------------------------------

            if(show_debug_)
            {
                cv::Mat debug;

                cv::cvtColor(
                    mask,
                    debug,
                    cv::COLOR_GRAY2BGR
                );

                cv::line(
                    debug,
                    cv::Point(
                        target_right_x_,
                        0
                    ),
                    cv::Point(
                        target_right_x_,
                        mask.rows
                    ),
                    cv::Scalar(
                        255,
                        0,
                        0
                    ),
                    2
                );

                cv::circle(
                    debug,
                    cv::Point(
                        right_x,
                        mask.rows / 2
                    ),
                    5,
                    cv::Scalar(
                        0,
                        0,
                        255
                    ),
                    -1
                );

                char buf[100];

                sprintf(
                    buf,
                    "ERR: %.2f",
                    filtered_error_
                );

                cv::putText(
                    debug,
                    buf,
                    cv::Point(20,40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(
                        0,
                        255,
                        0
                    ),
                    2
                );

                cv::imshow(
                    "right_follow",
                    debug
                );

                cv::waitKey(1);
            }

            return;
        }

        //------------------------------------------
        // STAGE 3
        //------------------------------------------

        if(stage_ == 3)
        {
            double elapsed =
                (ros::Time::now()
                 -
                 forward_start_time_).toSec();

            if(elapsed < forward_time_)
            {
                twist.linear.x = 0.12;
                twist.angular.z = 0.0;

                cmd_pub_.publish(twist);

                return;
            }

            stage_ = 4;
        }

        //------------------------------------------
        // STAGE 4
        //------------------------------------------

        if(stage_ == 4)
        {
            stopCar();

            ROS_INFO_THROTTLE(
                1.0,
                "FINAL STOP"
            );

            return;
        }
    }

    //------------------------------------------
    // 白线提取
    //------------------------------------------

    cv::Mat extractWhiteMask(
        const cv::Mat& roi)
    {
        cv::Mat blur;

        cv::GaussianBlur(
            roi,
            blur,
            cv::Size(5,5),
            0
        );

        cv::Mat hsv;

        cv::cvtColor(
            blur,
            hsv,
            cv::COLOR_BGR2HSV
        );

        cv::Mat mask;

        cv::inRange(
            hsv,
            cv::Scalar(
                0,
                0,
                200
            ),
            cv::Scalar(
                180,
                45,
                255
            ),
            mask
        );

        cv::Mat kernel =
            cv::Mat::ones(
                5,
                5,
                CV_8U
            );

        cv::morphologyEx(
            mask,
            mask,
            cv::MORPH_OPEN,
            kernel
        );

        cv::morphologyEx(
            mask,
            mask,
            cv::MORPH_CLOSE,
            kernel
        );

        cv::medianBlur(
            mask,
            mask,
            5
        );

        cv::GaussianBlur(
            mask,
            mask,
            cv::Size(5,5),
            0
        );

        std::vector<
            std::vector<cv::Point>
        > contours;

        cv::findContours(
            mask,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        cv::Mat clean_mask =
            cv::Mat::zeros(
                mask.size(),
                CV_8UC1
            );

        for(auto& cnt : contours)
        {
            double area =
                cv::contourArea(cnt);

            if(area > 260)
            {
                cv::drawContours(
                    clean_mask,
                    std::vector<
                        std::vector<cv::Point>
                    >{cnt},
                    -1,
                    cv::Scalar(255),
                    -1
                );
            }
        }

        return clean_mask;
    }

    //------------------------------------------
    // 找右边线
    //------------------------------------------

    int findRightLine(
        const cv::Mat& mask)
    {
        int h =
            mask.rows;

        std::vector<int> rows;

        rows.push_back(
            int(h * 0.50)
        );

        rows.push_back(
            int(h * 0.60)
        );

        rows.push_back(
            int(h * 0.70)
        );

        std::vector<int> points;

        for(auto y : rows)
        {
            const uchar* ptr =
                mask.ptr<uchar>(y);

            for(int x = mask.cols - 1;
                x >= 0;
                x--)
            {
                if(ptr[x] > 0)
                {
                    points.push_back(x);
                    break;
                }
            }
        }

        if(points.empty())
            return -1;

        double sum = 0.0;

        for(auto p : points)
            sum += p;

        return static_cast<int>(
            sum / points.size()
        );
    }

    //------------------------------------------
    // 停车
    //------------------------------------------

    void stopCar()
    {
        geometry_msgs::Twist twist;

        twist.linear.x = 0.0;
        twist.angular.z = 0.0;

        cmd_pub_.publish(twist);
    }
};

int main(
    int argc,
    char** argv)
{
    ros::init(
    argc,
    argv,
    "stable_right_follow_cpp"
);

    StableRightFollowNode node;

    ros::spin();

    return 0;
}