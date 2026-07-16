// ucar_controller 底盘驱动接口：负责串口通信、里程计计算、IMU/电池数据解析、LED/速度控制
#ifndef BASE_DRIVER_H_
#define BASE_DRIVER_H_

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Joy.h>
#include <iostream>
#include <serial/serial.h>  // ROS串口包 http://wjwwood.io/serial/doc/1.1.0/index.html
#include <math.h>
#include <fstream>
#include <ucar_controller/data_struct.h>
#include <ucar_controller/GetMaxVel.h>
#include <ucar_controller/SetMaxVel.h>
#include <sensor_msgs/BatteryState.h>
#include <ucar_controller/GetBatteryInfo.h>
#include <ucar_controller/SetLEDMode.h>
#include <boost/thread.hpp>
#include <string>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float64MultiArray.h>
#include <ros/package.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Pose2D.h>
#include <ucar_controller/fdilink_data_struct.h>
#include <ucar_controller/crc_table.h>

using namespace std;

// 常规姿态协方差（静止时使用）
#define ODOM_POSE_COVARIANCE {1e-3, 0, 0, 0, 0, 0,\
                              0, 1e-3, 0, 0, 0, 0,\
                              0, 0, 1e6, 0, 0, 0,\
                              0, 0, 0, 1e6, 0, 0,\
                              0, 0, 0, 0, 1e6, 0,\
                              0, 0, 0, 0, 0, 1e3}

// 高精度姿态协方差（运动时使用）
#define ODOM_POSE_COVARIANCE2 {1e-9, 0, 0, 0, 0, 0,\
                              0, 1e-3, 1e-9, 0, 0, 0,\
                              0, 0, 1e6, 0, 0, 0,\
                              0, 0, 0, 1e6, 0, 0,\
                              0, 0, 0, 0, 1e6, 0,\
                              0, 0, 0, 0, 0, 1e-9}

// 常规速度协方差（静止时使用）
#define ODOM_TWIST_COVARIANCE {1e-3, 0, 0, 0, 0, 0,\
                               0, 1e-3, 0, 0, 0, 0,\
                               0, 0, 1e6, 0, 0, 0,\
                               0, 0, 0, 1e6, 0, 0,\
                               0, 0, 0, 0, 1e6, 0,\
                               0, 0, 0, 0, 0, 1e3}

// 高精度速度协方差（运动时使用）
#define ODOM_TWIST_COVARIANCE2 {1e-9, 0, 0, 0, 0, 0,\
                                0, 1e-3, 1e-9, 0, 0, 0,\
                                0, 0, 1e6, 0, 0, 0,\
                                0, 0, 0, 1e6, 0, 0,\
                                0, 0, 0, 0, 1e6, 0,\
                                0, 0, 0, 0, 0, 1e-9}

namespace ucarController
{
#define Pi 3.1415926

#define WRITE_DATA_LONGTH 8 // 未使用
#define READ_MSG_LONGTH  14 // 底盘回传帧总长度（含校验）
#define READ_DATA_LONGTH 12 // 底盘回传数据域长度
#define WRITE_MSG_LONGTH 16 // 下发帧总长度（含校验）
#define CS_LONGTH 1         // 校验码长度

// LED 控制模式
#define LED_MODE_NORMAL  0  // 常亮
#define LED_MODE_BLINK   1  // 闪烁
#define LED_MODE_BREATH  2  // 呼吸

// 电机控制模式
#define MOTOR_MODE_JOY     0  // 手柄控制模式
#define MOTOR_MODE_CMD     1  // 指令控制模式（默认，接收 /cmd_vel）
#define MOTOR_MODE_MOVE    2  // 移动控制模式（用于特定移动服务）

/**
 * @class baseBringup
 * @brief UCAR 底盘驱动核心类
 *
 * 负责：
 * - 通过串口与底盘 MCU 通信（读写线程分离）
 * - 解析里程计、电池、IMU/AHRS 数据并发布 ROS 话题
 * - 订阅 /cmd_vel 和 /joy 实现速度控制
 * - 提供设置最大速度、停止移动、设置 LED、查询电池等服务
 * - 里程累积并持久化到本地文件
 */
class baseBringup
{
public:
  /**
   * @brief 构造函数：加载参数、初始化串口、创建发布/订阅/服务、启动读写线程
   */
  baseBringup();

  /**
   * @brief 析构函数：关闭串口
   */
  ~baseBringup();

  /**
   * @brief /cmd_vel 话题回调，接收 Twist 速度指令
   * @param msg 线速度/角速度指令
   * @note 仅在 MOTOR_MODE_CMD 模式下生效，同时记录 last_cmd_time_ 用于超时保护
   */
  void velCallback(const geometry_msgs::Twist::ConstPtr& msg);

  /**
   * @brief /joy 手柄话题回调，处理手柄控制逻辑
   * @param msg Joy 消息
   * @note 支持按键切换控制模式（CMD/JOY）、调整速度增益、设定手柄目标速度
   */
  void joyCallback(const sensor_msgs::Joy::ConstPtr& msg);

  /**
   * @brief 获取当前最大速度服务回调
   */
  bool getMaxVelCB(ucar_controller::GetMaxVel::Request &req, ucar_controller::GetMaxVel::Response &res);

  /**
   * @brief 设置最大速度服务回调，同时更新 ROS 参数服务器
   */
  bool setMaxVelCB(ucar_controller::SetMaxVel::Request &req, ucar_controller::SetMaxVel::Response &res);

  /**
   * @brief 停止移动服务回调，退出 MOTOR_MODE_MOVE 并清零速度
   */
  bool stopMoveCB (std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

  /**
   * @brief 获取电池状态服务回调
   * @note 若尚未收到电池数据，percentage 返回 -1
   */
  bool getBatteryStateCB(ucar_controller::GetBatteryInfo::Request &req, ucar_controller::GetBatteryInfo::Response &res);

  /**
   * @brief 设置底盘 LED 灯光服务回调（常亮/闪烁/呼吸模式及 RGB 值）
   */
  bool setLEDCallBack(ucar_controller::SetLEDMode::Request &req, ucar_controller::SetLEDMode::Response &res);

  /**
   * @brief 更新累积里程，变化超过 0.1m 时写入文件与参数服务器
   * @param vx x 方向线速度 (m/s)
   * @param vy y 方向线速度 (m/s)
   * @param dt 时间间隔 (s)
   * @return 是否成功写入
   */
  bool updateMileage(double vx, double vy, double dt);

  /**
   * @brief 从本地文件读取上次保存的累积里程
   * @return 是否读取成功
   */
  bool getMileage();

  /**
   * @brief 解析电池数据并发布 sensor_msgs/BatteryState
   */
  void processBattery();

  /**
   * @brief 串口读取与数据处理主循环（运行在独立线程）
   * @note 不断读取串口帧头，区分底盘数据(0x63 0x76)与 IMU 数据(0xfc...)，分别处理
   */
  void processLoop();

  /**
   * @brief 手柄处理循环（预留接口，当前未实现）
   */
  void joyLoop();

  /**
   * @brief 串口写入循环（运行在独立线程）
   * @note 按 rate_ 频率发送电机脉冲指令与 LED 控制值，含速度限幅与超时保护
   */
  void writeLoop();

  /**
   * @brief 计算写数据包的累加校验和并填入包尾
   * @param len 数据包长度
   */
  void setWriteCS(int len);

  /**
   * @brief 校验读数据包的累加校验和
   * @param len 数据包长度
   * @return 校验是否通过
   */
  bool checkCS(int len);

  /**
   * @brief 无参序列号检查（预留接口，当前未实现）
   */
  bool checkSN();

  /**
   * @brief 解析底盘回传的里程计脉冲，计算速度、位置、发布 odom 话题与 TF
   */
  void processOdometry();

  /**
   * @brief 解析 IMU/AHRS/INSGPS 数据帧并发布 sensor_msgs/Imu 与磁偏航角话题
   * @param head_type 帧类型（0x40 IMU / 0x41 AHRS / 0x50 等）
   */
  void processIMU(uint8_t head_type);

  /**
   * @brief 检查 IMU 相关数据帧的序列号连续性，统计丢包数
   * @param type 帧类型（TYPE_IMU / TYPE_AHRS / TYPE_INSGPS）
   */
  void checkSN(int type);

  /**
   * @brief IMU 话题回调（预留接口，当前未实现）
   */
  void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);

  /**
   * @brief 四元数转欧拉角（预留接口，当前未实现）
   */
  void quaternionToEuler(double Qw, double Qx, double Qy, double Qz, double &pitch, double &roll, double &yaw);

  /**
   * @brief 快速平方根倒数（预留接口，当前未实现）
   */
  float invSqrt(float number);

  /**
   * @brief Mahony AHRS 更新（预留接口，当前未实现）
   */
  void MahonyAHRSupdateIMU(float q[4], float gx, float gy, float gz, float ax, float ay, float az, float delta_s);

  /**
   * @brief 根据磁力计与姿态角计算磁偏航角（mag_yaw）
   */
  void magCalculateYaw(double roll, double pitch, double &magyaw, double magx, double magy, double magz);

  /**
   * @brief 配置串口参数（端口、波特率、数据位、校验、停止位、超时）
   */
  void setSerial();

  /**
   * @brief 循环尝试打开串口，直到成功
   */
  void openSerial();

  /**
   * @brief 打开串口（serial_.open() 的简单封装）
   */
  void callHandle();

  /**
   * @brief 更新序列号（预留接口，当前未实现）
   */
  void updateSN();

  /**
   * @brief 读取串口消息（预留接口，当前未实现）
   * @return 是否读取成功
   */
  bool read_msg();

  ros::NodeHandle nh_; ///< ROS 节点句柄

private:
  /**
   * @brief 下发速度指令到串口（预留接口，当前未实现）
   */
  bool write_msg(double linear_x, double linear_y, double angular_z);

  boost::thread* pJoyThread_;    ///< 手柄线程指针（预留）
  boost::thread* processThread_; ///< 串口读取/数据处理线程指针
  boost::thread* writeThread_;   ///< 串口写入线程指针
  boost::recursive_mutex Control_mutex_; ///< 保护共享数据的递归互斥锁

  // 版本信息（预留）
  std::string ws_version_;      ///< 软件版本
  std::string hw_version_;      ///< 硬件版本
  std::string base_type_name_;  ///< 底盘型号名称

  // 控制模式与开关
  bool provide_odom_tf_;        ///< 是否发布 odom->base_footprint 的 TF
  bool debug_log_;              ///< 是否打印调试日志（ROS_INFO/cout）
  int controll_type_;           ///< 当前电机控制模式（MOTOR_MODE_JOY/CMD/MOVE）
  bool joy_enable_;             ///< 手柄使能标志（预留）

  // 底盘机械参数
  int encode_resolution_;       ///< 编码器分辨率（脉冲/圈）
  double wheel_radius_;         ///< 车轮半径 (m)
  double period_;               ///< 控制周期 (50ms)
  double base_shape_a_, base_shape_b_; ///< 轮子到机器人中心的纵向距离的一半、轮子到机器人中心的横向距离的一半

  // 速度限制
  double linear_speed_min_;     ///< 最小线速度限制（预留）
  double angular_speed_min_;    ///< 最小角速度限制（预留）
  double linear_speed_max_;     ///< 最大线速度限制 (m/s)
  double angular_speed_max_;    ///< 最大角速度限制 (rad/s)

  // 里程与统计
  double Mileage_sum_;          ///< 累积行驶里程 (m)
  double Mileage_last_;         ///< 上次写入文件的里程值 (m)
  int sn_lost_ = 0;             ///< IMU/底盘数据丢包计数
  int cs_error_ = 0;            ///< 校验错误计数（预留）
  uint32_t write_sn_ = 0;       ///< 写帧序列号（预留）
  uint32_t read_sn_  = 0;       ///< 读帧序列号（用于丢包检测）

  // 状态标志
  bool read_first_;             ///< 是否已收到第一帧底盘数据
  bool imu_frist_sn_;           ///< 是否已收到第一帧 IMU 序列号

  // 串口配置
  std::string port_;            ///< 串口设备路径（如 /dev/base_serial_port）
  int baud_;                    ///< 串口波特率
  int serial_timeout_;          ///< 串口读超时 (ms)
  int rate_;                    ///< 写线程循环频率 (Hz)
  double duration_;             ///< 采样时间间隔（预留）

  // 当前位姿
  double x_, y_, th_;           ///< 当前里程计坐标 (m, m, rad)
  nav_msgs::Odometry current_odom_; ///< 当前里程计消息缓存

  // 电池
  float current_battery_percent_; ///< 当前电池电量百分比 [0, 100]，-1 表示未获取

  // LED 参数
  int   led_mode_type_;         ///< LED 模式（NORMAL/BLINK/BREATH）
  float led_frequency_;         ///< LED 闪烁/呼吸频率 (Hz)
  float led_red_value_;         ///< LED 红色分量 [0, 255]
  float led_green_value_;       ///< LED 绿色分量 [0, 255]
  float led_blue_value_;        ///< LED 蓝色分量 [0, 255]
  double led_t_0;               ///< LED 起始时间戳（预留）
  int led_timer;                ///< LED 闪烁计时器（按写循环周期计数）

  // 串口数据包
  pack_write pack_write_;       ///< 下发数据包（含电机脉冲与 LED 值）
  pack_read  pack_read_;        ///< 接收数据包（含编码器脉冲与电池电量）

  // FDILink IMU 数据帧
  FDILink::imu_frame_read   imu_frame_;   ///< IMU 原始帧
  FDILink::ahrs_frame_read  ahrs_frame_;  ///< AHRS 姿态帧
  FDILink::insgps_frame_read insgps_frame_; ///< INSGPS 组合导航帧

  // 手柄/指令/移动 目标速度
  double linear_gain_;          ///< 手柄线速度增益（默认 0.3）
  double twist_gain_;           ///< 手柄角速度增益（默认 0.7）
  double joy_linear_x_,  joy_linear_y_,  joy_angular_z_;  ///< 手柄控制目标速度
  double cmd_linear_x_,  cmd_linear_y_,  cmd_angular_z_;  ///< /cmd_vel 目标速度
  double move_linear_x_, move_linear_y_, move_angular_z_; ///< MOVE 模式目标速度
  double cmd_dt_threshold_;     ///< 指令超时阈值 (s)，超时后自动归零

  // 里程文件路径
  string Mileage_file_name_;        ///< 里程主文件路径
  string Mileage_backup_file_name_; ///< 里程备份文件路径

  // TF 与坐标系名称
  string base_frame_, odom_frame_; ///< base 与 odom 坐标系名称
  string imu_frame_id_;            ///< IMU 坐标系名称

  // 话题名称
  string vel_topic_, joy_topic_, odom_topic_;   ///< 速度/cmd_vel 遥控/joy 里程计话题/odom
  string battery_topic_;                        ///< 电池话题/battery_state
  string imu_topic_, mag_pose_2d_topic_;        ///< IMU/imu 磁偏航话题/mag_pose_2d

  // 时间戳
  ros::Time current_time_, last_time_;  ///< 当前与上一帧里程计时间
  ros::Time last_cmd_time_;             ///< 上次收到速度指令的时间

  // Publisher
  ros::Publisher odom_pub_;       ///< 里程计发布器
  ros::Publisher mileage_pub_;    ///< 里程发布器（预留）
  ros::Publisher battery_pub_;    ///< 电池状态发布器
  ros::Publisher imu_pub_;        ///< IMU 数据发布器
  ros::Publisher mag_pose_pub_;   ///< 磁偏航角 2D 姿态发布器

  // Subscriber
  ros::Subscriber vel_sub_;       ///< /cmd_vel 订阅器
  ros::Subscriber joy_sub_;       ///< /joy 订阅器

  // Service Server
  ros::ServiceServer set_max_vel_server_;       ///< 设置最大速度服务
  ros::ServiceServer get_max_vel_server_;       ///< 获取最大速度服务
  ros::ServiceServer stop_move_server_;         ///< 停止移动服务
  ros::ServiceServer get_battery_state_server_; ///< 获取电池状态服务
  ros::ServiceServer set_led_server_;           ///< 设置 LED 服务

  tf::TransformBroadcaster odom_broadcaster_;   ///< odom->base_footprint TF 广播器

  serial::Serial serial_; // 串口通信对象

};//baseBringup
} //ucarController


#endif
