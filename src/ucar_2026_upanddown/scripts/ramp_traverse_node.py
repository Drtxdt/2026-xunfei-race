#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""国赛坡道（上-平-下）通过节点，带完整中文运行日志。

2D 雷达在坡上必然失效，本节点：

1. 关闭雷达门控（冻结 AMCL，阻止坡道被标记为幻影墙）；
2. 低速直行过坡，航向保持；
3. 用 IMU 俯仰角分段（22°上坡 -> 平路 -> 25°下坡）；
4. 下坡结束后多走轴距余量，停稳；
5. 重新打开雷达门控，导航无缝恢复。

调用方（国赛总控）必须在 ~start 之前取消 move_base 目标，
并且只能在状态 DONE 之后恢复导航。

日志约定（方便 grep）：
  【坡道·初始化】 启动参数自检
  【坡道·服务】   start/abort 服务请求
  【坡道·门控】   雷达门控开关与验证
  【坡道·传感】   IMU/里程计就绪检查
  【坡道·状态】   状态机切换
  【坡道·分段】   坡道段切换事件（含每段里程/耗时）
  【坡道·巡检】   控制循环节流日志（pitch/航向/里程/速度）
  【坡道·余量】   出坡余量段
  【坡道·停止】   停车过程
  【坡道·完成】   成功总结
  【坡道·故障】   任何 FAULT 及急停
"""

from __future__ import annotations

import json
import math
import threading

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger, TriggerResponse

from ucar_2026_upanddown.logic import (
    HeadingHoldController,
    PitchFilter,
    RampSegmenter,
    SoftSpeedProfile,
    distance_budget_exceeded,
    normalize_angle,
    path_length,
    rpy_from_quaternion,
    validate_ramp_config,
)


TERMINAL_STATES = frozenset(("DONE", "FAULT", "ABORTED"))
HEADING_SOURCES = ("imu_orientation", "gyro", "odom")

SEGMENT_ZH = {
    "level": "坡前平地",
    "up": "上坡段",
    "plateau": "坡顶平路段",
    "down": "下坡段",
    "complete": "下坡完成",
}


class StageError(Exception):
    pass


def yaw_from_msg_quaternion(orientation):
    _roll, _pitch, yaw = rpy_from_quaternion(
        orientation.x, orientation.y, orientation.z, orientation.w)
    return yaw


def odom_yaw(msg):
    return yaw_from_msg_quaternion(msg.pose.pose.orientation)


class RampTraverseNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.state = "WAIT_START"
        self.fault_reason = ""
        self.abort_requested = False
        self.start_event = threading.Event()

        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.status_topic = rospy.get_param(
            "~status_topic", "/ramp_traverse/status")
        self.gate_service = rospy.get_param("~gate_service", "/scan_gate/set_open")
        self.gate_status_topic = rospy.get_param(
            "~gate_status_topic", "/scan_gate/status")
        self.gate_timeout_sec = float(rospy.get_param("~gate_timeout_sec", 5.0))
        self.reopen_gate_on_finish = bool(rospy.get_param(
            "~reopen_gate_on_finish", True))
        self.gate_reopen_verify_sec = float(rospy.get_param(
            "~gate_reopen_verify_sec", 2.0))
        self.rate_hz = float(rospy.get_param("~control_rate_hz", 20.0))
        self.log_interval_sec = max(
            0.0, float(rospy.get_param("~log_interval_sec", 1.0)))
        self._last_log_at = 0.0
        self._last_progress_log_at = 0.0

        self.pitch_sign = float(rospy.get_param("~pitch_sign", 1.0))
        self.level_offset_deg = float(rospy.get_param("~level_offset_deg", 0.0))
        self.pitch_filter = PitchFilter(
            rospy.get_param("~pitch_filter_window", 5))

        self.segmenter = RampSegmenter(
            up_enter_deg=rospy.get_param("~up_enter_deg", 8.0),
            up_exit_deg=rospy.get_param("~up_exit_deg", 3.0),
            down_enter_deg=rospy.get_param("~down_enter_deg", -8.0),
            down_exit_deg=rospy.get_param("~down_exit_deg", -3.0),
            confirm_frames=rospy.get_param("~confirm_frames", 3),
        )
        validate_ramp_config(
            self.segmenter.up_enter_deg, self.segmenter.up_exit_deg,
            self.segmenter.down_enter_deg, self.segmenter.down_exit_deg,
            rospy.get_param("~nominal_ramp_length_m", 1.5),
            rospy.get_param("~exit_extra_m", 0.28),
        )

        self.speed_approach = float(rospy.get_param("~speed_approach_mps", 0.12))
        self.speed_up = float(rospy.get_param("~speed_up_mps", 0.16))
        self.speed_plateau = float(rospy.get_param("~speed_plateau_mps", 0.15))
        self.speed_down = float(rospy.get_param("~speed_down_mps", 0.14))
        self.speed_exit = float(rospy.get_param("~speed_exit_mps", 0.12))
        if min(self.speed_approach, self.speed_up, self.speed_plateau,
               self.speed_down, self.speed_exit) <= 0.0:
            raise ValueError("ramp speeds must be positive")
        self.profile = SoftSpeedProfile(
            accel_limit=rospy.get_param("~accel_limit_mps2", 0.25),
            decel_limit=rospy.get_param("~decel_limit_mps2", 0.40),
        )

        self.heading_source = str(rospy.get_param(
            "~heading_source", "imu_orientation")).strip().lower()
        if self.heading_source not in HEADING_SOURCES:
            raise ValueError(
                "heading_source must be one of {}".format(HEADING_SOURCES))
        self.heading_ctrl = HeadingHoldController(
            kp=rospy.get_param("~heading_kp", 1.6),
            max_angular=rospy.get_param("~heading_max_angular", 0.35),
            deadband_rad=math.radians(
                rospy.get_param("~heading_deadband_deg", 1.5)),
        )

        self.nominal_length_m = float(rospy.get_param(
            "~nominal_ramp_length_m", 1.5))
        self.distance_margin_m = float(rospy.get_param(
            "~distance_margin_m", 0.90))
        self.exit_extra_m = float(rospy.get_param("~exit_extra_m", 0.28))
        self.imu_timeout_sec = float(rospy.get_param("~imu_timeout_sec", 1.0))
        self.odom_timeout_sec = float(rospy.get_param("~odom_timeout_sec", 1.0))
        self.traverse_timeout_sec = float(rospy.get_param(
            "~traverse_timeout_sec", 120.0))
        self.post_stop_settle_sec = float(rospy.get_param(
            "~post_stop_settle_sec", 0.6))
        self.post_stop_zero_sec = float(rospy.get_param(
            "~post_stop_zero_sec", 1.5))
        self.require_pitch_signature = bool(rospy.get_param(
            "~require_pitch_signature", True))
        self.min_up_pitch_seen_deg = float(rospy.get_param(
            "~min_up_pitch_seen_deg", 12.0))
        self.min_down_pitch_seen_deg = float(rospy.get_param(
            "~min_down_pitch_seen_deg", -12.0))

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True)
        self.gate_status = {}
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=10)
        rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=10)
        rospy.Subscriber(
            self.gate_status_topic, String, self.gate_status_callback,
            queue_size=5)
        rospy.Service("~start", Trigger, self.start_service)
        rospy.Service("~abort", Trigger, self.abort_service)
        rospy.on_shutdown(self.shutdown)

        self.imu_msg = None
        self.imu_at = 0.0
        self.odom_msg = None
        self.odom_at = 0.0
        # execute() 里会重置；这里先给初值，保证启动时 publish_status 可用。
        self.traveled_m = 0.0

        self._log_startup_parameters()
        self.publish_status("等待启动指令")
        rospy.loginfo("【坡道】节点就绪, 等待 /ramp_traverse/start 服务调用")
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    # ------------------------------ 启动自检 -------------------------------
    def _log_startup_parameters(self):
        """启动时打印全部关键参数, 供赛后日志核对 yaml 是否生效。"""
        if not bool(rospy.get_param("~config_loaded", False)):
            rospy.logwarn(
                "【坡道·初始化】警告: 未检测到 config_loaded 标记! "
                "ramp_traverse.yaml 没有被加载, 当前使用代码内置默认参数, "
                "比赛前必须修复 launch 文件!")
        else:
            rospy.loginfo("【坡道·初始化】ramp_traverse.yaml 已加载 (config_loaded=true)")
        rospy.loginfo(
            "【坡道·初始化】参数自检:\n"
            "  话题: imu=%s odom=%s cmd_vel=%s 状态=%s\n"
            "  门控: 服务=%s 超时=%.1fs 完成后开门=%s 开门验证=%.1fs\n"
            "  俯仰: pitch符号=%.1f 平地零偏=%.2f° 滤波窗口=%d帧\n"
            "  分段阈值: 上坡进入≥%.1f° 上坡退出≤%.1f° 下坡进入≤%.1f° "
            "下坡退出≥%.1f° 确认帧数=%d\n"
            "  速度(m/s): 接近=%.2f 上坡=%.2f 平路=%.2f 下坡=%.2f 出坡余量=%.2f "
            "| 加速上限=%.2f 减速上限=%.2f (m/s²)\n"
            "  航向: 来源=%s P增益=%.2f 角速度限幅=%.2frad/s 死区=%.1f°\n"
            "  距离: 坡长名义=%.2fm 距离硬预算=%.2fm 出坡余量=%.2fm\n"
            "  看门狗: IMU失联阈值=%.1fs odom失联阈值=%.1fs 总超时=%.0fs\n"
            "  完成校验: 启用=%s 需见过最大≥%.1f° 且 最小≤%.1f°",
            self.imu_topic, self.odom_topic, self.cmd_topic, self.status_topic,
            self.gate_service, self.gate_timeout_sec,
            self.reopen_gate_on_finish, self.gate_reopen_verify_sec,
            self.pitch_sign, self.level_offset_deg,
            self.pitch_filter.window,
            self.segmenter.up_enter_deg, self.segmenter.up_exit_deg,
            self.segmenter.down_enter_deg, abs(self.segmenter.down_exit_deg),
            self.segmenter.confirm_frames,
            self.speed_approach, self.speed_up, self.speed_plateau,
            self.speed_down, self.speed_exit,
            self.profile.accel_limit, self.profile.decel_limit,
            self.heading_source, self.heading_ctrl.kp,
            self.heading_ctrl.max_angular,
            math.degrees(self.heading_ctrl.deadband_rad),
            self.nominal_length_m, self.distance_margin_m, self.exit_extra_m,
            self.imu_timeout_sec, self.odom_timeout_sec,
            self.traverse_timeout_sec,
            self.require_pitch_signature,
            self.min_up_pitch_seen_deg, self.min_down_pitch_seen_deg,
        )

    # ------------------------------ callbacks ------------------------------
    def imu_callback(self, msg):
        with self.lock:
            self.imu_msg = msg
            self.imu_at = rospy.get_time()

    def odom_callback(self, msg):
        with self.lock:
            self.odom_msg = msg
            self.odom_at = rospy.get_time()

    def gate_status_callback(self, msg):
        try:
            with self.lock:
                self.gate_status = json.loads(msg.data)
        except (TypeError, ValueError):
            pass

    # ------------------------------ services -------------------------------
    def start_service(self, _request):
        with self.lock:
            if self.state not in ("WAIT_START",) + TERMINAL_STATES:
                rospy.logwarn(
                    "【坡道·服务】拒绝启动: 当前状态为 %s (仅 WAIT_START/%s 可启动)",
                    self.state, "/".join(TERMINAL_STATES))
                return TriggerResponse(
                    success=False,
                    message="cannot start from state {}".format(self.state))
            self.state = "WAIT_START"
            self.fault_reason = ""
            self.abort_requested = False
        rospy.loginfo("【坡道·服务】收到启动请求, 开始过坡流程")
        self.start_event.set()
        return TriggerResponse(success=True, message="ramp traverse started")

    def abort_service(self, _request):
        with self.lock:
            self.abort_requested = True
        rospy.logwarn("【坡道·服务】收到终止请求, 将在下一个控制周期安全停车")
        return TriggerResponse(success=True, message="abort requested")

    # ------------------------------ helpers --------------------------------
    def publish_status(self, detail="", **extra):
        with self.lock:
            payload = {
                "state": self.state,
                "detail": detail,
                "segment": self.segmenter.state,
                "pitch_deg": self.pitch_filter.value(),
                "max_pitch_deg": self.segmenter.max_pitch_deg,
                "min_pitch_deg": self.segmenter.min_pitch_deg,
                "distance_m": self.traveled_m,
                "speed_cmd": self.profile.current,
                "error": self.fault_reason,
                "stamp": rospy.get_time(),
            }
        payload.update(extra)
        self.status_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    def publish_zero(self):
        try:
            self.cmd_pub.publish(Twist())
        except rospy.exceptions.ROSException:
            pass

    def set_gate(self, open_state):
        action = "打开" if open_state else "关闭"
        rospy.loginfo("【坡道·门控】正在%s雷达门控 (冻结/恢复 AMCL)...", action)
        try:
            rospy.wait_for_service(self.gate_service, timeout=self.gate_timeout_sec)
            response = rospy.ServiceProxy(self.gate_service, SetBool)(open_state)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("雷达门控服务调用失败: {}".format(exc))
        if not response.success:
            raise StageError("雷达门控拒绝请求: {}".format(response.message))
        rospy.loginfo("【坡道·门控】门控已%s (响应: %s)", action, response.message)
        return True

    def verify_gate_reopened(self):
        """开门后确认扫描确实恢复流动, 否则告警。"""
        if self.gate_reopen_verify_sec <= 0.0:
            return
        with self.lock:
            before = int(self.gate_status.get("forwarded", 0))
        deadline = rospy.get_time() + self.gate_reopen_verify_sec
        while rospy.get_time() < deadline and not rospy.is_shutdown():
            with self.lock:
                after = int(self.gate_status.get("forwarded", 0))
            if after > before:
                rospy.loginfo(
                    "【坡道·门控】开门验证通过: 转发计数 %d -> %d, 雷达数据已恢复流动",
                    before, after)
                return
            rospy.sleep(0.1)
        rospy.logwarn(
            "【坡道·门控】开门验证失败: %.1fs 内转发计数无增长 "
            "(before=%d), 请检查雷达发布者; 继续流程",
            self.gate_reopen_verify_sec, before)

    def set_state(self, state, detail="", **extra):
        with self.lock:
            self.state = state
        rospy.loginfo("【坡道·状态】%s - %s", state, detail)
        self.publish_status(detail, **extra)

    def current_imu(self):
        with self.lock:
            return self.imu_msg, self.imu_at

    def current_odom(self):
        with self.lock:
            return self.odom_msg, self.odom_at

    def check_abort(self):
        with self.lock:
            if self.abort_requested:
                raise StageError("收到外部终止请求 (abort)")

    # ------------------------------ main loop ------------------------------
    def run(self):
        while not rospy.is_shutdown():
            self.start_event.wait()
            self.start_event.clear()
            if rospy.is_shutdown():
                return
            try:
                self.execute()
            except StageError as exc:
                self.emergency_stop(str(exc))
            except Exception as exc:  # noqa: BLE001 - 任何异常都安全停车
                rospy.logerr("【坡道·故障】捕获未预期异常: %r", exc)
                self.emergency_stop("未预期异常: {}".format(exc))
            rospy.loginfo("【坡道】本轮流程结束, 节点回到待命状态")

    def emergency_stop(self, reason):
        rospy.logerr("【坡道·故障】触发急停! 原因: %s", reason)
        rospy.logerr("【坡道·故障】急停时数据快照: 总里程=%.2fm 峰值上坡=%.1f° "
                     "峰值下坡=%.1f° 当前段=%s",
                     getattr(self, "traveled_m", 0.0),
                     self.segmenter.max_pitch_deg,
                     self.segmenter.min_pitch_deg,
                     SEGMENT_ZH.get(self.segmenter.state, self.segmenter.state))
        self.profile.reset(0.0)
        rospy.loginfo("【坡道·故障】持续发送零速度停车中...")
        for _ in range(10):
            self.publish_zero()
            rospy.sleep(0.05)
        try:
            self.set_gate(True)
        except StageError as gate_exc:
            reason = "{}; 急停后重新开门也失败: {}".format(reason, gate_exc)
            rospy.logerr("【坡道·故障】%s", reason)
        with self.lock:
            self.state = "FAULT"
            self.fault_reason = reason
        self.publish_status("急停: " + reason)

    def execute(self):
        self.pitch_filter.reset()
        self.segmenter.reset()
        self.profile.reset(0.0)
        self.traveled_m = 0.0
        self.exit_origin_m = None
        self.segment_marks = []   # (段名, 里程, 时刻) 边界记录
        self._last_segment = "level"

        self.set_state("GATE_CLOSING", "关闭雷达门控, 冻结 AMCL 定位")
        self.set_gate(False)

        rospy.loginfo("【坡道·传感】检查 IMU/里程计数据新鲜度...")
        imu_msg, imu_at = self.current_imu()
        odom_msg, odom_at = self.current_odom()
        now = rospy.get_time()
        if imu_at <= 0.0 or now - imu_at > self.imu_timeout_sec:
            raise StageError(
                "IMU 无新鲜数据 (话题 {} 最后更新 {:.2f}s 前, 阈值 {:.1f}s)".format(
                    self.imu_topic, max(0.0, now - imu_at), self.imu_timeout_sec))
        if odom_at <= 0.0 or now - odom_at > self.odom_timeout_sec:
            raise StageError(
                "里程计无新鲜数据 (话题 {} 最后更新 {:.2f}s 前, 阈值 {:.1f}s)".format(
                    self.odom_topic, max(0.0, now - odom_at),
                    self.odom_timeout_sec))
        rospy.loginfo("【坡道·传感】IMU 就绪 (延迟 %.0fms), 里程计就绪 (延迟 %.0fms)",
                      1000.0 * max(0.0, now - imu_at),
                      1000.0 * max(0.0, now - odom_at))

        first_pitch = self.raw_pitch_deg(imu_msg)
        if first_pitch is None:
            raise StageError("IMU 四元数无效, 无法解算俯仰角")
        rospy.loginfo(
            "【坡道·传感】起始俯仰角 = %+.1f° (平地应≈0°; 若在坡上属正常; "
            "符号不对请检查 yaml 的 pitch_sign)", first_pitch)

        start_pose = self.pose_tuple(odom_msg)
        self.target_yaw = self.seed_heading(imu_msg, odom_msg)
        self.gyro_yaw = self.target_yaw
        self.last_stamp = imu_msg.header.stamp.to_sec() or imu_at

        self.set_state(
            "TRAVERSE",
            "航向保持已锁定 目标航向={:.1f}° 航向源={}".format(
                math.degrees(self.target_yaw), self.heading_source))

        rate = rospy.Rate(self.rate_hz)
        started_at = rospy.get_time()
        last_control = started_at
        while not rospy.is_shutdown():
            self.check_abort()
            now = rospy.get_time()
            dt = max(0.0, min(0.2, now - last_control))
            last_control = now

            imu_msg, imu_at = self.current_imu()
            odom_msg, odom_at = self.current_odom()
            if now - imu_at > self.imu_timeout_sec:
                raise StageError(
                    "IMU 数据失联 {:.2f}s (超过阈值 {:.1f}s), 安全停车".format(
                        now - imu_at, self.imu_timeout_sec))
            if now - odom_at > self.odom_timeout_sec:
                raise StageError(
                    "里程计数据失联 {:.2f}s (超过阈值 {:.1f}s), 安全停车".format(
                        now - odom_at, self.odom_timeout_sec))
            if now - started_at > self.traverse_timeout_sec:
                raise StageError(
                    "过坡总耗时 {:+.1f}s 超过硬超时 {:.0f}s".format(
                        now - started_at, self.traverse_timeout_sec))

            raw_pitch = self.raw_pitch_deg(imu_msg)
            if raw_pitch is None:
                raise StageError("IMU 四元数无效, 无法解算俯仰角")
            pitch_deg = self.pitch_filter.push(raw_pitch)
            segment = self.segmenter.update(pitch_deg)

            pose = self.pose_tuple(odom_msg)
            self.traveled_m = path_length(start_pose, pose)
            if distance_budget_exceeded(
                    self.traveled_m, self.nominal_length_m,
                    self.distance_margin_m):
                raise StageError(
                    "里程预算超限: 已走 {:.2f}m > 名义 {:.2f}m + 余量 {:.2f}m, "
                    "当前段={} (疑似打滑/未上坡/分段异常)".format(
                        self.traveled_m, self.nominal_length_m,
                        self.distance_margin_m,
                        SEGMENT_ZH.get(segment, segment)))

            # 分段切换事件: 显著日志 + 边界记录 (用于完成总结)
            if segment != self._last_segment:
                self._log_segment_transition(
                    self._last_segment, segment, pitch_deg, now, started_at)
                self._last_segment = segment

            if segment == "complete":
                if self.require_pitch_signature and not \
                        self.segmenter.pitch_signature_valid(
                            self.min_up_pitch_seen_deg,
                            self.min_down_pitch_seen_deg):
                    raise StageError(
                        "坡度签名校验失败: 全程最大 {:+.1f}° / 最小 {:+.1f}° "
                        "(要求 ≥{:.1f}° 且 ≤{:.1f}°), 疑似没有真正过坡".format(
                            self.segmenter.max_pitch_deg,
                            self.segmenter.min_pitch_deg,
                            self.min_up_pitch_seen_deg,
                            self.min_down_pitch_seen_deg))
                if self.exit_origin_m is None:
                    self.exit_origin_m = self.traveled_m
                    self.set_state(
                        "EXIT",
                        "下坡完成, 起点里程 {:.2f}m, 需再前进 {:.2f}m 轴距余量".format(
                            self.exit_origin_m, self.exit_extra_m),
                        segment=segment)
            if self.state == "EXIT":
                if self.traveled_m - self.exit_origin_m >= self.exit_extra_m:
                    rospy.loginfo(
                        "【坡道·余量】余量段完成: 实际多走 %.2fm (要求 %.2fm)",
                        self.traveled_m - self.exit_origin_m, self.exit_extra_m)
                    break
                target_speed = self.speed_exit
            else:
                target_speed = {
                    "level": self.speed_approach,
                    "up": self.speed_up,
                    "plateau": self.speed_plateau,
                    "down": self.speed_down,
                }[segment]

            yaw_error = self.heading_error(imu_msg, odom_msg)
            speed = self.profile.update(target_speed, dt)
            angular = self.heading_ctrl.command(yaw_error)
            twist = Twist()
            twist.linear.x = speed
            twist.angular.z = angular
            self.cmd_pub.publish(twist)
            now_log = rospy.get_time()
            if now_log - self._last_log_at >= self.log_interval_sec:
                self._last_log_at = now_log
                rospy.loginfo(
                    "【坡道·巡检】段=%s 原始pitch=%+.1f° 滤波pitch=%+.1f° "
                    "里程=%.2fm 速度=%.2fm/s 航向误差=%+.1f° 角速度=%+.3frad/s "
                    "已用时=%.1fs",
                    SEGMENT_ZH.get(segment, segment), raw_pitch, pitch_deg,
                    self.traveled_m, speed, math.degrees(yaw_error), angular,
                    now_log - started_at)
            self.publish_status(
                "segment={} pitch={:.1f}deg dist={:.2f}m".format(
                    segment, pitch_deg, self.traveled_m))
            rate.sleep()

        # 停车阶段: 先按减速斜率降速, 再持续零速度保持。
        self.set_state("STOPPING", "坡道远端停车: 减速 -> 零速保持")
        deadline = rospy.get_time() + 2.0
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            self.check_abort()
            speed = self.profile.update(0.0, 0.05)
            twist = Twist()
            twist.linear.x = speed
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
        rospy.loginfo("【坡道·停止】减速完成, 零速度保持 %.1fs (防坡上溜车)",
                      self.post_stop_zero_sec)
        settle_until = rospy.get_time() + self.post_stop_zero_sec
        while not rospy.is_shutdown() and rospy.get_time() < settle_until:
            self.check_abort()
            self.publish_zero()
            rospy.sleep(0.05)

        if self.reopen_gate_on_finish:
            self.set_state("GATE_OPENING", "重新打开雷达门控, 恢复 AMCL")
            self.set_gate(True)
            self.verify_gate_reopened()
            rospy.sleep(max(0.0, self.post_stop_settle_sec))

        with self.lock:
            self.state = "DONE"
        self._log_done_summary(started_at)

    # ------------------------------ 日志辅助 -------------------------------
    def _log_segment_transition(self, previous, current, pitch_deg,
                                now, started_at):
        """段切换: 打印上一段的里程/耗时, 并记录边界。"""
        self.segment_marks.append((current, self.traveled_m, now))
        rospy.loginfo(
            "【坡道·分段】>>> %s -> %s | 滤波pitch=%+.1f° | 累计里程=%.2fm | "
            "累计用时=%.1fs",
            SEGMENT_ZH.get(previous, previous),
            SEGMENT_ZH.get(current, current),
            pitch_deg, self.traveled_m, now - started_at)

    def _log_done_summary(self, started_at):
        """成功总结: 里程/耗时/坡度极值/逐段统计。"""
        total_sec = rospy.get_time() - started_at
        rospy.loginfo(
            "【坡道·完成】过坡成功! 总里程=%.2fm 总耗时=%.1fs "
            "峰值上坡=%+.1f° 峰值下坡=%+.1f°",
            self.traveled_m, total_sec,
            self.segmenter.max_pitch_deg, self.segmenter.min_pitch_deg)
        marks = [("level", 0.0, started_at)] + list(self.segment_marks)
        for index in range(len(marks) - 1):
            name, start_m, start_t = marks[index]
            _nxt, end_m, end_t = marks[index + 1]
            rospy.loginfo(
                "【坡道·完成】  - %s: 段里程=%.2fm 段耗时=%.1fs",
                SEGMENT_ZH.get(name, name), end_m - start_m, end_t - start_t)
        rospy.loginfo(
            "【坡道·完成】距离预算使用率: %.0f%% (名义 %.2fm, 硬预算 %.2fm)",
            100.0 * self.traveled_m / (self.nominal_length_m +
                                       self.distance_margin_m),
            self.nominal_length_m, self.distance_margin_m)
        self.publish_status(
            "过坡成功: 总里程 {:.2f}m 总耗时 {:.1f}s".format(
                self.traveled_m, total_sec),
            distance_m=self.traveled_m,
        )

    # ------------------------------ sensing --------------------------------
    def pose_tuple(self, odom_msg):
        position = odom_msg.pose.pose.position
        return (position.x, position.y, odom_yaw(odom_msg))

    def raw_pitch_deg(self, imu_msg):
        """符号修正 + 零偏扣除后的原始（未滤波）俯仰角, 单位度。"""
        _roll, pitch, _yaw = rpy_from_quaternion(
            imu_msg.orientation.x, imu_msg.orientation.y,
            imu_msg.orientation.z, imu_msg.orientation.w)
        if not math.isfinite(pitch):
            return None
        return math.degrees(pitch) * self.pitch_sign - self.level_offset_deg

    def seed_heading(self, imu_msg, odom_msg):
        if self.heading_source == "imu_orientation":
            yaw = yaw_from_msg_quaternion(imu_msg.orientation)
            if math.isfinite(yaw):
                return yaw
        rospy.logwarn(
            "【坡道·传感】航向源 %s 不可用, 回退到里程计航向",
            self.heading_source)
        return odom_yaw(odom_msg)

    def heading_error(self, imu_msg, odom_msg):
        if self.heading_source == "imu_orientation":
            yaw = yaw_from_msg_quaternion(imu_msg.orientation)
            if math.isfinite(yaw):
                return normalize_angle(self.target_yaw - yaw)
        elif self.heading_source == "gyro":
            stamp = imu_msg.header.stamp.to_sec() or rospy.get_time()
            dt = max(0.0, min(0.2, stamp - self.last_stamp))
            if dt > 0.0:
                self.gyro_yaw = normalize_angle(
                    self.gyro_yaw + imu_msg.angular_velocity.z * dt)
                self.last_stamp = stamp
            return normalize_angle(self.target_yaw - self.gyro_yaw)
        return normalize_angle(self.target_yaw - odom_yaw(odom_msg))

    def shutdown(self):
        rospy.loginfo("【坡道】节点关闭: 停车并交还控制权")
        with self.lock:
            self.abort_requested = True
        self.publish_zero()


if __name__ == "__main__":
    rospy.init_node("ramp_traverse")
    RampTraverseNode()
    rospy.spin()
