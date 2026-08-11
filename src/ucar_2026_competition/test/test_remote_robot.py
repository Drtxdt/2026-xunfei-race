#!/usr/bin/env python3
import ast
import os
import sys
import unittest
import xml.etree.ElementTree as ET
from unittest import mock


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_SRC = os.path.join(PACKAGE_ROOT, "src")
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from ucar_2026_competition.remote_robot import (
    RobotDeploymentError,
    build_remote_agent_command,
    decode_launch_arguments,
    encode_launch_arguments,
    remote_preflight_script,
    repository_revision,
    resolve_physical_launch_arguments,
    ssh_base_command,
    validate_reusable_robot_master,
)


class RemoteRobotHelperTest(unittest.TestCase):
    def test_launch_argument_round_trip_is_safe_and_typed_as_strings(self):
        payload = encode_launch_arguments({
            "debug": False,
            "sim_bridge_port": 26003,
            "traffic_y": "-3.10",
        })
        self.assertEqual(decode_launch_arguments(payload), {
            "debug": "false",
            "sim_bridge_port": "26003",
            "traffic_y": "-3.10",
        })
        with self.assertRaises(RobotDeploymentError):
            encode_launch_arguments({"bad-name": "value"})
        with self.assertRaises(RobotDeploymentError):
            encode_launch_arguments({"debug": "true\ncommand"})

    def test_remote_command_uses_linux_paths_and_local_robot_master(self):
        command = build_remote_agent_command(
            "/home/ucar/2026-xunfei-race",
            "/home/ucar/.config/ucar_2026/robot_env.sh",
            "a" * 40,
            {"enable_simulation": True},
            30,
        )
        self.assertIn("/home/ucar/2026-xunfei-race/devel/setup.bash", command)
        self.assertNotIn("\\", command)
        self.assertIn("ROS_MASTER_URI=http://127.0.0.1:11311", command)
        self.assertIn("robot_competition_agent.py", command)
        self.assertIn("--startup-timeout", command)

    def test_ssh_is_noninteractive_and_liveness_checked(self):
        command = ssh_base_command("ucar@robot", 8)
        joined = " ".join(command)
        self.assertIn("BatchMode=yes", joined)
        self.assertIn("StrictHostKeyChecking=accept-new", joined)
        self.assertIn("ServerAliveInterval=2", joined)
        self.assertEqual(command[-1], "ucar@robot")
        debug_command = ssh_base_command("ucar@robot", 8, forward_x11=True)
        self.assertIn("-Y", debug_command)

    def test_local_bridge_uses_ssh_client_but_external_debug_keeps_override(self):
        local = resolve_physical_launch_arguments(
            {"enable_simulation": "true", "use_external_sim_bridge": "false",
             "sim_bridge_host": ""},
            "192.168.1.20 40000 192.168.1.6 22",
        )
        self.assertEqual(local["sim_bridge_host"], "192.168.1.20")
        self.assertNotIn("use_external_sim_bridge", local)
        external = resolve_physical_launch_arguments(
            {"enable_simulation": "true", "use_external_sim_bridge": "true",
             "sim_bridge_host": "10.0.0.5"},
            "",
        )
        self.assertEqual(external["sim_bridge_host"], "10.0.0.5")
        disabled = resolve_physical_launch_arguments(
            {"enable_simulation": "false", "use_external_sim_bridge": "false"}, "")
        self.assertNotIn("use_external_sim_bridge", disabled)

    def test_preflight_checks_files_permissions_clean_tree_and_revision(self):
        script = remote_preflight_script(
            "/home/ucar/2026-xunfei-race",
            "/home/ucar/.config/ucar_2026/robot_env.sh",
            "b" * 40,
        )
        self.assertIn("mode != 0o600", script)
        self.assertIn("status', '--porcelain", script)
        self.assertIn("competition revision mismatch", script)
        self.assertIn("physical_competition.launch", script)

    def test_local_dirty_repository_is_rejected(self):
        clean_revision = mock.Mock(returncode=0, stdout="abc\n", stderr="")
        dirty_status = mock.Mock(returncode=0, stdout=" M file\n", stderr="")
        runner = mock.Mock(side_effect=[clean_revision, dirty_status])
        with self.assertRaises(RobotDeploymentError):
            repository_revision("/workspace", runner=runner)

    def test_vendor_roscore_is_reused_but_running_competition_is_rejected(self):
        vendor_master = mock.Mock(returncode=0, stdout="/rosout\n", stderr="")
        self.assertEqual(
            validate_reusable_robot_master(runner=mock.Mock(return_value=vendor_master)),
            {"/rosout"},
        )
        occupied_master = mock.Mock(
            returncode=0,
            stdout="/rosout\n/competition_flow\n",
            stderr="",
        )
        with self.assertRaisesRegex(RobotDeploymentError, "already running"):
            validate_reusable_robot_master(
                runner=mock.Mock(return_value=occupied_master))

    def test_unhealthy_existing_robot_master_is_rejected(self):
        failed = mock.Mock(returncode=1, stdout="", stderr="master unavailable")
        with self.assertRaisesRegex(RobotDeploymentError, "not healthy"):
            validate_reusable_robot_master(runner=mock.Mock(return_value=failed))


class RemoteRobotLaunchContractTest(unittest.TestCase):
    def test_remote_output_is_silenced_before_shutdown_signals(self):
        path = os.path.join(PACKAGE_ROOT, "scripts", "robot_supervisor.py")
        with open(path, "r", encoding="utf-8") as stream:
            tree = ast.parse(stream.read(), filename=path)
        supervisor = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == "RobotSupervisor")
        shutdown = next(node for node in supervisor.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "shutdown")
        calls = [node for node in ast.walk(shutdown) if isinstance(node, ast.Call)]
        silence_line = next(node.lineno for node in calls
                            if isinstance(node.func, ast.Attribute)
                            and node.func.attr == "set")
        stop_line = next(node.lineno for node in calls
                         if isinstance(node.func, ast.Attribute)
                         and node.func.attr == "_stop_process_group")
        self.assertLess(silence_line, stop_line)

    def test_full_launch_forwards_physical_arguments_to_required_supervisor(self):
        root = ET.parse(os.path.join(PACKAGE_ROOT, "launch", "full_competition.launch")).getroot()
        node = root.find("node[@type='robot_supervisor.py']")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrib["required"], "true")
        params = {item.attrib["name"]: item.attrib.get("value")
                  for item in node.findall("param")}
        self.assertEqual(params["robot_ssh_target"], "$(arg robot_ssh_target)")
        launch_args = node.find("rosparam[@param='physical_launch_arguments']")
        self.assertIsNotNone(launch_args)
        self.assertEqual(launch_args.attrib["subst_value"], "true")
        self.assertIn("use_external_sim_bridge", launch_args.text)
        self.assertIn("coverage_goal_soft_timeout_sec", launch_args.text)

    def test_agent_owns_process_group_and_watches_ssh_stdin(self):
        path = os.path.join(PACKAGE_ROOT, "scripts", "robot_competition_agent.py")
        with open(path, "r", encoding="utf-8") as stream:
            tree = ast.parse(stream.read(), filename=path)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(
            isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"
            and any(keyword.arg == "start_new_session" for keyword in node.keywords)
            for node in calls))
        self.assertTrue(any(
            isinstance(node.func, ast.Attribute) and node.func.attr == "read"
            for node in calls))

    def test_spark_password_is_not_a_ros_parameter(self):
        path = os.path.abspath(os.path.join(
            PACKAGE_ROOT, "..", "ucar_2026_smart_factory_llm",
            "launch", "reason_pickup.launch"))
        root = ET.parse(path).getroot()
        self.assertIsNone(root.find("arg[@name='api_password']"))
        node = root.find("node")
        self.assertIsNone(node.find("param[@name='api_password']"))


if __name__ == "__main__":
    unittest.main()
