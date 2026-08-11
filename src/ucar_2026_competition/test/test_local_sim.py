#!/usr/bin/env python3
import ast
import os
import signal
import sys
import unittest
import xml.etree.ElementTree as ET
from unittest import mock


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_SRC = os.path.join(PACKAGE_ROOT, "src")
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from ucar_2026_competition.local_sim import (
    LocalSimConfigError,
    SIM_TERMINATION_STEPS,
    TERMINATION_STEPS,
    build_isolated_environment,
    build_launch_command,
    clean_seed_environment,
    ensure_port_available,
    parse_null_environment,
    route_source_address,
    ssh_target_host,
    validate_port,
    validate_control_master_uri,
    validate_workspace,
)


class LocalSimHelperTest(unittest.TestCase):
    def test_workspace_contract_and_missing_environment_value(self):
        with self.assertRaises(LocalSimConfigError):
            validate_workspace("")
        with mock.patch("ucar_2026_competition.local_sim.os.path.isdir", return_value=False):
            with self.assertRaises(LocalSimConfigError):
                validate_workspace("/sim/gazebo_ws")
        with mock.patch("ucar_2026_competition.local_sim.os.path.isdir", return_value=True), \
                mock.patch("ucar_2026_competition.local_sim.os.path.isfile", return_value=True):
            self.assertTrue(validate_workspace("/sim/gazebo_ws").endswith("gazebo_ws"))

    def test_port_validation(self):
        self.assertEqual(validate_port("11312", "port"), 11312)
        for invalid in (0, 65536, "invalid"):
            with self.assertRaises(LocalSimConfigError):
                validate_port(invalid, "port")

    def test_control_master_must_be_local_11311(self):
        self.assertEqual(
            validate_control_master_uri("http://localhost:11311"),
            "http://localhost:11311")
        for invalid in ("http://192.168.1.6:11311", "http://localhost:11312", ""):
            with self.assertRaises(LocalSimConfigError):
                validate_control_master_uri(invalid)

    def test_seed_environment_drops_physical_ros_overlay(self):
        source = {
            "HOME": "/home/ucar",
            "PATH": "/usr/bin",
            "DISPLAY": ":0",
            "ROS_MASTER_URI": "http://robot:11311",
            "ROS_IP": "192.168.1.20",
            "ROS_PACKAGE_PATH": "/physical/devel/share",
            "CMAKE_PREFIX_PATH": "/physical/devel",
            "PYTHONPATH": "/physical/devel/python3/dist-packages",
            "GAZEBO_MODEL_PATH": "/models",
        }
        cleaned = clean_seed_environment(source)
        self.assertEqual(cleaned["DISPLAY"], ":0")
        self.assertEqual(cleaned["GAZEBO_MODEL_PATH"], "/models")
        for key in ("ROS_MASTER_URI", "ROS_IP", "ROS_PACKAGE_PATH",
                    "CMAKE_PREFIX_PATH", "PYTHONPATH"):
            self.assertNotIn(key, cleaned)

    def test_null_environment_parser(self):
        self.assertEqual(
            parse_null_environment(b"A=1\0B=two=parts\0\0"),
            {"A": "1", "B": "two=parts"},
        )

    def test_isolated_environment_replaces_main_master_and_ros_network_identity(self):
        completed = mock.Mock(
            returncode=0,
            stdout=b"PATH=/usr/bin\0ROS_IP=10.0.0.2\0ROS_PACKAGE_PATH=/sim/share\0",
            stderr=b"",
        )
        runner = mock.Mock(return_value=completed)
        with mock.patch("ucar_2026_competition.local_sim.validate_workspace",
                        return_value="/sim/gazebo_ws"), \
                mock.patch("ucar_2026_competition.local_sim.os.path.isfile", return_value=True):
            environment = build_isolated_environment(
                "/sim/gazebo_ws", "http://127.0.0.1:11311",
                source={"PATH": "/usr/bin", "ROS_MASTER_URI": "http://robot:11311"},
                runner=runner,
            )
        self.assertEqual(environment["ROS_MASTER_URI"], "http://127.0.0.1:11311")
        self.assertEqual(environment["ROS_HOSTNAME"], "127.0.0.1")
        self.assertNotIn("ROS_IP", environment)
        seed = runner.call_args.kwargs["env"]
        self.assertNotIn("ROS_MASTER_URI", seed)
        self.assertNotIn("ROS_PACKAGE_PATH", seed)

    def test_port_conflict_is_rejected(self):
        probe = mock.MagicMock()
        probe.bind.side_effect = OSError("in use")
        with mock.patch("ucar_2026_competition.local_sim.socket.socket", return_value=probe):
            with self.assertRaises(LocalSimConfigError):
                ensure_port_available("127.0.0.1", 11312, "sim_master_port")
        probe.close.assert_called_once_with()

    def test_launch_command_binds_routed_address_and_waits_for_external_trigger(self):
        command = build_launch_command(True, "192.168.1.20", 26003)
        self.assertIn("target:=wait", command)
        self.assertIn("start_bridge:=true", command)
        self.assertIn("gui:=true", command)
        self.assertIn("bridge_host:=192.168.1.20", command)
        with self.assertRaises(LocalSimConfigError):
            build_launch_command(True, "127.0.0.1", 26003)

    def test_robot_target_and_route_source_address(self):
        self.assertEqual(ssh_target_host("ucar@192.168.1.6"), "192.168.1.6")
        probe = mock.MagicMock()
        probe.getsockname.return_value = ("192.168.1.20", 41000)
        resolver = mock.Mock(return_value=[
            (2, 2, 17, "", ("192.168.1.6", 22)),
        ])
        self.assertEqual(
            route_source_address(
                "ucar@192.168.1.6", resolver=resolver,
                socket_factory=mock.Mock(return_value=probe)),
            "192.168.1.20",
        )
        probe.connect.assert_called_once_with(("192.168.1.6", 22))
        probe.close.assert_called_once_with()

    def test_process_group_recovery_is_ordered_and_escalating(self):
        self.assertEqual(
            [item[0] for item in TERMINATION_STEPS],
            [signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGKILL", 9)],
        )
        self.assertEqual(
            [item[0] for item in SIM_TERMINATION_STEPS],
            [signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGKILL", 9)],
        )
        self.assertLess(
            sum(item[1] for item in SIM_TERMINATION_STEPS),
            sum(item[1] for item in TERMINATION_STEPS),
        )


class LocalSimLaunchContractTest(unittest.TestCase):
    def test_robot_preflight_gates_simulator_spawn(self):
        robot_path = os.path.join(PACKAGE_ROOT, "scripts", "robot_supervisor.py")
        with open(robot_path, "r", encoding="utf-8") as stream:
            robot_tree = ast.parse(stream.read(), filename=robot_path)
        robot_class = next(node for node in robot_tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "RobotSupervisor")
        robot_run = next(node for node in robot_class.body
                         if isinstance(node, ast.FunctionDef) and node.name == "run")
        robot_calls = [node for node in ast.walk(robot_run) if isinstance(node, ast.Call)]
        preflight_ok_line = next(
            node.lineno for node in robot_calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "publish_status"
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "preflight_ok")
        remote_command_line = next(
            node.lineno for node in robot_calls
            if isinstance(node.func, ast.Name) and node.func.id == "build_remote_agent_command")
        self.assertLess(preflight_ok_line, remote_command_line)

        sim_path = os.path.join(PACKAGE_ROOT, "scripts", "local_sim_supervisor.py")
        with open(sim_path, "r", encoding="utf-8") as stream:
            sim_tree = ast.parse(stream.read(), filename=sim_path)
        sim_class = next(node for node in sim_tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "LocalSimSupervisor")
        sim_run = next(node for node in sim_class.body
                       if isinstance(node, ast.FunctionDef) and node.name == "run")
        sim_calls = [node for node in ast.walk(sim_run) if isinstance(node, ast.Call)]
        wait_line = next(
            node.lineno for node in sim_calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "_wait_for_robot_preflight")
        spawn_line = next(
            node.lineno for node in sim_calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_spawn")
        self.assertLess(wait_line, spawn_line)

    def test_simulator_output_is_silenced_before_shutdown_signals(self):
        sim_path = os.path.join(PACKAGE_ROOT, "scripts", "local_sim_supervisor.py")
        with open(sim_path, "r", encoding="utf-8") as stream:
            sim_tree = ast.parse(stream.read(), filename=sim_path)
        sim_class = next(node for node in sim_tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "LocalSimSupervisor")
        shutdown = next(node for node in sim_class.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "shutdown")
        calls = [node for node in ast.walk(shutdown) if isinstance(node, ast.Call)]
        silence_line = next(
            node.lineno for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set")
        stop_line = next(
            node.lineno for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "_stop_process_group")
        self.assertLess(silence_line, stop_line)

        spawn = next(node for node in sim_class.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_spawn")
        popen = next(node for node in ast.walk(spawn)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "Popen")
        keywords = {item.arg: item.value for item in popen.keywords}
        self.assertIn("stdout", keywords)
        self.assertIn("stderr", keywords)

    def test_reusable_launch_has_required_supervisor(self):
        root = ET.parse(os.path.join(PACKAGE_ROOT, "launch", "local_sim.launch")).getroot()
        args = {node.attrib["name"]: node.attrib.get("default") for node in root.findall("arg")}
        self.assertEqual(args["sim_workspace"], "$(optenv UCAR_SIM_WS)")
        self.assertEqual(args["robot_ssh_target"], "$(optenv UCAR_ROBOT_HOST)")
        self.assertNotIn("sim_master_port", args)
        self.assertNotIn("sim_bridge_host", args)
        self.assertEqual(args["sim_bridge_port"], "26003")
        self.assertEqual(args["sim_gui"], "true")
        self.assertEqual(args["sim_startup_timeout_sec"], "120")
        node = root.find("node")
        self.assertEqual(node.attrib["type"], "local_sim_supervisor.py")
        self.assertEqual(node.attrib["required"], "true")

    def test_full_launch_is_ubuntu_host_entrypoint(self):
        root = ET.parse(os.path.join(PACKAGE_ROOT, "launch", "full_competition.launch")).getroot()
        args = {node.attrib["name"]: node.attrib.get("default") for node in root.findall("arg")}
        expected = {
            "enable_simulation": "true",
            "start_local_sim": "true",
            "use_external_sim_bridge": "$(eval enable_simulation and not start_local_sim)",
            "competition_workspace": "$(optenv UCAR_COMPETITION_WS)",
            "sim_workspace": "$(optenv UCAR_SIM_WS)",
            "robot_ssh_target": "$(optenv UCAR_ROBOT_HOST)",
            "sim_bridge_host": "",
            "sim_bridge_port": "26003",
            "sim_gui": "true",
            "sim_startup_timeout_sec": "120",
            "sim_connect_timeout_sec": "120",
        }
        for name, value in expected.items():
            self.assertEqual(args[name], value)
        self.assertNotIn("sim_master_port", args)
        group = root.find("group")
        self.assertEqual(group.attrib["if"], "$(eval enable_simulation and start_local_sim)")
        self.assertIn("local_sim.launch", group.find("include").attrib["file"])
        nodes = root.findall("node")
        self.assertEqual([node.attrib["type"] for node in nodes], ["robot_supervisor.py"])
        launch_args = nodes[0].find("rosparam[@param='physical_launch_arguments']")
        self.assertIn(
            'use_external_sim_bridge: "$(arg use_external_sim_bridge)"',
            launch_args.text,
        )
        self.assertNotIn("$(eval", launch_args.text)
        self.assertFalse(any("common_core.launch" in node.attrib.get("file", "")
                             for node in root.findall("include")))

    def test_physical_launch_contains_robot_stack_and_no_simulator(self):
        root = ET.parse(os.path.join(
            PACKAGE_ROOT, "launch", "physical_competition.launch")).getroot()
        includes = [node.attrib.get("file", "") for node in root.findall("include")]
        self.assertTrue(any("common_core.launch" in value for value in includes))
        self.assertTrue(any("flow_node.launch" in value for value in includes))
        self.assertFalse(any("local_sim.launch" in value for value in includes))

    def test_category_and_connect_timeout_reach_flow_node(self):
        root = ET.parse(os.path.join(PACKAGE_ROOT, "launch", "flow_node.launch")).getroot()
        args = {node.attrib["name"]: node.attrib.get("default") for node in root.findall("arg")}
        self.assertEqual(args["sim_target_category"], "$(arg sim_category)")
        self.assertEqual(args["sim_bridge_host"], "127.0.0.1")
        self.assertEqual(args["sim_connect_timeout_sec"], "120")
        params = {node.attrib["name"]: node.attrib.get("value")
                  for node in root.find("node").findall("param")}
        self.assertNotIn("sim_category", params)
        self.assertEqual(params["sim_target_category"], "$(arg sim_target_category)")
        self.assertEqual(params["sim_connect_timeout_sec"], "$(arg sim_connect_timeout_sec)")

    def test_task3_connection_budget_starts_after_retry(self):
        path = os.path.join(PACKAGE_ROOT, "scripts", "competition_flow.py")
        with open(path, "r", encoding="utf-8") as stream:
            tree = ast.parse(stream.read(), filename=path)
        controller = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "CompetitionFlow")
        methods = {node.name: node for node in controller.body if isinstance(node, ast.FunctionDef)}
        task3 = methods["task3"]
        connect_line = next(
            node.lineno for node in ast.walk(task3)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_connect_sim_bridge"
        )
        deadline_line = next(
            node.lineno for node in ast.walk(task3)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "deadline" for target in node.targets)
        )
        self.assertLess(connect_line, deadline_line)
        self.assertIn("_connect_sim_bridge", methods)


if __name__ == "__main__":
    unittest.main()
