cmake_minimum_required(VERSION 3.0.2)
project(yolo)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  camera_mjpeg_server.py
  coco2yolo.py
  collect_factory_sign.py
  keyboard_collect_yolo_images.py
  qr_collect_and_decode.py
  validate_model.py
  scripts/init_traffic_dataset.py
  scripts/build_traffic_yolo_dataset.py
  scripts/prepare_traffic_cls_dataset.py
  scripts/train_traffic_resnet18.py
  scripts/prepare_traffic_rknn_calibration.py
  scripts/convert_traffic_resnet18_to_rknn.py
  scripts/compare_traffic_onnx_rknn.py
  scripts/export_traffic_onnx_reference.py
  scripts/compare_traffic_rknnlite_reference.py
  scripts/run_traffic_rknn_gate_on_car.py
  scripts/deploy_traffic_resnet18_to_car.py
  scripts/run_traffic_ros_smoke_on_car.py
  scripts/check_factory_sign_rknn_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES traffic_yolov5_dataset.md traffic_classifier_training.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_traffic_dataset_tools.py)
endif()
