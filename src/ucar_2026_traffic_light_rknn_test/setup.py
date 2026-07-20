#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="ucar_2026_traffic_light_rknn_test",
    version="0.2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
