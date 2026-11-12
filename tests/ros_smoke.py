"""Manual ROS integration check with synthetic sensors, no velocity publisher.

Run from the repository root: python3 tests/ros_smoke.py
Uses localhost-only DDS in domain 72, isolated topics and temporary log files.
"""

import csv
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['ROS_DOMAIN_ID'] = '72'
os.environ['ROS_LOG_DIR'] = '/tmp/mte544_lab3_ros_logs'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from localization import localization


def main():
    with tempfile.TemporaryDirectory(prefix='lab3_ros_smoke_') as directory:
        logfile = Path(directory) / 'robot_pose.csv'
        rclpy.init(args=[
            'lab3_smoke', '--ros-args', '-p', 'q_scale:=0.1', '-p', 'r_scale:=0.5',
            '-p', f'log_file:={logfile}', '-p', 'odom_topic:=/lab3_test/odom',
            '-p', 'imu_topic:=/lab3_test/imu',
        ])
        localizer = localization()
        publisher = Node('lab3_synthetic_sensors')
        odom_pub = publisher.create_publisher(Odometry, '/lab3_test/odom', qos_profile_sensor_data)
        imu_pub = publisher.create_publisher(Imu, '/lab3_test/imu', qos_profile_sensor_data)
        executor = SingleThreadedExecutor()
        executor.add_node(localizer)
        executor.add_node(publisher)
        try:
            deadline = time.monotonic() + 5
            while (odom_pub.get_subscription_count() == 0 or imu_pub.get_subscription_count() == 0) and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
            assert odom_pub.get_subscription_count() > 0, 'Odometry subscriber not discovered'
            assert imu_pub.get_subscription_count() > 0, 'IMU subscriber not discovered'
            for i in range(5):
                odom, imu = Odometry(), Imu()
                odom.header.stamp.sec = imu.header.stamp.sec = 10
                odom.header.stamp.nanosec = imu.header.stamp.nanosec = i * 100_000_000
                odom.pose.pose.orientation.w = 1.0
                odom.twist.twist.linear.x = 0.2
                odom.twist.twist.angular.z = 0.3
                imu.linear_acceleration.y = 0.06
                odom_pub.publish(odom)
                imu_pub.publish(imu)
                target = 10_000_000_000 + i * 100_000_000
                deadline = time.monotonic() + 2
                while localizer._last_stamp_ns != target and time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=0.02)
                assert localizer._last_stamp_ns == target, f'Pair {i} was not fused'
            with logfile.open() as handle:
                rows = list(csv.DictReader(handle, skipinitialspace=True))
            assert len(rows) == 5
            assert localizer.kf.Q[0, 0] == 0.1
            assert localizer.kf.R[0, 0] == 0.5
            assert logfile.with_suffix('.settings.json').exists()
            print('ROS smoke passed: real subscribers, QoS, synchronized delivery, parameter overrides, five logged EKF updates.')
            print('Final pose:', localizer.pose[:3])
        finally:
            executor.shutdown()
            localizer.destroy_node()
            publisher.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
