"""Exercise the real fusion callback without starting DDS or robot motion."""

import csv
import json
from unittest.mock import Mock

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from kalman_filter import kalman_filter
from localization import LOG_HEADERS, localization
from utilities import Logger


def make_callback_node(tmp_path):
    # Node construction opens DDS. This fixture isolates the callback while
    # still using the installed ROS message classes and timestamp conversion.
    node = localization.__new__(localization)
    node.pose = None
    node._last_stamp_ns = None
    node._initialized = False
    node.dt = 0.1
    node.odom_topic, node.imu_topic = '/odom', '/imu'
    node.log_path = tmp_path / 'robot_pose.csv'
    node.loc_logger = Logger(str(node.log_path), LOG_HEADERS)
    node.kf = kalman_filter(np.eye(6) * 0.1, np.eye(6) * 0.5, np.eye(4) * 0.5, np.zeros(6), 0.1)
    node.get_logger = Mock(return_value=Mock())
    return node


def messages(stamp_ns, x=2.0, y=-3.0, yaw=0.6):
    odom, imu = Odometry(), Imu()
    odom.header.stamp.sec, odom.header.stamp.nanosec = divmod(stamp_ns, 10**9)
    imu.header.stamp.sec, imu.header.stamp.nanosec = divmod(stamp_ns + 20_000_000, 10**9)
    odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
    odom.pose.pose.orientation.z = float(np.sin(yaw / 2))
    odom.pose.pose.orientation.w = float(np.cos(yaw / 2))
    odom.twist.twist.linear.x = 0.2
    odom.twist.twist.angular.z = 0.4
    imu.linear_acceleration.x = 0.1
    imu.linear_acceleration.y = 0.08
    return odom, imu


def rows(node):
    with node.log_path.open() as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def test_initialization_real_dt_logging_and_no_repeated_pose_fusion(tmp_path):
    node = make_callback_node(tmp_path)
    # Large epoch timestamp with nanoseconds that would lose precision if
    # converted to float before subtraction/logging.
    origin = 1_700_000_000_000_000_007
    first_odom, first_imu = messages(origin)
    node.fusion_callback(first_odom, first_imu)
    np.testing.assert_allclose(node.kf.x[:3], [2, -3, 0.6])
    assert node.pose[3] is first_odom.header.stamp
    settings = json.loads(node.log_path.with_suffix('.settings.json').read_text())
    np.testing.assert_allclose(settings['initial_state'][:3], [2, -3, 0.6])
    np.testing.assert_allclose(settings['Q'], np.eye(6) * 0.5)
    # A wildly different raw pose must only be logged after initialization,
    # not used as another absolute measurement in the filter correction.
    node.fusion_callback(*messages(origin + 230_000_000, x=999.0, y=999.0))
    assert node.kf.dt == 0.23
    assert np.linalg.norm(node.kf.x[:2] - [2, -3]) < 0.2
    logged = rows(node)
    assert len(logged) == 2
    last = logged[-1]
    assert int(last['stamp']) == origin + 230_000_000
    assert int(last['imu_stamp']) == origin + 250_000_000
    assert float(last['odom_x']) == 999.0
    assert float(last['odom_vx']) == 0.2
    assert float(last['kf_ay']) == float(last['kf_vx']) * float(last['kf_w'])
    assert float(last['kf_th']) == node.pose[2]


def test_zero_first_stamp_duplicate_backwards_and_nonfinite_pairs(tmp_path):
    node = make_callback_node(tmp_path)
    node.fusion_callback(*messages(0))
    node.fusion_callback(*messages(100_000_000))
    state, covariance = node.kf.x.copy(), node.kf.P.copy()
    for stamp in [100_000_000, 50_000_000]:
        node.fusion_callback(*messages(stamp))
    odom, imu = messages(200_000_000)
    imu.linear_acceleration.x = float('nan')
    node.fusion_callback(odom, imu)
    np.testing.assert_array_equal(node.kf.x, state)
    np.testing.assert_array_equal(node.kf.P, covariance)
    assert node._last_stamp_ns == 100_000_000
    assert len(rows(node)) == 2
    assert node.get_logger.return_value.warning.call_count == 3
