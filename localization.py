"""ROS 2 odometry/IMU fusion for the six-state Lab 3 EKF."""

import json
from pathlib import Path

import message_filters
import numpy as np
from nav_msgs.msg import Odometry as odom
from rclpy import init, shutdown, spin
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu

from kalman_filter import kalman_filter
from utilities import Logger, euler_from_quaternion

rawSensors = 0
kalmanFilter = 1
odom_qos = qos_profile_sensor_data

# Keep kf_x, kf_y, stamp last for the starter plotter's x-y selection.
LOG_HEADERS = [
    "imu_ax", "imu_ay", "kf_ax", "kf_ay", "kf_vx", "kf_w",
    "odom_vx", "odom_w", "odom_x", "odom_y", "odom_th", "kf_th",
    "imu_stamp", "kf_x", "kf_y", "stamp",
]


class localization(Node):
    def __init__(self, type=kalmanFilter, dt=0.1, loggerName="robot_pose.csv", loggerHeaders=None):
        super().__init__("localizer")
        if type not in (rawSensors, kalmanFilter):
            raise ValueError("Unknown localization type")
        self.pose = None
        self._last_stamp_ns = None
        self._initialized = False
        self.dt = kalman_filter._timestep(dt)
        self.odom_topic = self.declare_parameter("odom_topic", "/odom").value
        self.imu_topic = self.declare_parameter("imu_topic", "/imu").value
        self.log_path = Path(self.declare_parameter("log_file", loggerName).value)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.loc_logger = Logger(str(self.log_path), LOG_HEADERS if loggerHeaders is None else loggerHeaders)
        if type == rawSensors:
            self.initRawSensors()
        else:
            self.initKalmanfilter(dt)

    def initRawSensors(self):
        self.odom_sub = self.create_subscription(odom, self.odom_topic, self.odom_callback, qos_profile=odom_qos)

    def initKalmanfilter(self, dt):
        # Identity-scaled covariances are the lab's starting experiment,
        # not calibrated sensor covariances. Change one scale at a time.
        q = self.declare_parameter("q_scale", 0.5).value
        r = self.declare_parameter("r_scale", 0.5).value
        p = self.declare_parameter("initial_covariance", 0.1).value
        x = np.zeros(6)
        Q = float(q) * np.eye(6)
        R = float(r) * np.eye(4)
        P = float(p) * np.eye(6)
        self.kf = kalman_filter(P, Q, R, x, dt)
        self._write_settings()

        # Best-effort, volatile QoS accepts sensor-data and reliable publishers.
        # The synchronizer pairs messages using their ROS header stamps.
        self.odom_sub = message_filters.Subscriber(self, odom, self.odom_topic, qos_profile=odom_qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, self.imu_topic, qos_profile=qos_profile_sensor_data)
        self.time_syncher = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.imu_sub], queue_size=10, slop=0.1,
        )
        self.time_syncher.registerCallback(self.fusion_callback)

    def _write_settings(self):
        """Save the actual matrices and initialization beside each CSV."""
        settings = {
            "state_order": ["x", "y", "theta", "omega", "v", "vdot"],
            "measurement_order": ["odom_v", "odom_w", "imu_ax", "imu_ay"],
            "Q": self.kf.Q.tolist(), "R": self.kf.R.tolist(),
            "initial_P": self.kf.P.tolist(), "initial_state": self.kf.x.tolist(),
            "nominal_dt_s": self.dt,
            "timing": "Prediction uses differences between accepted odometry header stamps; first pair only updates.",
            "odom_topic": self.odom_topic, "imu_topic": self.imu_topic,
            "sync_slop_s": 0.1, "sync_queue_size": 10,
            "frame_assumption": "Planar motion; IMU acceleration axes aligned with the robot body, gravity absent from horizontal components.",
            "initialization": "Pose anchored once to first synchronized odometry pose; velocity/acceleration prior is zero.",
        }
        self.log_path.with_suffix(".settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
        # Use odometry body-frame twist, not differences of world-frame pose.
        z = np.array([
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.angular.z,
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
        ])
        odom_pose = odom_msg.pose.pose
        raw_pose = np.array([
            odom_pose.position.x, odom_pose.position.y,
            euler_from_quaternion(odom_pose.orientation),
        ])
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(raw_pose)):
            self.get_logger().warning("Ignoring non-finite odometry/IMU measurements")
            return
        stamp_ns = Time.from_msg(odom_msg.header.stamp).nanoseconds
        if self._last_stamp_ns is not None and stamp_ns <= self._last_stamp_ns:
            # Do not fabricate dt for duplicate/out-of-order samples. Restart
            # this node after resetting simulation time or the odometry frame.
            self.get_logger().warning("Ignoring duplicate/out-of-order sensor pair")
            return

        if not self._initialized:
            # Establish a common frame once. Subsequent pose estimates come
            # from integration and cross-covariances, not repeated odom pose.
            self.kf.x[:3] = raw_pose
            self._write_settings()
            self._initialized = True
        else:
            elapsed = (stamp_ns - self._last_stamp_ns) / 1e9
            self.kf.predict(dt=elapsed)
        self.kf.update(z)
        self._last_stamp_ns = stamp_ns
        xhat = self.kf.get_states()
        self.pose = [xhat[0], xhat[1], xhat[2], odom_msg.header.stamp]

        # h(xhat) supplies estimated ax and ay, including ay = v*omega.
        predicted_z = self.kf.measurement_model()
        self.loc_logger.log_values([
            z[2], z[3], predicted_z[2], predicted_z[3], xhat[4], xhat[3],
            z[0], z[1], raw_pose[0], raw_pose[1], raw_pose[2], xhat[2],
            Time.from_msg(imu_msg.header.stamp).nanoseconds,
            xhat[0], xhat[1], stamp_ns,
        ])

    def odom_callback(self, pose_msg):
        self.pose = [
            pose_msg.pose.pose.position.x,
            pose_msg.pose.pose.position.y,
            euler_from_quaternion(pose_msg.pose.pose.orientation),
            pose_msg.header.stamp,
        ]

    def getPose(self):
        return self.pose


if __name__ == "__main__":
    init()
    localizer = localization()
    try:
        spin(localizer)
    except KeyboardInterrupt:
        pass
    finally:
        localizer.destroy_node()
        shutdown()
