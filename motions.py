# Imports
import rclpy
from pathlib import Path

from rclpy.node import Node

from utilities import Logger, euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Part 3: Message types for velocity commands and sensor readings.
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry

from rclpy.time import Time
from rclpy.utilities import remove_ros_args

# You may add any other imports you may need/want to use below
# import ...


CIRCLE=0; SPIRAL=1; ACC_LINE=2
motion_types=['circle', 'spiral', 'line']

class motion_executioner(Node):
    
    def __init__(self, motion_type=0):
        
        super().__init__("motion_types")
        
        self.type=motion_type
        
        self.radius_=0.0
        self.motion_start_time=None
        
        self.successful_init=False
        self.imu_initialized=False
        self.odom_initialized=False
        self.laser_initialized=False
        
        # Part 3: Publish velocity commands with a queue depth of 10.
        self.vel_publisher=self.create_publisher(Twist, '/cmd_vel', 10)
                
        # Keep sensor logs beside this script, regardless of the terminal's directory.
        self.data_dir=Path(__file__).resolve().parent / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.imu_logger=Logger(self.data_dir / ('imu_content_'+motion_types[motion_type]+'.csv'), headers=["acc_x", "acc_y", "angular_z", "stamp"])
        self.odom_logger=Logger(self.data_dir / ('odom_content_'+motion_types[motion_type]+'.csv'), headers=["x","y","th", "stamp"])

        # Create lidar headers on the first scan, when the number of ranges is known.
        self.laser_logger=None

        # Part 3: The Gazebo sensor publishers use RELIABLE delivery.
        # On the real robot check: ros2 topic info /odom --verbose
        # Also check /imu and /scan, use what's needed according to what the publishers require.
        qos=QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Part 5: Receive sensor messages using the QoS profile from Part 3.
        self.imu_subscription=self.create_subscription(Imu, '/imu', self.imu_callback, qos)
        self.odom_subscription=self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.laser_subscription=self.create_subscription(LaserScan, '/scan', self.laser_callback, qos)

        self.create_timer(0.1, self.timer_callback)


    # Part 5: Log sensor values and message timestamps in nanoseconds.

    def imu_callback(self, imu_msg: Imu):
        self.imu_logger.log_values([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.angular_velocity.z,
            Time.from_msg(imu_msg.header.stamp).nanoseconds,
        ])
        self.imu_initialized=True

    def odom_callback(self, odom_msg: Odometry):
        pose=odom_msg.pose.pose
        quat=pose.orientation
        yaw=euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.odom_logger.log_values([
            pose.position.x, pose.position.y, yaw,
            Time.from_msg(odom_msg.header.stamp).nanoseconds,
        ])
        self.odom_initialized=True
                
    def laser_callback(self, laser_msg: LaserScan):
        if self.laser_logger is None:
            headers=['range_'+str(i) for i in range(len(laser_msg.ranges))]
            self.laser_logger=Logger(
                self.data_dir / ('laser_content_'+motion_types[self.type]+'.csv'),
                headers=headers+['angle_increment', 'stamp'],
            )
        # One numeric column per beam; retain inf/NaN for cleaning when plotting.
        self.laser_logger.log_values(list(laser_msg.ranges)+[
            laser_msg.angle_increment,
            Time.from_msg(laser_msg.header.stamp).nanoseconds,
        ])
        self.laser_initialized=True
                
    def timer_callback(self):
        
        # Wait for all three sensors before starting motion.
        if self.odom_initialized and self.laser_initialized and self.imu_initialized:
            self.successful_init=True
        if not self.successful_init:
            return

        # Start timing when the first motion command is generated.
        if self.motion_start_time is None:
            self.motion_start_time=self.get_clock().now()
        
        cmd_vel_msg=Twist()
        
        if self.type==CIRCLE:
            cmd_vel_msg=self.make_circular_twist()
        
        elif self.type==SPIRAL:
            cmd_vel_msg=self.make_spiral_twist()
                        
        elif self.type==ACC_LINE:
            cmd_vel_msg=self.make_acc_line_twist()
            
        else:
            print("type not set successfully, 0: CIRCLE 1: SPIRAL and 2: ACCELERATED LINE")
            raise SystemExit

        self.vel_publisher.publish(cmd_vel_msg)
        
    
    # Part 4: Generate velocity commands for the three motion patterns.

    def make_circular_twist(self):
        msg=Twist()
        # Constant speeds give a 0.5 m radius: r = v / omega.
        msg.linear.x=0.3  # m/s
        msg.angular.z=0.6  # rad/s
        return msg

    def make_spiral_twist(self):
        msg=Twist()
        elapsed=(self.get_clock().now()-self.motion_start_time).nanoseconds / 1e9
        # Increase from 0.5 m at 0.01 m/s, then circle at a 1.5 m radius.
        self.radius_=min(0.5+0.01*elapsed, 1.5)
        msg.linear.x=0.3
        msg.angular.z=msg.linear.x/self.radius_
        return msg
    
    def make_acc_line_twist(self):
        msg=Twist()
        elapsed=(self.get_clock().now()-self.motion_start_time).nanoseconds / 1e9
        # Accelerate at 0.02 m/s^2 up to 0.15 m/s, with no turning.
        msg.linear.x=min(0.1*elapsed, 0.5)
        msg.angular.z=0.0
        return msg

import argparse

if __name__=="__main__":
    

    argParser=argparse.ArgumentParser(description="input the motion type")


    argParser.add_argument("--motion", type=str, default="circle")



    rclpy.init()

    # Keep ROS options out of the motion argument parser.
    args = argParser.parse_args(remove_ros_args()[1:])

    if args.motion.lower() == "circle":

        ME=motion_executioner(motion_type=CIRCLE)
    elif args.motion.lower() == "line":
        ME=motion_executioner(motion_type=ACC_LINE)

    elif args.motion.lower() =="spiral":
        ME=motion_executioner(motion_type=SPIRAL)

    else:
        print(f"we don't have {arg.motion.lower()} motion type")


    
    try:
        rclpy.spin(ME)
    except KeyboardInterrupt:
        print("Exiting")
