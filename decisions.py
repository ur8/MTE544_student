import sys

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from pid import PID_ctrl

from rclpy import init, shutdown
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.utilities import remove_ros_args
from rclpy.node import Node
from geometry_msgs.msg import Twist


from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from localization import localization, rawSensors, kalmanFilter

from planner import TRAJECTORY_PLANNER, POINT_PLANNER, SPIRAL_4TUNE, planner
from controller import controller, trajectoryController


class decision_maker(Node):
    
    
    def __init__(self, publisher_msg, publishing_topic, qos_publisher, rate=10, motion_type=POINT_PLANNER):

        super().__init__("decision_maker")

        self.publisher=self.create_publisher(publisher_msg, publishing_topic, qos_profile=qos_publisher)
        
        
        publishing_period=1/rate

        self.reachThreshold=0.01


        # Feed estimated x/y/heading to the existing controller. Sensor timing
        # is measured in fusion_callback; this is only the nominal interval.
        self.localizer=localization(type=kalmanFilter, dt=publishing_period)
        
        if motion_type==POINT_PLANNER:
            self.controller=controller(klp=0.2, klv=0.5, kap=0.8, kav=0.6)      
            self.planner=planner(POINT_PLANNER)

        
        elif motion_type==TRAJECTORY_PLANNER:
            self.controller=trajectoryController(klp=0.2, klv=0.5, kap=0.8, kav=0.6)      
            self.planner=planner(TRAJECTORY_PLANNER)
        elif motion_type==SPIRAL_4TUNE:
            self.controller=None
            self.planner=None
            self.linearVelocity = 0
        
        else:
            print("Error! you don't have this type of planner", file=sys.stderr)
            return

        self.motion_type=motion_type
        self.goal=self.planner.plan() if self.motion_type != SPIRAL_4TUNE else None
        

        self.create_timer(publishing_period, self.timerCallback)



    def timerCallback(self):
        
        if self.localizer.getPose() is  None:
            print("waiting for synchronized odom/IMU msgs ....")
            return
        
        
        vel_msg=Twist()

        if self.motion_type == SPIRAL_4TUNE:
            self.linearVelocity += 0.01 if self.linearVelocity < 1.0 else 0.0
            vel_msg.linear.x=self.linearVelocity
            vel_msg.angular.z=1.0
            self.publisher.publish(vel_msg)
            return

        
        if type(self.goal) == list:
            reached_goal=True if calculate_linear_error(self.localizer.getPose(), self.goal[-1]) <self.reachThreshold else False
        else: 
            reached_goal=True if calculate_linear_error(self.localizer.getPose(), self.goal) <self.reachThreshold else False


        if reached_goal:
            print("reached goal")
            self.publisher.publish(vel_msg)
            
            self.controller.PID_angular.logger.save_log()
            self.controller.PID_linear.logger.save_log()
            
            raise SystemExit
        
        velocity, yaw_rate = self.controller.\
            vel_request(self.localizer.getPose(), self.goal, True)

        
        vel_msg.linear.x=velocity
        vel_msg.angular.z=yaw_rate
        
        self.publisher.publish(vel_msg)


import argparse
def main(args=None):
    # Keep ROS parameter/remapping flags for rclpy, but not argparse.
    argv = sys.argv if args is None else ["decisions.py", *args]
    parser = argparse.ArgumentParser(description="Lab 3 EKF motion tests")
    parser.add_argument("--motion", choices=["point", "trajectory", "spiral"], default="spiral")
    options = parser.parse_args(remove_ros_args(args=argv)[1:])
    init(args=argv)
    motion_types = {"point": POINT_PLANNER, "trajectory": TRAJECTORY_PLANNER, "spiral": SPIRAL_4TUNE}
    DM = decision_maker(Twist, "/cmd_vel", 10, motion_type=motion_types[options.motion])
    # Spin both nodes together. Nested spin_once in a timer can block control
    # while waiting for a sensor; the executor services sensor callbacks too.
    executor = SingleThreadedExecutor()
    executor.add_node(DM)
    executor.add_node(DM.localizer)
    try:
        executor.spin()
    except SystemExit:
        print(f"reached there successfully {DM.localizer.pose}")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Stop open-loop spiral motion as well as point control on exit.
        if DM.context.ok():
            DM.publisher.publish(Twist())
        executor.shutdown()
        DM.localizer.destroy_node()
        DM.destroy_node()
        if DM.context.ok():
            shutdown()
if __name__=="__main__":
    main()
