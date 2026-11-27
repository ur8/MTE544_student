

import argparse
import sys
from math import isfinite

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy import init, try_shutdown
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from controller import controller, trajectoryController
from localization import localization, rawSensors
from planner import POINT_PLANNER, TRAJECTORY_PLANNER, planner
from utilities import calculate_linear_error


ODOM_TIMEOUT = 0.5


class decision_maker(Node):
    
    
    def __init__(self, publisher_msg, publishing_topic, qos_publisher, rate=10, motion_type=POINT_PLANNER, heuristic="euclidean"):

        super().__init__("decision_maker")

        self.publisher=self.create_publisher(publisher_msg, publishing_topic, qos_profile=qos_publisher)


        self.create_subscription(PoseStamped, "/goal_pose", self.designPathFor, 10)
        
        self.pathPublisher = self.create_publisher(Path, '/designedPath', 10)
        
        publishing_period=1/rate
        self.motion_type = motion_type

        # TODO PART 5 choose your threshold
        # Stop when within 10 cm of the final waypoint.
        self.reachThreshold = 0.10

        # TODO PART 5 your localization type
        # Use odometry directly; allowed by the lab instructions.
        self.localizer = localization(rawSensors)


        
        if motion_type==POINT_PLANNER:
            self.controller=controller(klp=0.2, klv=0.5, kap=0.8, kav=0.6)      
            self.planner=planner(POINT_PLANNER)

        
        elif motion_type==TRAJECTORY_PLANNER:
            # TODO PART 5 Put the gains that you concluded from lab 2
            # Initial proportional gains for simulation; tune during testing.
            self.controller = trajectoryController(
                klp=0.2, klv=0.0, kli=0.0,
                kap=0.8, kav=0.0, kai=0.0
            )     
            self.planner = planner(
                TRAJECTORY_PLANNER,
                mapName="room",
                heuristic=heuristic
            )
        
        else:
            print("Error! you don't have this type of planner", file=sys.stderr)


        self.goal = None

        # Keep the odometry timeout active even when Gazebo pauses its clock.
        self.create_timer(publishing_period, self.timerCallback,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))


        print("waiting for your input position, use 2D nav goal in rviz2")




    # This is for the rviz2 interface
    def designPathFor(self, msg: PoseStamped):
        # Stop the previous motion before planning a new route.
        self.publisher.publish(Twist())
        self.goal = None
        self.publishPathOnRviz2([])

        pose = self.localizer.get_fresh_odom_pose(ODOM_TIMEOUT)
        if pose is None:
            self.get_logger().warning("Waiting for fresh odometry. Send a new goal when odometry returns.")
            return

        goal_xy = [msg.pose.position.x, msg.pose.position.y]
        if msg.header.frame_id not in ("", "map") or not all(map(isfinite, goal_xy)):
            self.get_logger().warning("Expected a finite goal in the map frame.")
            return

        path = self.planner.plan(pose[:2], goal_xy)
        if not path:
            print("No valid path found. Check that the start and goal are connected in mapped free space.")
            return

        # Start each route with fresh controller error history.
        self.controller.PID_linear.history.clear()
        self.controller.PID_angular.history.clear()
        self.goal_odometry_epoch = self.localizer.odom_epoch

        # Store waypoint lists for RViz and the arrival check in both modes.
        self.goal = [path] if self.motion_type == POINT_PLANNER else path
        self.publishPathOnRviz2(self.goal)

    def timerCallback(self):
        if self.goal is None:
            return

        pose = self.localizer.get_fresh_odom_pose(ODOM_TIMEOUT)
        if pose is None or self.localizer.odom_epoch != self.goal_odometry_epoch:
            self.publisher.publish(Twist())
            self.goal = None
            self.publishPathOnRviz2([])
            self.get_logger().warning("Odometry lost or reset. Stopped. Send a new goal to resume.")
            return

        vel_msg = Twist()
        if calculate_linear_error(pose, self.goal[-1]) < self.reachThreshold:
            self.publisher.publish(vel_msg)
            self.goal = None
            self.controller.PID_angular.logger.save_log()
            self.controller.PID_linear.logger.save_log()
            print("reached goal")
            print("waiting for the new position input, use 2D nav goal on map")
            return

        target = self.goal[-1] if self.motion_type == POINT_PLANNER else self.goal
        velocity, yaw_rate = self.controller.vel_request(pose, target, True)
        # Limit forward speed and turning speed in both directions.
        vel_msg.linear.x = float(max(0.0, min(0.1, velocity)))
        vel_msg.angular.z = float(max(-0.5, min(0.5, yaw_rate)))
        self.publisher.publish(vel_msg)
        self.publishPathOnRviz2(self.goal)

    def publishPathOnRviz2(self, path):

        Path_ =  Path()

        Path_.header.frame_id ="map"
        Path_.header.stamp = self.get_clock().now().to_msg()

        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]

            # Set the orientation of the pose. Here, it's set to a default orientation.
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            Path_.poses.append(pose)

        self.pathPublisher.publish(Path_)


def main(args):
    # Keep ROS active through Ctrl+C so the final zero velocity can be sent.
    init(signal_handler_options=SignalHandlerOptions.NO)
    motion_type = POINT_PLANNER if args.motion == "point" else TRAJECTORY_PLANNER
    dm = decision_maker(Twist, "/cmd_vel", 10, motion_type=motion_type, heuristic=args.heuristic)

    executor = SingleThreadedExecutor()
    executor.add_node(dm.localizer)
    executor.add_node(dm)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        dm.publisher.publish(Twist())
        executor.shutdown()
        dm.localizer.destroy_node()
        dm.destroy_node()
        try_shutdown()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="point or trajectory")
    argParser.add_argument("--motion", choices=["point", "trajectory"], default="trajectory")
    argParser.add_argument(
        "--heuristic",
        choices=["manhattan", "euclidean"],
        default="euclidean"
    )
    main(argParser.parse_args(remove_ros_args(sys.argv)[1:]))
