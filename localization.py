import sys

from utilities import Logger, euler_from_quaternion
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry as odom

from rclpy import init, spin, try_shutdown 

rawSensor = 0
class localization(Node):
    
    def __init__(self, localizationType=rawSensor):

        super().__init__("localizer")
        
        # TODO - DONE - Part 3: Define the QoS profile variable based on whether you are using the simulation (Turtlebot 3 Burger) or the real robot (Turtlebot 4)
        # Remember to define your QoS profile based on the information available in "ros2 topic info /odom --verbose" as explained in Tutorial 3
        # Accept live odometry samples and retain a queue of up to 10 messages.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # if publisher reports reliable, change from BEST_EFFORT to RELIABLE
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        self.loc_logger=Logger("robot_pose.csv", ["x", "y", "theta", "stamp"])
        self.pose=None
        
        if localizationType == rawSensor:
        # TODO - DONE Part 3: subscribe to the position sensor topic (Odometry)
            self.odom_subscription = self.create_subscription(
                odom, 
                "/odom",
                self.odom_callback, 
                odom_qos,
            )
        else:
            print("This type doesn't exist", sys.stderr)
    
    
    def odom_callback(self, pose_msg):
        # TODO - DONE - Part 3: Read x,y, theta, and record the stamp
        position = pose_msg.pose.pose.position  # pose.pose is intentional - outer pose is the estimated pose with uncertainty, inner pose is position and orientation
        orientation = pose_msg.pose.pose.orientation 

        # Convert quaternion orientation to a planar heading in radians.
        theta = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ])

        # Preserve the measurement timestamp for later controller calculations.
        self.pose = [
            position.x,
            position.y,
            theta,
            pose_msg.header.stamp,
        ]
        
        # Log the data - position in metres, heading in radians, and timestamp in nanoseconds.
        self.loc_logger.log_values([self.pose[0], self.pose[1], self.pose[2], Time.from_msg(self.pose[3]).nanoseconds])
    
    def getPose(self):
        return self.pose

# TODO - DONE - Part 3
# Here put a guard that makes the node run, ONLY when run as a main thread!
# This is to make sure this node functions right before using it in decision.py
if __name__ == "__main__":
    # Run this test node only when file is executed directly 
    init()
    localizer = localization(rawSensor)

    try:
        spin(localizer)
    except KeyboardInterrupt:
        pass
    finally:
        localizer.destroy_node()
        try_shutdown()

    
