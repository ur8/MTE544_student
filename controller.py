
from pid import PID_ctrl
from math import pi

import numpy as np

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error

M_PI=3.1415926535

P=0; PD=1; PI=2; PID=3
TURN_IN_PLACE_ANGLE = pi / 4

class controller:
    
    
    
    def __init__(self, klp=0.2, klv=0.2, kli=0.2, kap=0.2, kav=0.2, kai=0.2):
        
        self.PID_linear=PID_ctrl(PID, klp, klv, kli, filename_="linear.csv")
        self.PID_angular=PID_ctrl(PID, kap, kav, kai, filename_="angular.csv")

    
    def vel_request(self, pose, goal, status):
        
        e_lin=calculate_linear_error(pose, goal)
        e_ang=calculate_angular_error(pose, goal)

        
        linear_vel=self.PID_linear.update([e_lin, pose[3]], status)
        angular_vel=self.PID_angular.update([e_ang, pose[3]], status) 

        # Slow down through turns and face sharp turns before moving forward.
        linear_vel *= max(0.0, 1.0 - abs(e_ang) / TURN_IN_PLACE_ANGLE)

        linear_vel = 0.5 if linear_vel > 1.0 else linear_vel
        angular_vel= 0.5 if angular_vel > 1.0 else angular_vel


        return linear_vel, angular_vel
    

class trajectoryController(controller):

    def __init__(self, klp=0.2, klv=0.2, kli=0.2, kap=0.2, kav=0.2, kai=0.2,
                 lookAhead=1.0, targetVel=1.0, *, is_segment_traversable=None):
        super().__init__(klp, klv, kli, kap, kav, kai)
        self.lookAhead=lookAhead
        self.targetVelocity=targetVel
        self.is_segment_traversable = is_segment_traversable
    
    def vel_request(self, pose, listGoals, status):
        """Return the requested velocities, or None if every target is blocked."""
        
        goal = self.lookFarFor(pose, listGoals)
        if goal is None:
            return None
        
        finalGoal=listGoals[-1]
        
  
        e_lin=calculate_linear_error(pose, finalGoal)
        e_ang=calculate_angular_error(pose, goal)

        
        linear_vel=self.PID_linear.update([e_lin, pose[3]], status)
        angular_vel=self.PID_angular.update([e_ang, pose[3]], status) 

        linear_vel *= max(0.0, 1.0 - abs(e_ang) / TURN_IN_PLACE_ANGLE)

        linear_vel = 0.1 if linear_vel > 0.1 else linear_vel
        angular_vel= 0.5 if angular_vel > 0.5 else angular_vel


        return linear_vel, angular_vel


    def lookFarFor(self, pose, listGoals):
        if len(listGoals) == 0:
            return None

        position = pose[:2]
        offsets = np.asarray(listGoals) - position
        closest = np.argmin(np.sum(offsets**2, axis=1))
        last = len(listGoals) - 1
        farthest = min(closest + 3, last)
        if self.is_segment_traversable is None:
            return listGoals[farthest]

        nearest = min(closest + 1, last)
        for index in range(farthest, nearest - 1, -1):
            target = listGoals[index]
            if self.is_segment_traversable(position, target):
                return target
        return None
