# Type of planner
POINT_PLANNER=0; TRAJECTORY_PLANNER=1

import numpy as np



class planner:
    def __init__(self, type_):

        self.type=type_

    
    def plan(self, goalPoint=[-1.0, -1.0]):
        
        if self.type==POINT_PLANNER:
            return self.point_planner(goalPoint)
        
        elif self.type==TRAJECTORY_PLANNER:
            return self.trajectory_planner()


    def point_planner(self, goalPoint):
        x = goalPoint[0]
        y = goalPoint[1]
        return x, y

    # TODO - DONE - Part 6: Implement the trajectories here
    def trajectory_planner(self):
        # Select the curve to generate for this run.
        trajectory_type = "parabola"  # "parabola" or "sigmoid"

        if trajectory_type == "parabola":
            # Sample y = x^2 over the required interval.
            x_values = np.linspace(0.0, 1.5, 31)
            y_values = x_values ** 2

        elif trajectory_type == "sigmoid":
            # Sample the required sigmoid over its specified interval.
            x_values = np.linspace(0.0, 2.5, 51)
            y_values = 2.0 / (1.0 + np.exp(-2.0 * x_values)) - 1.0

        else:
            raise ValueError("Choose 'parabola' or 'sigmoid'.")

        # Return the waypoint format expected by the trajectory controller.
        trajectory = []
        for x, y in zip(x_values, y_values):
            trajectory.append([float(x), float(y)])

        return trajectory

