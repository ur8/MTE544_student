
from mapUtilities import *
from a_star import *

POINT_PLANNER=0; TRAJECTORY_PLANNER=1

class planner:
    def __init__(self, type_, mapName="room", heuristic="euclidean"):
        self.type = type_
        self.mapName = mapName
        self.heuristic = heuristic
        self.costMap = None

    
    def plan(self, startPose, endPose):
        
        if self.type==POINT_PLANNER:
            return self.point_planner(endPose)
        
        elif self.type == TRAJECTORY_PLANNER:
            # Build the static cost map once, then reuse it for new goals.
            if self.costMap is None:
                self.initTrajectoryPlanner()

            return self.trajectory_planner(startPose, endPose)


    def point_planner(self, endPose):
        return endPose

def initTrajectoryPlanner(self):
    # TODO PART 5 Create the cost-map, the laser_sig is 
    # the standard deviation for the gausiian for which
    # the mean is located on the occupant grid. 

    # Gaussian width controls the blocked margin around obstacles.
    self.m_utilites = mapManipulator(
        filename_=self.mapName,
        laser_sig=0.23
    )

    self.costMap = self.m_utilites.make_likelihood_field()

    # Our saved map uses 205 for unknown pixels.
    # Prevent planning through unexplored space.
    unknown = self.m_utilites.getMap() == 205
    self.costMap[unknown] = 1.0
        

    def trajectory_planner(self, startPoseCart, endPoseCart):


        # This is to convert the cartesian coordinates into the 
        # the pixel coordinates of the map image, remmember,
        # the cost-map is in pixels. You can by the way, convert the pixels
        # to the cartesian coordinates and work by that index, the a_star finds
        # the path regardless. 
        startPose=self.m_utilites.position_2_cell(startPoseCart)
        endPose=self.m_utilites.position_2_cell(endPoseCart)
        
        # TODO PART 5 convert the cell pixels into the cartesian coordinates  
        # A* returns cells ordered from start to goal.
        cell_path = search(
            self.costMap,
            startPose,
            endPose,
            heuristic=self.heuristic
        )

        if not cell_path:
            return []  # No valid route was found.
        
        Path = [
            list(self.m_utilites.cell_2_position(cell))
            for cell in cell_path
        ]



        # TODO PART 5 return the path as list of [x,y]
        # Convert cells back into [x, y] waypoints in metres.

        return Path




if __name__=="__main__":

    m_utilites=mapManipulator()
    
    map_likelihood=m_utilites.make_likelihood_field()

    # you can use this part of the code to test your 
    # search algorithm regardless of the ros2 hassles
    
