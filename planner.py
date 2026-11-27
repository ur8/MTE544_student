
from math import floor

from a_star import BLOCKED_CELL_COST, search
from mapUtilities import mapManipulator

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
        

    def is_segment_traversable(self, start, end):
        """Check every cost-map cell touched by a world-coordinate segment."""
        if self.costMap is None:
            return False

        x0, y0 = self.m_utilites.position_in_grid(start)
        x1, y1 = self.m_utilites.position_in_grid(end)
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0

        height, width = self.costMap.shape
        dx, dy = x1 - x0, y1 - y0
        # Include both sides when the segment touches a grid edge or corner.
        tolerance = 1e-9
        first_row, last_row = floor(y0 - tolerance), floor(y1 + tolerance)
        if first_row < 0 or last_row >= height:
            return False

        for row in range(first_row, last_row + 1):
            # Clip the segment to this row and check its full span of columns.
            if dy > tolerance:
                enter = max(0.0, (row - tolerance - y0) / dy)
                leave = min(1.0, (row + 1 + tolerance - y0) / dy)
                enter_x, leave_x = x0 + dx * enter, x0 + dx * leave
            else:
                enter_x, leave_x = x0, x1

            first_col = floor(min(enter_x, leave_x) - tolerance)
            last_col = floor(max(enter_x, leave_x) + tolerance)
            if first_col < 0 or last_col >= width:
                return False
            costs = self.costMap[height - 1 - row, first_col:last_col + 1]
            if (costs > BLOCKED_CELL_COST).any():
                return False
        return True

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
    
