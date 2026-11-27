import matplotlib.pyplot as plt
import numpy as np
import yaml

from sensor_msgs.msg import LaserScan
from math import floor
import math

from math import pi as M_PI
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PointStamped, Quaternion, Point
from utilities import *

class mapManipulator(Node):


    def __init__(self, filename_: str = "room.yaml", laser_sig=0.01):
        
        
        super().__init__('likelihood_field')
        
        filenameYaml=None
        filenamePGM=None
        if ".pgm" in filename_:
            
            filenamePGM=filename_
            filenameYaml=filename_.replace(".pgm", ".yaml")
            
        elif ".yaml" in filename_:
            
            filenameYaml=filename_
            filenamePGM=filename_.replace(".yaml", ".pgm")
        
        else:
            filenameYaml=filename_ + ".yaml"
            filenamePGM=filename_+".pgm"

        

        width, height, max_value, pixels = self.read_pgm(filenamePGM)


        self.width = width
        self.height = height
        
        
        

        self.image_array = np.array(pixels).reshape((height, width))
        self.o_x, self.o_y, self.yaw, self.res, self.thresh = self.read_description(filenameYaml)

        self.laser_sig=laser_sig
        
        self.likelihood_msg=None

        
    def getAllObstacles(self):
        image_array=self.image_array.T

        
        
        
        indices = np.where(image_array < 10)
        
        return [self.cell_2_position([i, j]) for i, j in zip(indices[0], indices[1])]

    def getLikelihoodField(self):
        return self.likelihood_field
    
    def getMetaData(self):
        return self.o_x, self.o_y, self.res, self.thresh
            
    def getMap(self):
        return self.image_array
    
    def timer_callback(self):
        if self.likelihood_msg is None: 
            return
        self.map_publisher.publish(self.likelihood_msg)
        
    def read_pgm(self, filename):
        with open(filename, 'rb') as f:
            # Check if it's a PGM file
            header = f.readline().decode().strip()
            
            if header != 'P5':
                raise ValueError('Invalid PGM file format')

            # Skip comments
            line = f.readline().decode().strip()
            while line.startswith('#'):
                line = f.readline().decode().strip()

            # Read width, height, and maximum gray value
            width, height = map(int, line.split())
            max_value = int(f.readline().decode().strip())

            # Read the image data
            image_data = f.read()

        # Convert image data to a list of pixel values
        pixels = [x for x in image_data]

        return width, height, max_value, pixels

    def plot_pgm_image(self, image_array):
        # Convert pixel values to a NumPy array


        # Plot the image
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.title('PGM Image')
        plt.show()



    def read_description(self, filenameYAML):
        with open(filenameYAML, 'r') as file:
            description = yaml.safe_load(file)
        origin_x, origin_y, yaw = map(float, description['origin'])
        resolution = float(description['resolution'])
        threshold = float(description['occupied_thresh'])
        return origin_x, origin_y, yaw, resolution, threshold

    def getOrigin(self):
        return np.array([self.o_x, self.o_y])
    
    def getResolution(self):
        return self.res
    
    def cell_2_position(self, pix):
        col, row = pix
        # PGM rows count down from the top. Return the center of the cell.
        x = (col + 0.5) * self.res
        y = (self.height - row - 0.5) * self.res
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.o_x + c * x - s * y, self.o_y + s * x + c * y

    def position_in_grid(self, pos):
        """Return continuous grid coordinates measured from the lower left."""
        dx, dy = pos[0] - self.o_x, pos[1] - self.o_y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * dx + s * dy) / self.res, (-s * dx + c * dy) / self.res

    def position_2_cell(self, pos):
        col, row_from_bottom = self.position_in_grid(pos)
        return floor(col), self.height - 1 - floor(row_from_bottom)

    def make_likelihood_field(self):
        
        image_array=self.image_array

        from sklearn.neighbors import KDTree
        
        

        rows, cols = np.where(image_array < 10)
        # A step between adjacent grid cells is self.res metres.
        occupied_cells = np.column_stack((rows, cols))
        all_cells = np.indices(image_array.shape).reshape(2, -1).T
        if len(occupied_cells):
            dists = KDTree(occupied_cells).query(all_cells, k=1)[0] * self.res
        else:
            dists = np.full((image_array.size, 1), np.inf)
        probabilities=np.exp( -(dists**2) / (2*self.laser_sig**2))
        
        likelihood_field=probabilities.reshape(image_array.shape)
        
        likelihood_field_img=np.array(255-255*probabilities.reshape(image_array.shape), dtype=np.int32)
        
        self.likelihood_img=likelihood_field_img
        
        self.occ_points = np.asarray(self.getAllObstacles()).reshape(-1, 2)
        
                
        #self.plot_pgm_image(likelihood_field_img)

        self.likelihood_field = likelihood_field
        
        return likelihood_field
                
    
    def _numpy_to_data(self, data):
        """
        Convert the numpy array containing grid data to a python
        list suitable for use as the data field in an OccupancyGrid
        message.
        """
        flat_grid = data.reshape((data.size,)) * 100
        data_ = set(np.array(np.round(flat_grid), dtype='int'))

        return data_
    
    def to_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map. """
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"

        likelihoodField = self.getLikelihoodField()

        grid.info.resolution = self.getResolution()  # Set the resolution (m/cell)
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin = Pose()
        grid.info.origin.position.x, grid.info.origin.position.y = self.o_x, self.o_y
        grid.info.origin.orientation.w = math.cos(self.yaw / 2)
        grid.info.origin.orientation.z = math.sin(self.yaw / 2)

        values = np.ceil(likelihoodField * 100).astype(int)
        values[self.image_array == 205] = -1  # ROS OccupancyGrid unknown value.
        grid.data = np.flipud(values).ravel().tolist()  # ROS rows start at the bottom.
        return grid

    def calculate_score(self, x, y):
        col, row = self.position_2_cell((x, y))
        if 0 <= col < self.width and 0 <= row < self.height:
            return self.likelihood_field[row, col]
        return 0

    def map_localation_query(self, laser_msg: LaserScan):
        
        # this part was for kidnapping section of the particle filter algorithm
        # the complete algorithm is in this repo
        # https://github.com/aalghooneh/ros2-particlefilter-python

        pass
            





     



import argparse
if __name__=="__main__":
    


    rclpy.init()

    parser=argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default="./your_map/room.yaml", help='the absolute path to argument')
    parser.add_argument('--std', type=float, help='the std', default=0.01)


    args = parser.parse_args()

    MAP_UTILITIS=mapManipulator(args.map, args.std)

    #rclpy.spin(MAP_UTILITIS)


# Usage example

