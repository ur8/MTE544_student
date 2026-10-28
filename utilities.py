from math import atan2, asin, sqrt, pi

M_PI=3.1415926535

class Logger:
    
    def __init__(self, filename, headers=["e", "e_dot", "e_int", "stamp"]):
        
        self.filename = filename

        with open(self.filename, 'w') as file:
            
            header_str=""

            for header in headers:
                header_str+=header
                header_str+=", "
            
            header_str+="\n"
            
            file.write(header_str)


    def log_values(self, values_list):

        with open(self.filename, 'a') as file:
            
            vals_str=""
            
            for value in values_list:
                vals_str+=f"{value}, "
            
            vals_str+="\n"
            
            file.write(vals_str)
            

    def save_log(self):
        pass

class FileReader:
    def __init__(self, filename):
        
        self.filename = filename
        
        
    def read_file(self):
        
        read_headers=False

        table=[]
        headers=[]
        with open(self.filename, 'r') as file:

            if not read_headers:
                for line in file:
                    values=line.strip().split(',')

                    for val in values:
                        if val=='':
                            break
                        headers.append(val.strip())

                    read_headers=True
                    break
            
            next(file)
            
            # Read each line and extract values
            for line in file:
                values = line.strip().split(',')
                
                row=[]                
                
                for val in values:
                    if val=='':
                        break
                    row.append(float(val.strip()))

                table.append(row)
        
        return headers, table
    
    

# TODO- DONE - Part 3: Implement the conversion from Quaternion to Euler Angles
def euler_from_quaternion(quat):
    """
    Convert quaternion (w in last place) to euler roll, pitch, yaw.
    quat = [x, y, z, w]
    """
    x, y, z, w = quat

    # extract rotation about the vertical axis for planar navigation 
    sin_yaw = 2.0 * (w*z + x*y)
    cos_yaw = 1.0 - (2.0 * (y*y + z*z))
    yaw = atan2(sin_yaw, cos_yaw)

    # just unpack yaw
    return yaw


#TODO - DONE - Part 4: Implement the calculation of the linear error
def calculate_linear_error(current_pose, goal_pose):
    # returns the straight line distance to the goal in metres
        
    # Compute the linear error in x and y
    # Remember that current_pose = [x,y, theta, time stamp] and goal_pose = [x,y]
    # Remember to use the Euclidean distance to calculate the error.

    dx = goal_pose[0] - current_pose[0] 
    dy = goal_pose[1] - current_pose[1] 

    # Euclidean distance gives remaining distance to the goal 
    error_linear = sqrt(dx*dx + dy*dy)

    return error_linear

#TODO - DONE - Part 4: Implement the calculation of the angular error
def calculate_angular_error(current_pose, goal_pose):
    # return the signed heading error to the goal in radians 

    # Compute the linear error in x and y
    # Remember that current_pose = [x,y, theta, time stamp] and goal_pose = [x,y]
    # Use atan2 to find the desired orientation
    # Remember that this function returns the difference in orientation between where the robot currently faces and where it should face to reach the goal
    dx = goal_pose[0] - current_pose[0] 
    dy = goal_pose[1] - current_pose[1]

    # find direction from current position towards goal
    desired_heading = atan2(dy/dx)

    # positive error = CCW turn 
    error_angular = desired_heading - current_pose[2]

    # Remember to handle the cases where the angular error might exceed the range [-π, π]
    # wrap to [-pi, pi), so that the robot can take the shorter turning direction 
    error_angular = ((error_angular + pi) % (2.0 * pi)) - pi 
    
    return error_angular
