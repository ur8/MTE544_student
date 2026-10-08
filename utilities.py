from math import atan2, asin, sqrt

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

            # Part 5: Write numeric values in the same format as the header.
            for value in values_list:
                vals_str+=str(value)+", "
            
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
            # Skip the header line

            if not read_headers:
                for line in file:
                    values=line.strip().split(',')

                    for val in values:
                        if val=='':
                            break
                        headers.append(val.strip())

                    read_headers=True
                    break
            
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


# Part 5: Extract the robot's heading from its orientation quaternion.
def euler_from_quaternion(quat):
    """
    Return Euler yaw in radians from a unit quaternion.
    quat = [x, y, z, w]
    """
    # Since robot moving on flat floor, heading is 2D angle, simplify R21 and R11 expression.
    # yaw = atan2(R21, R11)
    x, y, z, w=quat
    yaw=atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))
    return yaw

