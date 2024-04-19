import sys
import os
import time

## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer
import test_astar as Astar

def main(data_folder_name):
    # Set the path to config files and data files for the environment
    #Astar.example()
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    
    # Instantiate the environment
    env = Env(data_folder)
    
    # config files for the agents
    rescuer1_file = os.path.join(data_folder, "rescuer_config_1.txt")
    rescuer2_file = os.path.join(data_folder, "rescuer_config_2.txt")
    rescuer3_file = os.path.join(data_folder, "rescuer_config_3.txt")
    rescuer4_file = os.path.join(data_folder, "rescuer_config_4.txt")

    explorer1_file = os.path.join(data_folder, "explorer_config_1.txt")
    explorer2_file = os.path.join(data_folder, "explorer_config_2.txt")
    explorer3_file = os.path.join(data_folder, "explorer_config_3.txt")
    explorer4_file = os.path.join(data_folder, "explorer_config_4.txt")
    
    # Instantiate agents rescuer and explorer
    resc1 = Rescuer(env, rescuer1_file,data_folder_name)
    # resc2 = Rescuer(env, rescuer2_file)
    # resc3 = Rescuer(env, rescuer3_file)
    # resc4 = Rescuer(env, rescuer4_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp1 = Explorer(env, explorer1_file, resc1)
    exp2 = Explorer(env, explorer2_file, resc1)
    exp3 = Explorer(env, explorer3_file, resc1)
    exp4 = Explorer(env, explorer4_file, resc1)
    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        #data_folder_name = os.path.join("datasets", "data_10v_12x12")
        #data_folder_name = os.path.join("datasets", "data_42v_20x20")
        #data_folder_name = os.path.join("datasets", "data_132v_100x80")
        #data_folder_name = os.path.join("datasets", "data_225v_100x80")
        data_folder_name = os.path.join("datasets", "data_300v_90x90")
     
    main(data_folder_name)

