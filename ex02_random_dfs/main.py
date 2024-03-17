import sys
import os
import time

## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

def main(data_folder_name):
   
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    
    # Instantiate the environment
    env = Env(data_folder)
    
    # config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")
    explorer1_file = os.path.join(data_folder, "explorer_config1.txt")
    explorer2_file = os.path.join(data_folder, "explorer_config2.txt") 
    explorer3_file = os.path.join(data_folder, "explorer_config3.txt")
    explorer4_file = os.path.join(data_folder, "explorer_config4.txt")
    
    # Instantiate agents rescuer and explorer
    resc = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp1 = Explorer(env, explorer1_file, resc)
    exp2 = Explorer(env, explorer2_file, resc)
    exp3 = Explorer(env, explorer3_file, resc)
    exp4 = Explorer(env, explorer4_file, resc)
    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_10v_12x12")
        
    main(data_folder_name)
