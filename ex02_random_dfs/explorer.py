# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

exp1_finished = False
exp2_finished = False
exp3_finished = False
exp4_finished = False
exp1_map = Map()
exp2_map = Map()
exp3_map = Map()
exp4_map = Map()
exp1_victims = {}
exp2_victims = {}
exp3_victims = {}
exp4_victims = {}

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """
        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.time_to_comeback = math.ceil(self.TLIM * 0.6)  # set the time to come back to the base
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        if self.NAME == "EXPLORER1BLUE":
            # Loop until a CLEAR position is found
            while True:
                direction = random.randint(0,7)           
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                 return Explorer.AC_INCR[direction]
        elif self.NAME == "EXPLORER2GREEN":
            # Loop until a CLEAR position is found
            while True:  
                direction = random.randint(0,7)         
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                 return Explorer.AC_INCR[direction]
        elif self.NAME == "EXPLORER3PURPLE":
            # Loop until a CLEAR position is foun 
            while True:
                direction = random.randint(0,7)      
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                 return Explorer.AC_INCR[direction]
        else:
            # Loop until a CLEAR position is found
            while True:   
                direction = random.randint(0,7)       
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                 return Explorer.AC_INCR[direction]
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        global exp1_finished
        global exp2_finished
        global exp3_finished
        global exp4_finished
        global exp1_map
        global exp2_map
        global exp3_map
        global exp4_map
        global exp1_victims
        global exp2_victims
        global exp3_victims
        global exp4_victims

        if self.NAME == "EXPLORER1BLUE":
            if self.get_rtime() > self.time_to_comeback:
                self.explore()
                return True
            else:
                # time to come back to the base
                if self.walk_stack.is_empty():
                    # time to wake up the rescuer
                    # pass the walls and the victims (here, they're empty)
                    exp1_finished = True
                    exp1_map = self.map
                    exp1_victims = self.victims
                    print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
                    input(f"{self.NAME}: type [ENTER] to proceed")
                    if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                        self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                    return False
                else:
                    self.come_back()
                    return True
        elif self.NAME == "EXPLORER2GREEN":
            if self.get_rtime() > self.time_to_comeback:
                self.explore()
                return True
            else:
                # time to come back to the base
                if self.walk_stack.is_empty():
                    # time to wake up the rescuer
                    # pass the walls and the victims (here, they're empty)
                    exp2_finished = True
                    exp2_map = self.map
                    exp2_victims = self.victims
                    print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
                    input(f"{self.NAME}: type [ENTER] to proceed")
                    if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                        self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                    return False
                else:
                    self.come_back()
                    return True    
                
        elif self.NAME == "EXPLORER3PURPLE":
            if self.get_rtime() > self.time_to_comeback:
                self.explore()
                return True
            else:
                # time to come back to the base
                if self.walk_stack.is_empty():
                    # time to wake up the rescuer
                    # pass the walls and the victims (here, they're empty)
                    exp3_finished = True
                    exp3_map = self.map
                    exp3_victims = self.victims
                    print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
                    input(f"{self.NAME}: type [ENTER] to proceed")
                    if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                        self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                    return False
                else:
                    self.come_back()
                    return True
                
        else:
            if self.get_rtime() > self.time_to_comeback:
                self.explore()
                return True
            else:
                # time to come back to the base
                if self.walk_stack.is_empty():
                    # time to wake up the rescuer
                    # pass the walls and the victims (here, they're empty)
                    exp4_finished = True
                    exp4_map = self.map
                    exp4_victims = self.victims
                    print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
                    input(f"{self.NAME}: type [ENTER] to proceed")
                    if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                        self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)                  
                    return False
                else:
                    self.come_back()
                    return True        
        
       
    def unifica(self,exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims):        
        #combined victims
        merged_victims = { **exp1_victims, **exp2_victims, **exp3_victims, **exp4_victims}
        print(merged_victims.items())

        self.resc.go_save_victims(exp1_map, merged_victims)  