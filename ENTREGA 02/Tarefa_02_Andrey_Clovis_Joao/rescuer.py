##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


import os
import random
import plotly.express as px
import numpy as np
import pandas as pd
import classificadores as cls
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from sklearn.metrics import silhouette_score


## Classe que define o Agente Rescuer com um plano fixo

class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.category = 'rescuer'
        self.map = None             # explorer will pass the map
        self.victims = None         # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan

                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def plan_and_rescue(self, rescuer_agents, map, victims):
        """Separates the victims into clusters and indicates one cluster to each rescuer"""

        labels = self.victims_clustering(victims) # Clustering

        # Victims separation
        rescuer_1_victims = dict()
        rescuer_2_victims = dict()
        rescuer_3_victims = dict()
        rescuer_4_victims = dict()
        index = 0
        for key, value in victims.items():
            if labels[index] == 0:
                rescuer_1_victims.update({key: value})               
            elif labels[index] == 1:
                rescuer_2_victims.update({key: value})
            elif labels[index] == 2:
                rescuer_3_victims.update({key: value})
            else:
                rescuer_4_victims.update({key: value})
            index += 1
        victims_per_cluster = [rescuer_1_victims, rescuer_2_victims, rescuer_3_victims, rescuer_4_victims]
        victims_per_cluster_grav = cls.dict2df(victims_per_cluster) #funcaoq ue realiza a classificacao e iserção da gravidade no dict das vitimas
        for agent, victims_per_agent in zip(rescuer_agents, victims_per_cluster_grav):
            agent.mind.go_save_victims(map, victims_per_agent)

        #exit() # Fim do projeto 1

    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        print(f"\n\n*** {self.NAME} ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        self.map.draw()

        print()
        print(f"{self.NAME} List of found victims received from the explorer")
        self.victims = victims

        # print the found victims - you may comment out
        # for seq, data in self.victims.items():
        #    coord, vital_signals = data
        #    x, y = coord
        #    print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")

        #print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.__planner()
        print(f"{self.NAME} PLAN")
        i = 1
        self.plan_x = 0
        self.plan_y = 0
        for a in self.plan:
            self.plan_x += a[0]
            self.plan_y += a[1]
            print(f"{self.NAME} {i}) dxy=({a[0]}, {a[1]}) vic: a[2] => at({self.plan_x}, {self.plan_y})")
            i += 1

        print(f"{self.NAME} END OF PLAN")
                  
        self.set_state(VS.ACTIVE)
        
    def __depth_search(self, actions_res):
        enough_time = True

        if self.NAME == "RESCUER1PINK":
            direction_stack = [7, 0, 1, 2, 3, 4, 5, 6]
        elif self.NAME == "RESCUER2CYAN":
            direction_stack = [5, 4, 3, 2, 1, 0, 7, 6]
        elif self.NAME == "RESCUER3YELLOW":
            direction_stack = [3, 4, 5, 6, 7, 0, 1, 2]
        else:
            direction_stack = [1, 0, 7, 6, 5, 4 ,3, 2]

        ##print(f"\n{self.NAME} actions results: {actions_res}")
        for i, ar in enumerate(actions_res):

            if ar != VS.CLEAR:
                ##print(f"{self.NAME} {i} not clear")
                continue

            # planning the walk
            dx, dy = Rescuer.AC_INCR[direction_stack[i]]  # get the increments for the possible action
            target_xy = (self.plan_x + dx, self.plan_y + dy)

            # checks if the explorer has not visited the target position
            if not self.map.in_map(target_xy):
                ##print(f"{self.NAME} target position not explored: {target_xy}")
                continue

            # checks if the target position is already planned to be visited 
            if (target_xy in self.plan_visited):
                ##print(f"{self.NAME} target position already visited: {target_xy}")
                continue

            # Now, the rescuer can plan to walk to the target position
            self.plan_x += dx
            self.plan_y += dy
            difficulty, vic_seq, next_actions_res = self.map.get((self.plan_x, self.plan_y))
            #print(f"{self.NAME}: planning to go to ({self.plan_x}, {self.plan_y})")

            if dx == 0 or dy == 0:
                step_cost = self.COST_LINE * difficulty
            else:
                step_cost = self.COST_DIAG * difficulty

            #print(f"{self.NAME}: difficulty {difficulty}, step cost {step_cost}")
            #print(f"{self.NAME}: accumulated walk time {self.plan_walk_time}, rtime {self.plan_rtime}")

            # check if there is enough remaining time to walk back to the base
            if self.plan_walk_time + step_cost > self.plan_rtime:
                enough_time = False
                #print(f"{self.NAME}: no enough time to go to ({self.plan_x}, {self.plan_y})")
            
            if enough_time:
                # the rescuer has time to go to the next position: update walk time and remaining time
                self.plan_walk_time += step_cost
                self.plan_rtime -= step_cost
                self.plan_visited.add((self.plan_x, self.plan_y))

                if vic_seq == VS.NO_VICTIM:
                    self.plan.append((dx, dy, False)) # walk only
                    #print(f"{self.NAME}: added to the plan, walk to ({self.plan_x}, {self.plan_y}, False)")

                if vic_seq != VS.NO_VICTIM:
                    # checks if there is enough remaining time to rescue the victim and come back to the base
                    if self.plan_rtime - self.COST_FIRST_AID < self.plan_walk_time:
                        print(f"{self.NAME}: no enough time to rescue the victim")
                        enough_time = False
                    else:
                        self.plan.append((dx, dy, True))
                        #print(f"{self.NAME}:added to the plan, walk to and rescue victim({self.plan_x}, {self.plan_y}, True)")
                        self.plan_rtime -= self.COST_FIRST_AID

            # let's see what the agent can do in the next position
            if enough_time:
                self.__depth_search(self.map.get((self.plan_x, self.plan_y))[2]) # actions results
            else:
                return

        return
    
    def __planner(self):
        """ A private method that calculates the walk actions in a OFF-LINE MANNER to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        """ This plan starts at origin (0,0) and chooses the first of the possible actions in a clockwise manner starting at 12h.
        Then, if the next position was visited by the explorer, the rescuer goes to there. Otherwise, it picks the following possible action.
        For each planned action, the agent calculates the time will be consumed. When time to come back to the base arrives,
        it reverses the plan."""

        # This is a off-line trajectory plan, each element of the list is a pair dx, dy that do the agent walk in the x-axis and/or y-axis.
        # Besides, it has a flag indicating that a first-aid kit must be delivered when the move is completed.
        # For instance (0,1,True) means the agent walk to (x+0,y+1) and after walking, it leaves the kit.

        self.plan_visited.add((0,0)) # always start from the base, so it is already visited
        difficulty, vic_seq, actions_res = self.map.get((0,0))
        self.__depth_search(actions_res)

        # push actions into the plan to come back to the base
        if self.plan == []:
            return

        come_back_plan = []

        for a in reversed(self.plan):
            # triple: dx, dy, no victim - when coming back do not rescue any victim
            come_back_plan.append((a[0]*-1, a[1]*-1, False))

        self.plan = self.plan + come_back_plan
        
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           #input(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if there_is_vict:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

    def victims_clustering(self, victims):
        """Executes a K-Means algorithm 100 times to find the best clustering
        @:param victims: the unified victims dictionary"""
        victims_features = []

        for victim in victims.values():
            victims_features.append([0, np.array(list(victim[0]))]) # considera somente as
                                                                    # coordenadas x  y para o clustering

        min_inertia = np.inf
        final_clustering = []
        for iteration in range(300):
            inertia, clustering = self.k_means_clustering(victims_features)
            if inertia < min_inertia:
                min_inertia = inertia
                final_clustering = clustering

        print(f'K-means clustering result')
        print(f'Inertia: {min_inertia}')

        labels = [i[0] for i in final_clustering]
        x = [i[1][0] for i in final_clustering]
        y = [i[1][1] for i in final_clustering]
        coordinates = np.array(list(zip(x, y)))

        print(f'Silhouette score: {silhouette_score(coordinates, labels)}')

        scatter = px.scatter(x=x, y=y, color=[str(i) for i in labels], color_discrete_sequence=px.colors.qualitative.D3)
        scatter.update_layout(xaxis=dict(zeroline=False, showticklabels=False,
                                         showline=False, title=''),
                              yaxis=dict(zeroline=False, showticklabels=False,
                                         showline=False, title=''),
                              plot_bgcolor='rgb(245, 245, 245)',
                              margin=dict(l=0, r=0, t=0, b=0))
        scatter.update_yaxes(autorange='reversed')
        scatter.update_traces(showlegend=False)
        scatter.write_image('./manual_k_means_clustering.png', format='png', width=500, height=500)

        return np.array(labels)

    @staticmethod
    def k_means_clustering(victims_features):
        """K-Means algorithm
        @:param victims_features: list of labels and features by victim"""
        k = 4

        #Definição dos centroides iniciais
        indexes = np.random.choice(list(range(0, len(victims_features))), k, replace=False)
        centroids = np.array([victims_features[i][1] for i in indexes])

        iteration = 0
        while True:
            iteration += 1
            stop = True

            #'Rotulagem dos dados'
            victim_number = 0
            distances_to_centroids = []
            for victim_data in victims_features:
                distance_per_victim = []
                for centroid in centroids:
                    distance_per_victim.append(np.linalg.norm(victim_data[1]-centroid))
                distance_per_victim = np.array(distance_per_victim)
                cluster = np.argmin(distance_per_victim)
                distances_to_centroids.append(distance_per_victim)
                if cluster != victims_features[victim_number][0]:
                    victims_features[victim_number][0] = cluster
                    stop = False
                victim_number += 1

            #'Cálculo dos novos centroides'
            for cluster in range(0, k):
                victims_in_cluster = list(filter(lambda x: x[0] == cluster, victims_features))
                victims_in_cluster = [i[1] for i in victims_in_cluster]
                centroids[cluster] = np.mean(victims_in_cluster, axis=0)

            if stop and iteration > 1:
                break

        # Cálculo da inércia
        mins = np.min(distances_to_centroids, axis=1)
        inertia = np.sum(np.square(mins))
        return inertia, victims_features

