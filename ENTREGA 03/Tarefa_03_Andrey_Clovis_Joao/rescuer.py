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
import clustering as clus
import genetic_search as ag
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from sklearn.metrics import silhouette_score
from vs.environment import Env
from collections import Counter
import pandas as pd

## Classe que define o Agente Rescuer com um plano fixo

class Rescuer(AbstAgent):
    def __init__(self, env, config_file, datafolder):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = 4    # number of explorers
        self.received_maps = 0                       # counts the number of explorers' maps
        self.category = 'rescuer'
        self.map = Map()            # explorer will pass the map
        self.victims = {}        # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        self.seq = pd.DataFrame(columns=['ID', 'x', 'y', 'grav', 'classe'])   #lista de vitimas salvas
        self.datafolder = datafolder

                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)


    def cluster_and_plan(self, rescuer_agents, map, victims):
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
        victims_per_cluster = cls.dict2df(victims_per_cluster) #funcaoque realiza a classificacao e iserção da gravidade no dict das vitimas
        for agent, victims_per_agent in zip(rescuer_agents, victims_per_cluster):
            agent.go_save_victims(map, victims_per_agent)

        #exit() # Fim do projeto 1

    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        print(f"\n\n*** {self.NAME} ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        #self.map.draw()

        print()
        print(f"{self.NAME} List of found victims received from the explorer")
        self.victims = victims

        # print the found victims - you may comment out
        #for seq, data in self.victims.items():
            #coord, vital_signals = data
            #x, y = coord
            #grav = vital_signals[6]
            #print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}, gravidade: {grav}")

        #print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.plan = ag.planner_genetic_algorithm(map, victims, self.TLIM)

        print(f"{self.NAME} PLAN")

        i=1
        self.plan_x = 0
        self.plan_y = 0
        for a in self.plan:
            self.plan_x += a[0]
            self.plan_y += a[1]
            print(f"{self.NAME} {i}) dxy=({a[0]}, {a[1]}) vic: a[2] => at({self.plan_x}, {self.plan_y}, {a[2]})")
            i += 1

        print(f"{self.NAME} END OF PLAN")
                  
        self.set_state(VS.ACTIVE)
        
        
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
                    #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                    key_vitima = [k for k, v in self.victims.items() if v[0] == (self.x,self.y)]
                    gravidade = self.victims.get(key_vitima[0])
                    gravidade = gravidade[1][6]
                    if not key_vitima[0] in self.seq['ID'].values:
                        new_row = [key_vitima[0], self.x, self.y, 0, gravidade]
                        self.seq.loc[len(self.seq)] = new_row
                        if self.NAME == "RESCUER1PINK":
                            self.seq.to_csv("clusters\\seq1.txt", header=False, index=False) #salva resultados sequencia 1
                        elif self.NAME == "RESCUER2CYAN":
                            self.seq.to_csv("clusters\\seq2.txt", header=False, index=False) #salva resultados sequencia 2
                        elif self.NAME == "RESCUER3YELLOW":
                            self.seq.to_csv("clusters\\seq3.txt", header=False, index=False) #salva resultados sequencia 3
                        else:
                            self.seq.to_csv("clusters\\seq4.txt", header=False, index=False) #salva resultados sequencia 4
                    
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

    def victims_clustering(self, victims):
        """Executes a K-Means algorithm 200 times to find the best clustering
        @:param victims: the unified victims dictionary"""
        victims_features = []

        for victim in victims.values():
            victims_features.append([0, np.array(list(victim[0]))]) # considera somente as
                                                                    # coordenadas x  y para o clustering
        # K FIXO EM 4
        min_inertia = np.inf
        best_clustering = []
        for iteration in range(200):
            try:
                inertia, clustering = clus.k_means_clustering(victims_features)
            except:
                pass
            if inertia < min_inertia:
                min_inertia = inertia
                best_clustering = clustering

        labels = [i[0] for i in best_clustering]
        x = [i[1][0] for i in best_clustering]
        y = [i[1][1] for i in best_clustering]
        coordinates = np.array(list(zip(x, y)))

        silhouette = silhouette_score(coordinates, labels)

        print(f'K-means clustering result')
        print(f'Inertia: {min_inertia}')
        print(f'Silhouette score: {silhouette}')

        # K VARIÁVEL
        """best_silhouette = 0

        for k in range(3, 8):
            min_inertia = np.inf
            best_clustering = []
            for iteration in range(200):
                try:
                    inertia, clustering = self.k_means_clustering(victims_features, k=4)
                except:
                    inertia = np.inf
                if inertia < min_inertia:
                    min_inertia = inertia
                    best_clustering = clustering

            labels = [i[0] for i in best_clustering]
            x = [i[1][0] for i in best_clustering]
            y = [i[1][1] for i in best_clustering]
            coordinates = np.array(list(zip(x, y)))

            silhouette_ = silhouette_score(coordinates, labels)

            if silhouette_ > best_silhouette:
                final_labels = [i[0] for i in best_clustering]
                final_x = [i[1][0] for i in best_clustering]
                final_y = [i[1][1] for i in best_clustering]
                final_inertia = min_inertia
                final_k = k
                best_silhouette = silhouette_"""

        scatter = px.scatter(x=x, y=y, color=[str(i) for i in labels], color_discrete_sequence=px.colors.qualitative.D3)
        scatter.update_layout(xaxis=dict(zeroline=False, showticklabels=False,
                                         showline=False, title=''),
                              yaxis=dict(zeroline=False, showticklabels=False,
                                         showline=False, title=''),
                              plot_bgcolor='rgb(245, 245, 245)',
                              margin=dict(l=0, r=0, t=0, b=0))
        scatter.update_yaxes(autorange='reversed')
        scatter.update_traces(showlegend=False)
        scatter.write_image('./k_means_clustering.png', format='png', width=500, height=500)

        return np.array(labels)

            
    
    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """
        self.map.map_data.update(explorer_map)
        self.victims.update(victims)
        
        self.received_maps += 1
        if self.received_maps == self.nb_of_explorers:
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent
            for i in range (1,4):
                filename = f"rescuer_config_{i+1:1d}.txt"
                config_file = os.path.join(self.datafolder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, self.datafolder) 
                rescuers[i].map = self.map     # each rescuer have the map
            self.cluster_and_plan(rescuers, self.map, self.victims)

            