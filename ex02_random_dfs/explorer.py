# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
import pandas as pd
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from clustering import victims_clustering
import numpy as np
import test_astar


''' GRUPO DE VARIAVEIS GLOBAIS
PARA CONTROLE DE EXPLORADORES'''

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
        self.path = Stack()        #caminho para retornar a base
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.time_to_comeback = math.ceil(self.TLIM * 0.6)  # set the time to come back to the base
        self.time_to_base = 0        #tempo total para retornar a base atraves de A*
        self.returning = 0          #variavel para ver se o agente esta retornando
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.count_astar = 0       #quantidade de explorar até refazer metodo A*
        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.tempo_apos_astar = self.TLIM #tempo inicial restante
        self.cicles = 0            #ciclos antes de entrar no metodo A*

    def get_next_position(self):
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        
        #cada explorador faz seu método de busca, por isso divide por nome
        if self.NAME == "EXPLORER1BLUE":
            i=0
            while True:
                direction_stack = [0,6,2,4] #ordem de direção a explorar
                direction = direction_stack[i]          
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                    dx,dy = Explorer.AC_INCR[direction]
                    explorado = self.state_explored(self.map, self.x, self.y, dx, dy) #verifica se próima posição já foi explorada
                    if not explorado:
                        ix, iy = Explorer.AC_INCR[direction]
                        return ix, iy, False
                    else:
                        i+=1 #se já foi explorado, tenta outra posição
                else:
                    i+=1 #se há obstaculo, tenta outra posição
                if i>3:
                    ix, iy = self.retorna_position()
                    return ix, iy, True #caso não tenha mais opções, retorna uma posição e tenta explorar novamente
        
                        
        elif self.NAME == "EXPLORER2GREEN":
            i=0
            while True:
                direction_stack = [2,6,0,4] #ordem de direção a explorar
                direction = direction_stack[i]          
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                    dx,dy = Explorer.AC_INCR[direction]
                    explorado = self.state_explored(self.map, self.x, self.y, dx, dy) #verifica se próima posição já foi explorada
                    if not explorado:
                        ix, iy = Explorer.AC_INCR[direction]
                        return ix, iy, False
                    else:
                        i+=1 #se já foi explorado, tenta outra posição
                else:
                    i+=1 #se há obstaculo, tenta outra posição
                if i>3:
                    ix, iy = self.retorna_position()
                    return ix, iy, True #caso não tenha mais opções, retorna uma posição e tenta explorar novamente
                    
        elif self.NAME == "EXPLORER3PURPLE":
            i=0
            while True:
                direction_stack = [4,2,0,6] #ordem de direção a explorar
                direction = direction_stack[i]       
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                    dx,dy = Explorer.AC_INCR[direction]
                    explorado = self.state_explored(self.map, self.x, self.y, dx, dy) #verifica se próima posição já foi explorada
                    if not explorado:
                        ix, iy = Explorer.AC_INCR[direction]
                        return ix, iy, False
                    else:
                        i+=1 #se já foi explorado, tenta outra posição
                else:
                    i+=1 #se há obstaculo, tenta outra posição
                if i>3:
                    ix, iy = self.retorna_position()
                    return ix, iy, True #caso não tenha mais opções, retorna uma posição e tenta explorar novamente
    
        else:
            i=0
            while True:
                direction_stack = [6,4,2,0] #ordem de direção a explorar
                direction = direction_stack[i]          
            # Check if the corresponding position in walls_and_lim is CLEAR
                if obstacles[direction] == VS.CLEAR:
                    dx,dy = Explorer.AC_INCR[direction]
                    explorado = self.state_explored(self.map, self.x, self.y, dx, dy) #verifica se próima posição já foi explorada
                    if not explorado:
                        ix, iy = Explorer.AC_INCR[direction]
                        return ix, iy, False
                    else:
                        i+=1 #se já foi explorado, tenta outra posição
                else:
                    i+=1 #se há obstaculo, tenta outra posição
                if i>3:
                    ix, iy = self.retorna_position()
                    return ix, iy, True #caso não tenha mais opções, retorna uma posição e tenta explorar novamente
                    
        
    def explore(self):
        # get an increment for x and y       
        #try:                           #caso tenha uma exceção de erro, fica parada até próximo passo
        dx, dy, retornou = self.get_next_position()
        # except:
        #     dx=0
        #     dy=0
        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result  == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
           # print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            if not retornou:
                self.walk_stack.push((dx, dy))
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy          
            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}: position: {self.x, self.y}. returning path: {self.path.items}")
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")
            
        return

    def come_back(self):
        #print(f"{self.NAME} coming back!!!")
        #MÉTODO 2
        '''função de retorno para base, conforme caminho fornecido pelo A*'''
        tx, ty = self.path.pop()  #pega proxima posição a retornar       
        dx = tx-self.x            #diferença entre posição atual e a proxima
        dy = ty - self.y          #diferença entre posição atual e a proxima

        # print(f"dx: {dx}, dy: {dy}")
        # print(f"Tempo estimato base: {self.time_to_base}")
        # print(f"Tempo restante: {self.get_rtime()}")
        # print(f" ")
        #MÉTODO 1
        # dx, dy = self.walk_stack.pop()
        # dx = dx * -1
        # dy = dy * -1
        
        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            #print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        #uso de varivais globais para ver término, mapa e vitimas de cada agente
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
      
        if self.get_rtime()>700:
            qtd_ciclos = 50
        elif self.get_rtime()>100:
            qtd_ciclos = 30
        elif self.get_rtime()>50:
            qtd_ciclos = 20
        else:
            qtd_ciclos = 10
        
        if self.NAME == "EXPLORER1BLUE":           
            if (self.time_to_base*3 < self.tempo_apos_astar) and (self.returning==0): #tempo de retornar é menor que tempo restante, continua a explorar         
                if self.cicles == 0: 
                    self.path, self.time_to_base = self.astar_method(self.map, self.x, self.y)
                    self.tempo_apos_astar = self.get_rtime()
                self.explore()
                self.cicles+=1
                if self.cicles>qtd_ciclos:
                    self.cicles=0                                
                return True
            if self.path.is_empty() or (self.x == 0 and self.y == 0):
                # time to come back to the base
                # time to wake up the rescuer
                exp1_finished = True
                exp1_map = self.map
                exp1_victims = self.victims
                #se todos exploradores finalizaram, chama função de unificar
                if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                    self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)                 
            else:
                self.returning==1
                self.come_back()
                return True
            
        elif self.NAME == "EXPLORER2GREEN":
            if (self.time_to_base*3 < self.tempo_apos_astar) and (self.returning==0): #tempo de retornar é menor que tempo restante, continua a explorar
                if self.cicles == 0: 
                    self.path, self.time_to_base = self.astar_method(self.map, self.x, self.y)
                    self.tempo_apos_astar = self.get_rtime()
                self.explore()
                self.cicles+=1
                if self.cicles>qtd_ciclos:
                    self.cicles=0                                
                return True
            if self.path.is_empty() or (self.x == 0 and self.y == 0):
                # time to come back to the base
                # time to wake up the rescuer
                exp2_finished = True
                exp2_map = self.map
                exp2_victims = self.victims
                #se todos exploradores finalizaram, chama função de unificar
                if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                    self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                
            else:
                self.returning==1
                self.come_back()
                return True

        elif self.NAME == "EXPLORER3PURPLE":
            if (self.time_to_base*3 < self.tempo_apos_astar) and (self.returning==0): #tempo de retornar é menor que tempo restante, continua a explorar
                if self.cicles == 0: 
                    self.path, self.time_to_base = self.astar_method(self.map, self.x, self.y)
                    self.tempo_apos_astar = self.get_rtime()
                self.explore()
                self.cicles+=1
                if self.cicles>qtd_ciclos:
                    self.cicles=0                                
                return True
            if self.path.is_empty() or (self.x == 0 and self.y == 0):
                # time to come back to the base
                # time to wake up the rescuer
                exp3_finished = True
                exp3_map = self.map
                exp3_victims = self.victims
                #se todos exploradores finalizaram, chama função de unificar
                if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                    self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                    return False
                
            else:
                self.returning==1
                self.come_back()
                return True

        else:
            if (self.time_to_base*3 < self.tempo_apos_astar) and (self.returning==0): #tempo de retornar é menor que tempo restante, continua a explorar
                if self.cicles == 0: 
                    self.path, self.time_to_base = self.astar_method(self.map, self.x, self.y)
                    self.tempo_apos_astar = self.get_rtime()
                self.explore()
                self.cicles+=1
                if self.cicles>qtd_ciclos:
                    self.cicles=0                                
                return True
            if self.path.is_empty() or (self.x == 0 and self.y == 0):
                # time to come back to the base
                # time to wake up the rescuer
                exp4_finished = True
                exp4_map = self.map
                exp4_victims = self.victims
                #se todos exploradores finalizaram, chama função de unificar
                if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
                    self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                    return False
                
            else:
                self.returning==1
                self.come_back()
                return True   

    #unifica mapa e vitimas de todos os agentes   
    def unifica(self,exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims):        
        #combined victims
        merged_maps = Map()
        merged_victims = { **exp1_victims, **exp2_victims, **exp3_victims, **exp4_victims}
        #combined maps
        merged_maps.map_data = { **exp1_map.map_data, **exp2_map.map_data, **exp3_map.map_data, **exp4_map.map_data}
        victims_clustering(merged_victims) #função de agrupamento e divisão de vitimas encontradas
        self.resc.go_save_victims(merged_maps, merged_victims) 
    
    def state_explored(self, mapa, actual_x, actual_y, dx, dy):
        '''Verifica es próxima posição já foi explorada pelo agente'''
        next_position = (actual_x+dx, actual_y+dy)
        if next_position in mapa.map_data:
            return True
        else:
            return False
        
    def astar_method(self,mapa,actual_x,actual_y):  
        '''Metodo A* para retorno a base, verifica caminho e tempo'''
        max_x = 0
        max_y = 0
        min_x = 0
        min_y = 0
        #abs_map = mapa.map_data.copy() #copia do mapa recebido
        abs_map = dict()
        if self.path.is_empty():
            obj_x = 0
            obj_y = 0
        else:
            obj_x = self.path.items[-1][0]
            obj_y = self.path.items[-1][1]
        
        for key in mapa.map_data.keys():            #verifica qual menos posição relativa
            if key[0] < min_x:                #para fazer uma alteração em relação as coordenadas do mapa
                min_x = key[0]
            if key[1] < min_y:
                min_y = key[1]    
        base_x = 0 - min_x                   #altera posição relativa da base e posição atual                           
        base_y = 0 - min_y
        new_x = actual_x - min_x
        new_y = actual_y - min_y
        obj_x = obj_x - min_x
        obj_y = obj_y - min_y
        
        for key in mapa.map_data.keys():                        #altera posição do mapa relativo
            new_k0 = key[0] - min_x
            new_k1 = key[1] - min_y
            abs_map[(new_k0, new_k1)] = mapa.map_data[key]
           # del abs_map[key]               #NOVO
            
        for key in abs_map.keys():         #adquire maior indices para criar matriz de posições
            if key[0] > max_x:
                max_x = key[0]
            if key[1] > max_y:
                max_y = key[1]
            
        tam_maze = max(max_x,max_y)+1   #tamanho matriz
        maze_matrix=np.full((tam_maze,tam_maze), 100.0)     #preenche matriz com dificuldade 100
        for i in abs_map.keys():      
            #maze_matrix[i[1]][i[0]] = abs_map.get(i)[0]       #altera matriz com as posi;óe e dificuldades conhecidas  
            maze_matrix[i[1]][i[0]] = abs_map.get(i)[0] 
        
        path, time_to_base = test_astar.solve_comeback(new_x,new_y,maze_matrix,obj_x,obj_y, self.COST_LINE, self.COST_DIAG) #função A*
        
        for i in range(0,len(path.items)):
            path.items[i] = (path.items[i][0] + min_x, path.items[i][1] + min_y)  #retorna posições absoluta matriz para relativa do mapa do agente
          
        for i in range(1,len(self.path.items)+1):  
            path.items.insert(0,self.path.items[-i])   
        
        time_to_base = time_to_base+self.time_to_base
                      
        return path, time_to_base
    

    def retorna_position(self):
    
        if self.walk_stack.is_empty():
            obstacles = self.check_walls_and_lim()
            for i in range(0,7):
                if obstacles[i] == VS.CLEAR:
                    dx,dy = Explorer.AC_INCR[i]  
        else:
            dx, dy = self.walk_stack.pop()
            dx = dx * -1
            dy = dy * -1
        
        return dx, dy

           
        
    def ag_dead(self):
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
            exp1_finished = True
            exp1_map = Map()
            exp1_victims = {}
        elif self.NAME == "EXPLORER2GREEN":
            exp2_finished = True
            exp2_map = Map()
            exp2_victims = {}
        elif self.NAME == "EXPLORER3PURPLE":
            exp3_finished = True
            exp3_map = Map()
            exp3_victims = {}
        else:
            exp4_finished = True
            exp4_map = Map()
            exp4_victims = {}
            
        if exp1_finished & exp2_finished & exp3_finished & exp4_finished:
            self.unifica(exp1_map,exp2_map,exp3_map,exp4_map,exp1_victims,exp2_victims,exp3_victims,exp4_victims)  
                 
