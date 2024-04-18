# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
from warnings import warn
import heapq
import  math

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


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node):
    path = Stack()
    current = current_node
    while current is not None:
        path.push(current.position)
        current = current.parent
    #return path[::-1]  # Return reversed path
    return path


def astar(maze, start, end, cost_line, cost_diag, allow_diagonal_movement = True, rescue=False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    if rescue:
        max_iterations = 50000
    else:
        max_iterations = 1000

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          print("Não realizado A*")

          return return_path(current_node), current_node.position
        
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node), None

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[1]][node_position[0]] == 100.0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue
              
            dif_x = abs(child.position[0]-child.parent.position[0])  
            dif_y = abs(child.position[1]-child.parent.position[1]) 
            if (dif_x+dif_y) == 1:
              mov_cost = cost_line*maze[child.position[1]][child.position[0]]
            else:
              mov_cost = cost_diag*maze[child.position[1]][child.position[0]]
            
            # Create the f, g, and h values
            child.g = current_node.g + mov_cost
            child.h = math.sqrt(((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None, None


def solve_comeback(actual_x, actual_y, adj_map, base_x, base_y, cost_line, cost_diag, rescue=False):
        '''revolve retorno e tempo de retorno do agente para base'''
        start = (actual_x,actual_y)    #posição atual de inicio
        end = (base_x,base_y) #posição final (base)
        path, last_node = astar(adj_map,start,end, cost_line, cost_diag, rescue=rescue) #função que retorna caminho


        #calculo do tempo

        time_to_base = 0                                      
        for i in range(0,len(path.items)-1):                  #veriica se caminho é reto ou diagonal e passa custo
          dif_x = abs(path.items[i][0]-path.items[i+1][0])  
          dif_y = abs(path.items[i][1]-path.items[i+1][1]) 
          if (dif_x+dif_y) == 1:
            time_to_base = time_to_base+cost_line*adj_map[path.items[i+1][1]][path.items[i+1][0]]
          else:
            time_to_base = time_to_base+cost_diag*adj_map[path.items[i+1][1]][path.items[i+1][0]]

        # Adiciona a distância de manhatan ao percurso faltante caso o A* não seja finalizado
        if last_node is not None and rescue:
            time_to_base += 1.5*manhattan_distance(last_node, end)

        return path, time_to_base


def manhattan_distance(start, end):
    return abs(start[0]-end[0]) + abs(start[1]-end[1])