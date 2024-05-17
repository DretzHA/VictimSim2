import test_astar
import random
import numpy as np
import classificadores
import pandas as pd
import warnings
pd.set_option('future.no_silent_downcasting', True)
warnings.simplefilter(action='ignore', category=FutureWarning)
cost_matrix = []

def planner_genetic_algorithm(map, victims, tlim):
    """Genetic algorithm for permutation problem. Finds a suboptimal sequence for victim sequence within a cluster
    @:param map: map from explorers
     @:param: victims: victims dict
     @:param: tlim: time limit for rescuer
     """

    global cost_matrix

    p = 100  # tamanho da população
    pCROSS = 0.8  # probabilidade de crossover
    pMUT = 0.08  # probabilidade de mutação

    victims_list = [[i[0]] + [i[1][6]] + [i[1][7]] for i in list(victims.values())]  # Coordenadas + classe gravidade + gravidade

    cost_matrix = np.full((len(victims_list),len(victims_list)), None).tolist()  # zera a matriz de custos dos trajetos
    # entre pontos do cluster

    population = generate_population(len(victims_list), p)  # geração da população

    generation = 0

    while True:

        if generation > 100:
            break

        #print(f'Generation {generation}')

        fits = []
        costs = []

        population = crossover(population, pCROSS)  # processo de crossover

        population = mutation(population, pMUT)  # processo de mutação

        # Cálculo do custo e da função fit para cada indivíduo
        for individual in population:
            total_time, sequence = calculate_cost(individual, victims_list)
            costs.append(total_time)
            fit = fitness_function(sequence, victims_list, map)
            fits.append(fit)

        print(f'Mean fit: {sum(fits)/len(fits)}\nMaximum fit: {max(fits)}')

        # ranquamento por fit
        zipped_individuals = zip(fits, costs, population)
        zipped_individuals = sorted(zipped_individuals, key=lambda x: x[0], reverse=True)

        # eliminação da metade da população com menor fit
        fits, costs, population = zip(*zipped_individuals)
        fits = list(fits[:len(fits)//2])
        costs = list(costs[:len(costs) // 2])
        population = list(population[:len(population)//2])

        generation += 1

    # Determinação da sequência com fit máximo E custo (tempo) mínimo
    max_value = max(fits)
    max_fits = [i for i, val in enumerate(fits) if val == max_value]
    max_fits_costs = [costs[index] for index in max_fits]
    min_cost_index = costs.index(min(max_fits_costs))

    print(f'Best sequence with fit {fits[min_cost_index]} and cost {costs[min_cost_index]} '
          f'is\n {population[min_cost_index]}')

    victims_coordinates = [victims_list[i] for i in population[min_cost_index]]
    print(victims_coordinates)
    victims_coordinates = [i[0] for i in victims_coordinates]

    # Cálculo do trajeto completo por meio do A*
    plan = detail_rescue_plan(victims_coordinates, map, tlim)

    plan = plan.items[::-1]

    directions = []

    # Transformando a lista de coordenadas em lista de dx, dy
    for i in range(len(plan)-1):

        if plan[i+1] in victims_coordinates:
            directions.append((plan[i+1][0] - plan[i][0], plan[i+1][1] - plan[i][1], True))
        else:
            directions.append((plan[i+1][0] - plan[i][0], plan[i+1][1] - plan[i][1], False))

    return directions


def fitness_function(sequence, victims_list, map):
    """Função de fit para uma dada sequência de vítimas
    @:param sequence: sequência de vítimas
    @:param victims_list lista de dados das vítimas"""
    
    df_prioridades = pd.DataFrame(columns=['x1', 'x2', 'x3', 'x4', 'p']) #cria dataframe 
    dict_prioridades = {} # dictoriny que será transformada em dataframe, para agilizar processamento
    i_d = 0 #contador para add entradas no dict
    
    
    fit = 0

    _fourth = len(sequence)//4
    
    for idx, i in enumerate(sequence):
        
        # TAREFA 02
        
        # Vítimas mais críticas aumentam mais o fit de uma sequência.
        # Quanto mais ao início da sequência estiver uma vítima, maior o aumento de fit

        if idx < _fourth:
            factor = 4
        elif idx < 2*_fourth:
            factor = 3
        elif idx < 3*_fourth:
            factor = 2
        else:
            factor = 1

        if victims_list[i][1] == 1:
            fit += 10*factor
        elif victims_list[i][1] == 2:
            fit += 6*factor
        elif victims_list[i][1] == 3:
            fit += 2*factor
        elif victims_list[i][1] == 4:
            fit += factor
                
        # TAREFA 03 - FIT ARTAVÉS DA PRIORIDADE DA REDE NEURAL##################################
        
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)) #espaços adjacentes para calcular dificuldad ed acesso
        coord = victims_list[i][0] #coordenada da vitima
        coord_map = map.map_data.get(coord) #get nos values do dict do mapa
        x1 = coord_map[0] #dificuldade no local da vitima
        
        for new_position in adjacent_squares: #para cada posição adjacesnte da vitima
            new_coord = (coord[0] + new_position[0], coord[1] + new_position[1]) # coordenada adjacesnte
            if new_coord in map.map_data: #se existe no mapa, então soma o valor da dificuldade
                coord_map = map.map_data.get(new_coord)
                x1 += coord_map[0] 
            else: # se não existe no mapa, atribui valor 100
                x1 += 100

        x2 = victims_list[i][2] #gravidade da vitima  
        x3 = np.sqrt(pow(coord[0],2)+pow(coord[1],2)) #distancia da vitima ate a base
        x4 = idx+1 #posicao na sequencia
        dict_prioridades[i_d] = {'x1': x1,'x2': x2,'x3': x3,'x4': x4,'p': 0} #insere valores no dict das prioridaddes
        i_d = i_d + 1
        
    df_prioridades = pd.DataFrame.from_dict(dict_prioridades, 'index') #transofrma o dict em dataframe
    prior_value = classificadores.test_neural_regressor_prior(df_prioridades)  #retorna dataframe com prioridades
    fit = fit + prior_value['p'].sum() #soma prioridades ao fit
    
    return fit


def mutation(population, pMUT):
    """Função de mutação sobre uma população
    @:param population: população do AG
    @:param pMUT: probabilidade de mutação"""

    for individual in population:

        if random.random() > pMUT:
            continue

        # Mutação por troca mútua de alelos no mesmo indivíduo

        idx1, idx2 = random.sample(range(0, len(individual)), 2)

        temp1 = individual[idx1]
        temp2 = individual[idx2]

        individual[idx1] = temp2
        individual[idx2] = temp1

    return population


def crossover(population, pCROSS):
    """Função de crossover sobre uma população
    @:param population: população do AG
    @:param pCROSS: probabilidade de crossover"""

    random.shuffle(population)

    for i in range(0, len(population) - 1, 2):

        list1 = population[i].copy()
        list2 = population[i + 1].copy()

        if random.random() > pCROSS:
            population.append(list1)
            population.append(list2)
            continue

        min_len = min(len(list1), len(list2))

        # índices de início e fim do crossover
        start = random.randint(0, min_len - 1)
        length = random.randint(1, min_len - start)

        # troca de partes entre cromossomos
        temp1 = list1[start:start + length]
        temp2 = list2[start:start + length]
        list1[start:start + length] = temp2
        list2[start:start + length] = temp1

        # Mecanismo que corrige crossovers infactíveis
        # Os elementos duplicados que não sofreram crossover são substituídos até que não haja mai duplicados
        for index, element in enumerate(list1):
            if index not in range(start, start+length):
                if list1.count(element) > 1:
                    while True:
                        list1[index] = random.choice(range(0, len(list1)))
                        if list1.count(list1[index]) == 1:
                            break
        for index, element in enumerate(list2):
            if index not in range(start, start+length):
                if list2.count(element) > 1:
                    while True:
                        list2[index] = random.choice(range(0, len(list2)))
                        if list2.count(list2[index]) == 1:
                            break

        # Adição dos novos cromossomos à população
        population.append(list1)
        population.append(list2)

    return population


def generate_population(n_alleles, n_individuals):
    """Criação de uma população
    @:param n_alleles: número de alelos de cada indivíduo
    @:param n_individuals: número de cromossomos da população"""

    return [list(np.random.permutation(n_alleles)) for i in range(n_individuals)]


def manhattan_distance(x_start, y_start, x_end, y_end):
    """Cálculo da distância de Manhattan para estimação do custo do caminho entre coordenadas"""

    return abs(x_end - x_start) + abs(y_end - y_start)


def calculate_cost(sequence, victims_list):
    """Cálculo do custo de uma sequência. O custo é o tempo estimado para a sequência. A estimativa é pela
    distância de Manhattan entre os pontos
    @:param sequence: a sequência de vítimas
    @:param victims_list: lista de vítimas, com coordenadas e gravidades"""

    # Custo da base ao primeiro ponto
    cost = 1 + manhattan_distance(0, 0, victims_list[sequence[0]][0][0], victims_list[sequence[0]][0][1])

    for i in range(len(sequence)-1):

        # Consulta uma variável de memória dos custos dos trajetos
        try:
            marginal_cost = cost_matrix[sequence[i]][sequence[i+1]] + 1  # +1 por conta do kit de primeiros socorros
            cost += marginal_cost
        # Se a distãncia não estiver na variável cost_matrix, calcula a distância de Manhattan e guarda
        except TypeError:
            marginal_cost = manhattan_distance(victims_list[sequence[i]][0][0], victims_list[sequence[i]][0][1],
                                               victims_list[sequence[i+1]][0][0], victims_list[sequence[i+1]][0][1])
            cost_matrix[sequence[i]][sequence[i+1]] = marginal_cost
            cost_matrix[sequence[i+1]][sequence[i]] = marginal_cost
            cost += (marginal_cost + 1)

    # Custo do último ponto à base
    cost += manhattan_distance(victims_list[sequence[-1]][0][0], victims_list[sequence[-1]][0][1], 0, 0)

    return cost, sequence


def detail_rescue_plan(victims_coordinates, map, TLIM):
    """Obtenção do trajeto completo de uma sequência de vítimas. Usa o A*
    @:param victims_coordinates: lista de dados das vítimas
    @:param map: mapa recebido dos exploradores
    @:param TLIM: tempo limite"""

    total_time = 0
    # tempo e caminho da base ao primeiro ponto
    path, marginal_time = astar_method(map, 0, 0, victims_coordinates[0][0], victims_coordinates[0][1])
    total_time += marginal_time + 1

    for i, victim in enumerate(victims_coordinates[:-1]):
        # tempo e caminho do ponto atual ao próximo ponto
        marginal_path, marginal_time = astar_method(map,
                                                    victims_coordinates[i][0], victims_coordinates[i][1],
                                                    victims_coordinates[i+1][0], victims_coordinates[i+1][1])

        # tempo e caminho entre o próximo ponto e a base
        path_to_base, time_to_base = astar_method(map, victims_coordinates[i+1][0], victims_coordinates[i+1][1], 0, 0)

        # Se não houver tempo para seguir mais um ponto e retornar à base, retorna a base a partir do ponto atual
        if total_time + marginal_time + time_to_base >= 0.95*TLIM:

            total_time += last_time_to_base

            for j in range(2, len(last_path_to_base.items) + 1):
                path.items.insert(0, last_path_to_base.items[-j])

            print(f"Full path WON'T be executed. Returning partial path with total time of {total_time}...")
            return path

        total_time += marginal_time + 1

        for j in range(2, len(marginal_path.items) + 1):
            path.items.insert(0, marginal_path.items[-j])

        # Armazenamento do tempo e caminho à base para a próxima iteração
        last_path_to_base = path_to_base
        last_time_to_base = time_to_base

    # Inserção do caminho do último ponto à base
    for j in range(2, len(last_path_to_base.items) + 1):
        path.items.insert(0, last_path_to_base.items[-j])

    total_time += last_time_to_base
    print(f"Full path WILL be executed. Returning path with total time of {total_time}...")

    return path


def astar_method(mapa, actual_x, actual_y, obj_x, obj_y):
        """Computes the path and time from current position to destination through A*"""
        max_x = 0
        max_y = 0
        min_x = 0
        min_y = 0
        # abs_map = mapa.map_data.copy() #copia do mapa recebido
        abs_map = dict()

        for key in mapa.map_data.keys():  # verifica qual menos posição relativa
            if key[0] < min_x:  # para fazer uma alteração em relação as coordenadas do mapa
                min_x = key[0]
            if key[1] < min_y:
                min_y = key[1]
                
        new_x = actual_x - min_x
        new_y = actual_y - min_y
        obj_x = obj_x - min_x
        obj_y = obj_y - min_y

        for key in mapa.map_data.keys():  # altera posição do mapa relativo
            new_k0 = key[0] - min_x
            new_k1 = key[1] - min_y
            abs_map[(new_k0, new_k1)] = mapa.map_data[key]
        # del abs_map[key]               #NOVO

        for key in abs_map.keys():  # adquire maior indices para criar matriz de posições
            if key[0] > max_x:
                max_x = key[0]
            if key[1] > max_y:
                max_y = key[1]

        tam_maze = max(max_x, max_y) + 1  # tamanho matriz geral
        maze_matrix = np.full((tam_maze, tam_maze), 100.0)  # preenche matriz com dificuldade 100
        for i in abs_map.keys():
            maze_matrix[i[1]][i[0]] = abs_map.get(i)[0]

        path, time_to_base = test_astar.solve_comeback(new_x, new_y, maze_matrix, obj_x, obj_y, 1.0, 1.5,
                                                       rescue=True)  # função A*

        for i in range(0, len(path.items)):
            path.items[i] = (path.items[i][0] + min_x, path.items[i][
                1] + min_y)  # retorna posições absoluta matriz para relativa do mapa do agente

        return path, time_to_base



