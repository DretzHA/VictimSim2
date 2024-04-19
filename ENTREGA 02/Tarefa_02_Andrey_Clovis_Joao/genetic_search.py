import test_astar
import random
import numpy as np

cost_matrix = []


def planner_genetic_algorithm(map, victims, tlim):

    global cost_matrix

    p = 50  # tamanho da população
    pCROSS = 0.8
    pMUT = 0.05

    victims_list = [[i[0]] + [i[1][6]] for i in list(victims.values())]
    print(victims_list)

    cost_matrix = np.full((len(victims_list),len(victims_list)), None).tolist()

    population = generate_population(len(victims_list), p)
    #[print(i) for i in population]

    #total_fits = sum(fits)
    #probas = [fit/total_fits for fit in fits]

    generation = 0

    while True:

        if generation > 60:
            break

        #print(f'Generation {generation}')

        fits = []
        costs = []

        population = crossover(population, pCROSS)

        population = mutation(population, pMUT)

        for individual in population:
            total_time, sequence = calculate_cost(individual, map, victims_list, tlim)
            costs.append(total_time)
            fit = fitness_function(sequence, victims_list, generation)
            fits.append(fit)

        #print(f'Mean fit: {sum(fits)/len(fits)}\nMaximum fit: {max(fits)}')

        zipped_individuals = zip(fits, costs, population)
        zipped_individuals = sorted(zipped_individuals, key=lambda x: x[0], reverse=True)

        fits, costs, population = zip(*zipped_individuals)
        fits = list(fits[:len(fits)//2])
        costs = list(costs[:len(costs) // 2])
        population = list(population[:len(population)//2])

        generation += 1

    max_value = max(fits)
    max_fits = [i for i, val in enumerate(fits) if val == max_value]
    max_fits_costs = [costs[index] for index in max_fits]
    min_cost_index = costs.index(min(max_fits_costs))

    print(f'Best sequence with fit {fits[min_cost_index]} and cost {costs[min_cost_index]} '
          f'is\n {population[min_cost_index]}')

    victims_coordinates = [victims_list[i] for i in population[min_cost_index]]
    print(victims_coordinates)
    victims_coordinates = [i[0] for i in victims_coordinates]

    plan = detail_rescue_plan(victims_coordinates, map, tlim)

    plan = plan.items[::-1]

    directions = []

    for i in range(len(plan)-1):

        if plan[i+1] in victims_coordinates:
            directions.append((plan[i+1][0] - plan[i][0], plan[i+1][1] - plan[i][1], True))
        else:
            directions.append((plan[i+1][0] - plan[i][0], plan[i+1][1] - plan[i][1], False))

    return directions


def fitness_function(sequence, victims_list, generation):

    Vs1 = Vs2 = Vs3 = Vs4 = 0

    if generation < 10:
        sequence = sequence[:len(sequence)//4]
    elif generation < 30:
        sequence = sequence[:len(sequence) // 2]
    else:
        sequence = sequence[:3*len(sequence) // 4]

    for i in sequence:
        if victims_list[i][1] == 1:
            Vs1 += 1
        elif victims_list[i][1] == 2:
            Vs2 += 1
        elif victims_list[i][1] == 3:
            Vs3 += 1
        elif victims_list[i][1] == 4:
            Vs4 += 1

    return 6*Vs1 + 3*Vs2 + 2*Vs3 + Vs4


def selection(probas):
    r = random.random()
    sum = 0
    i = 0
    for proba in probas:
        sum += proba
        if sum > r:
            return i
        i += 1


def mutation(population, pMUT):

    for individual in population:

        if random.random() > pMUT:
            continue

        idx1, idx2 = random.sample(range(0, len(individual)), 2)

        temp1 = individual[idx1]
        temp2 = individual[idx2]

        individual[idx1] = temp2
        individual[idx2] = temp1

    return population


def crossover(population, pCROSS):

    random.shuffle(population)

    for i in range(0, len(population) - 1, 2):

        list1 = population[i].copy()
        list2 = population[i + 1].copy()

        if random.random() > pCROSS:
            population.append(list1)
            population.append(list2)
            continue  # Skip crossover for this pair

        min_len = min(len(list1), len(list2))

        # Choose random start index and length for the crossover
        start = random.randint(0, min_len - 1)
        length = random.randint(1, min_len - start)

        # Perform the crossover
        temp1 = list1[start:start + length]
        temp2 = list2[start:start + length]
        list1[start:start + length] = temp2
        list2[start:start + length] = temp1

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

        population.append(list1)
        population.append(list2)

    return population


def generate_population(n_alleles, n_individuals):

    return [list(np.random.permutation(n_alleles)) for i in range(n_individuals)]


def manhattan_distance(x_start, y_start, x_end, y_end):
    return abs(x_end - x_start) + abs(y_end - y_start)


def calculate_cost(sequence, map, victims_list, tlim):
    #cost = astar_method(map, 0, 0, victims_list[sequence[0]][0][0], victims_list[sequence[0]][0][1]) + 1
    cost = 1 + manhattan_distance(0, 0, victims_list[sequence[0]][0][0], victims_list[sequence[0]][0][1])
    for i in range(len(sequence)-1):
        try:
            marginal_cost = cost_matrix[sequence[i]][sequence[i+1]] + 1
            cost += marginal_cost # +1 por conta do tempo do kit de primeiros socorros
        except TypeError:
            #marginal_cost = astar_method(map,
                                         #victims_list[sequence[i]][0][0], victims_list[sequence[i]][0][1],
                                         #victims_list[sequence[i+1]][0][0], victims_list[sequence[i+1]][0][1])
            marginal_cost = manhattan_distance(victims_list[sequence[i]][0][0], victims_list[sequence[i]][0][1],
                                               victims_list[sequence[i+1]][0][0], victims_list[sequence[i+1]][0][1])
            cost_matrix[sequence[i]][sequence[i+1]] = marginal_cost
            cost_matrix[sequence[i+1]][sequence[i]] = marginal_cost
            cost += (marginal_cost + 1)

    #cost += astar_method(map, victims_list[sequence[-1]][0][0], victims_list[sequence[-1]][0][1], 0, 0)
    cost += manhattan_distance(victims_list[sequence[-1]][0][0], victims_list[sequence[-1]][0][1], 0, 0)
    return cost, sequence


def detail_rescue_plan(victims_coordinates, map, TLIM):

    total_time = 0
    path, marginal_time = astar_method(map, 0, 0, victims_coordinates[0][0], victims_coordinates[0][1])
    total_time += marginal_time + 1
    
    
    for i, victim in enumerate(victims_coordinates[:-1]):
        marginal_path, marginal_time = astar_method(map,
                                                    victims_coordinates[i][0], victims_coordinates[i][1],
                                                    victims_coordinates[i+1][0], victims_coordinates[i+1][1])

        for j in range(2, len(marginal_path.items)+1):
            path.items.insert(0, marginal_path.items[-j])

        total_time += marginal_time + 1

        if total_time >= 0.9*TLIM:
            print("calculando retorno")
            marginal_path, marginal_time = astar_method(map, victims_coordinates[i+1][0], victims_coordinates[i+1][1],
                                                        0, 0)

            for j in range(2, len(marginal_path.items) + 1):
                path.items.insert(0, marginal_path.items[-j])

            total_time += marginal_time

            print(f"Full path WON'T be executed. Returning partial path with total time of {total_time}...")
            return path

    marginal_path, marginal_time = astar_method(map, victims_coordinates[-1][0], victims_coordinates[-1][1], 0, 0)

    for i in range(2, len(marginal_path.items) + 1):
        path.items.insert(0, marginal_path.items[-i])

    total_time += marginal_time
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



