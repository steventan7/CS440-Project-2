'''
Implementation for Bot4
@author Yashas Ravi
'''
from __future__ import division
import random
from collections import deque
import math
import heapq
# from colorama import init, Back, Style
# init(autoreset=True)

DIRECTIONS = [0, 1, 0, -1, 0]
D = 20
alpha = 0.4


# This function checks if a cell has exactly one open neighbor
def has_one_open_neighbor(ship, r, c):
    num_open_neighbors = 0
    for i in range(4):
        nx, ny = DIRECTIONS[i] + r, DIRECTIONS[i + 1] + c
        if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] == 1:
            continue
        num_open_neighbors += 1
    return num_open_neighbors <= 1


# Finds all the deadends located in the ship
def find_deadends(ship, open_cells, deadends):
    for curr_x, curr_y in open_cells:
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or not has_one_open_neighbor(ship, nx, ny):
                continue
            deadends.add((nx, ny))


# Creates the ship with logic in correspondence to the assignment write-up
def create_ship(ship, blocked_one_window_cells, open_cells):
    while blocked_one_window_cells:
        curr_x, curr_y = random.choice(list(blocked_one_window_cells))
        blocked_one_window_cells.remove((curr_x, curr_y))
        if not has_one_open_neighbor(ship, curr_x, curr_y):
            continue
        ship[curr_x][curr_y] = 0
        open_cells.add((curr_x, curr_y))
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] == 0:
                continue
            if has_one_open_neighbor(ship, nx, ny):
                blocked_one_window_cells.add((nx, ny))

    deadends = set()
    find_deadends(ship, open_cells, deadends)

    length = len(deadends)
    for i in range(length // 2):
        deadend_x, deadend_y = random.choice(list(deadends))
        blocked_cells = set()
        for i in range(4):
            nx, ny = DIRECTIONS[i] + deadend_x, DIRECTIONS[i + 1] + deadend_y
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 1:
                continue
            blocked_cells.add((nx, ny))
        if blocked_cells:
            open_x, open_y = random.choice(list(blocked_cells))
            ship[open_x][open_y] = 0
            open_cells.add((open_x, open_y))
        deadends.remove((deadend_x, deadend_y))


'''
A* search to make a path that avoids low-probability cells when determining a path from bot to the cell we are 
considering. This will affect num_moves since a more optimized path will effect the number of cells in the path 
from the bot to the cell under consideration.
'''
def astar(ship, start_x, start_y, goal, probArray):
    pq = []

    bot_position = (start_x, start_y)

    distTo = {bot_position: 0}

    heapq.heappush(pq, (distTo[bot_position], bot_position))
    processed_cells = set()

    parent = {(start_x, start_y): None}

    while pq:
        cell_priority, curr_cell = heapq.heappop(pq)
        (curr_x, curr_y) = curr_cell

        if curr_cell == goal:
            path, coord = [], (curr_x, curr_y)
            while coord != None:
                path.append(coord)
                coord = parent[coord]
            return path[::-1]

        if curr_cell not in processed_cells:
            for i in range(4):
                nx = DIRECTIONS[i] + curr_x
                ny = DIRECTIONS[i + 1] + curr_y

                neighbor = (nx, ny)
                if nx in [-1, D] or ny in [-1, D]:
                    continue
                if ship[nx][ny] != 0 or (nx, ny) in processed_cells:
                    continue

                temp_dist = distTo[curr_cell] + 1

                if neighbor not in distTo or temp_dist < distTo[neighbor]:
                    distTo[neighbor] = temp_dist
                    parent[neighbor] = curr_cell
                    prob_factor = D*probArray[nx][ny]
                    heapq.heappush(pq, (distTo[neighbor] - prob_factor, neighbor))
            processed_cells.add(curr_cell)
    return None


# Performs a BFS implementation that returns the path starting from the bot's current location to the button
def bfs(ship, start_x, start_y, goal):
    fringe = deque([(start_x, start_y)])
    closed_set = set()
    previous_state = {(start_x, start_y): None}

    while fringe:
        curr_x, curr_y = fringe.popleft()
        if (curr_x, curr_y) == goal:
            path, coord = [], (curr_x, curr_y)
            while coord != None:
                path.append(coord)
                coord = previous_state[coord]

            return path[::-1]
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set:
                continue
            fringe.append((nx, ny))
            previous_state[(nx, ny)] = (curr_x, curr_y)
            closed_set.add((nx, ny))
        closed_set.add((curr_x, curr_y))
    return None


# Calculates the probability of a beep from the bot's cell if a beep is heard
def probIsBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, leak, prob_Array):
    d_steps = len(bfs(ship, bot_x, bot_y, (cellx, celly)))

    prob_leak_in_j = prob_Array[cellx][celly]
    prob_beep_in_a_given_leak_in_j = (math.e) ** ((-1) * alpha * (d_steps - 1))
    prob_beep_in_i = 0.0

    for (i, j) in potential_leaks:
        d_steps = len(bfs(ship, bot_x, bot_y, (i, j)))
        prob_beep_in_i += prob_Array[i][j] * (math.e ** ((-1) * alpha * (d_steps - 1)))

    prob = (prob_leak_in_j * prob_beep_in_a_given_leak_in_j) / prob_beep_in_i
    return prob


# Calculates the probability of a beep from the bot's cell if a beep is not heard
def probNoBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, leak, prob_Array):
    d_steps = len(bfs(ship, bot_x, bot_y, (cellx, celly)))

    prob_leak_in_j = prob_Array[cellx][celly]
    prob_not__beep_in_a_given_leak_in_j = (1 - (math.e ** ((-1) * alpha * (d_steps - 1))))
    prob_no_beep_in_i = 0.0

    for (i, j) in potential_leaks:
        d_steps = len(bfs(ship, bot_x, bot_y, (i, j)))
        prob_no_beep_in_i += prob_Array[i][j] * (1 - (math.e ** ((-1) * alpha * (d_steps - 1))))

    prob = (prob_leak_in_j * prob_not__beep_in_a_given_leak_in_j) / prob_no_beep_in_i
    return prob


# Updates the probability board with the new calculations
def updateProb(ship, curr_x, curr_y, probArray, potential_leaks):
    num = probArray[curr_x][curr_y]
    dem = 0.0

    for (i, j) in potential_leaks:
        dem += probArray[i][j]
    return num / dem

'''
'''
def detect(ship, curr_x, curr_y, leak, potential_leaks, prob_array):
    d_steps = len(bfs(ship, curr_x, curr_y, leak))

    prob_beep = (math.e) ** ((-1) * alpha * (d_steps - 1))
    num, beep = random.uniform(0, 1), False

    if num <= prob_beep:
        beep = True
    prob_array_sample = [[0 for i in range(D)] for j in range(D)]

    if beep:
        for (nx, ny) in potential_leaks:
            prob_array_sample[nx][ny] = probIsBeep(ship, curr_x, curr_y, nx, ny, potential_leaks, leak, prob_array)
    else:
        count = 0
        for (nx, ny) in potential_leaks:
            print(count)
            prob_array_sample[nx][ny] = probNoBeep(ship, curr_x, curr_y, nx, ny, potential_leaks, leak, prob_array)
            count += 1

    for i in range(D):
        for j in range(D):
            if (i, j) in potential_leaks:
                prob_array[i][j] = prob_array_sample[i][j]

'''
'''
def move(ship, curr_x, curr_y, leak, potential_leaks, prob_array):
    if (curr_x, curr_y) == leak:
        return 0

    num_moves = 0
    while True:
        if (curr_x, curr_y) == leak:
            return num_moves

        else:
            if prob_array[curr_x][curr_y] != 0:
                potential_leaks.remove((curr_x, curr_y))

                prob_array[curr_x][curr_y] = 0
                prob_array_sample_two = [[0 for i in range(D)] for j in range(D)]

                for (nx, ny) in potential_leaks:
                    prob_array_sample_two[nx][ny] = updateProb(ship, nx, ny, prob_array, potential_leaks)

                for i in range(D):
                    for j in range(D):
                        if (i, j) in potential_leaks:
                            prob_array[i][j] = prob_array_sample_two[i][j]

        prob_array[curr_x][curr_y] = 0
        detect(ship, curr_x, curr_y, leak, potential_leaks, prob_array)
        num_moves += 1

        _max, endX, endY = 0, 0, 0

        for i in range(D):
            for j in range(D):
                if prob_array[i][j] > _max:
                    _max = prob_array[i][j]
                    endX = i
                    endY = j

        path = astar(ship, curr_x, curr_y, (endX,endY), prob_array)

        for (cellx, celly) in path:

            if curr_x == cellx and curr_y == celly:
                continue
            probArraySample3 = [[0 for i in range(D)] for j in range(D)]
            if (cellx, celly) == leak:
                return num_moves
            else:
                if prob_array[cellx][celly] != 0:
                    prob_array[cellx][celly] = 0
                    potential_leaks.remove((cellx, celly))
                    curr_x = cellx
                    curr_y = celly
                    for (i, j) in potential_leaks:
                        probArraySample3[i][j] = updateProb(ship, i, j, prob_array, potential_leaks)
                    for i in range(D):
                        for j in range(D):
                            prob_array[i][j] = probArraySample3[i][j]
            num_moves = num_moves + 1



# Prints out the probability array
def printProbArray(prob_array):
    _sum = 0
    for i in range(len(prob_array)):
        for j in range(len(prob_array)):
            print((i, j), prob_array[i][j])

    for i in range(len(prob_array)):
        for j in range(len(prob_array[i])):
            _sum += prob_array[i][j]
    return _sum
        

'''
Sets up the location of the robot's starting point, the button, the leak, and the ship itself. It first calls detect
on the current location of the bot and then randomly assigns the location of the leak, just to ensure that the location 
of the leak is not within the first sense action. It then runs the game for Bot 4. It returns the total number of moves 
computed to find where the leak is.
'''
def run_bot4():
    ship = [[1 for i in range(D)] for j in range(D)]
    start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)   # start coordinates for the bot
    ship[start_x][start_y], open_cells = 0, set()
    blocked_one_window_cells = {(start_x, start_y)}
    create_ship(ship, blocked_one_window_cells, open_cells)

    K = (D // 2) - 1

    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                open_cells.add((i, j))

    open_cells.remove((start_x, start_y))
    probArray = [[0 for i in range(D)] for j in range(D)]
    equalProb = 1 / len(open_cells)
    
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                probArray[i][j] = equalProb
    
    probArray[start_x][start_y] = 0         
    potential_leaks = open_cells.copy()
    leak_cell = random.choice(list(potential_leaks))
    num_moves = move(ship, start_x, start_y, leak_cell, potential_leaks, probArray)

    return num_moves


if __name__ == '__main__':
    total_moves = 0
    print(run_bot4())
