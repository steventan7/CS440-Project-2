'''
Implementation for Bot7
@author Yashas Ravi
'''
from __future__ import division
import random
from collections import deque
import math

DIRECTIONS = [0, 1, 0, -1, 0]
D = 10
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


# Performs a BFS implementation that returns the path starting from the bot's current location to the button
def bfs(ship, start_x, start_y, goal1):
    fringe = deque([(start_x, start_y)])
    closed_set = set()
    previous_state = {(start_x, start_y): None}

    while fringe:
        curr_x, curr_y = fringe.popleft()
        if (curr_x, curr_y) == goal1:
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


def bfs_leak(ship, start_x, start_y, goal1, goal2):
    fringe = deque([(start_x, start_y)])
    closed_set = set()
    previous_state = {(start_x, start_y): None}

    while fringe:
        curr_x, curr_y = fringe.popleft()
        if (curr_x, curr_y) == goal1 or (curr_x, curr_y) == goal2:
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


def probIsBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, prob_array):
    d_steps = len(bfs(ship, bot_x, bot_y, (cellx, celly)))

    prob_leak_in_j = prob_array[cellx][celly]
    prob_beep_in_a_given_leak_in_j = (math.e) ** ((-1) * alpha * (d_steps - 1))
    prob_beep_in_i = 0.0

    for (i, j) in potential_leaks:
        d_steps = len(bfs(ship, bot_x, bot_y, (i, j)))
        prob_beep_in_i += prob_array[i][j] * (math.e ** ((-1) * alpha * (d_steps - 1)))

    prob = (prob_leak_in_j * prob_beep_in_a_given_leak_in_j) / prob_beep_in_i
    return prob


def probNoBeep(ship, bot_x, bot_y, cell_x, cell_y, potential_leaks, prob_array):
    d_steps = len(bfs(ship, bot_x, bot_y, (cell_x, cell_y)))

    prob_leak_in_j = prob_array[cell_x][cell_y]
    prob_not_beep_in_a_given_leak_in_j = (1 - (math.e ** ((-1) * alpha * (d_steps - 1))))
    prob_no_beep_in_i = 0.0

    for (i, j) in potential_leaks:
        d_steps = len(bfs(ship, bot_x, bot_y, (i, j)))
        prob_no_beep_in_i += prob_array[i][j] * (1 - (math.e ** ((-1) * alpha * (d_steps - 1))))

    prob = (prob_leak_in_j * prob_not_beep_in_a_given_leak_in_j) / prob_no_beep_in_i
    return prob


def updateProb(curr_x, curr_y, prob_array, potential_leaks):
    num = prob_array[curr_x][curr_y]
    dem = 0.0
    
    for (i,j) in potential_leaks:
        dem += prob_array[i][j]
    return num/dem


def detect(ship, curr_x, curr_y, leak1, leak2, potential_leaks, prob_array):
    d_steps = len(bfs_leak(ship, curr_x, curr_y, leak1, leak2))
    prob_beep = (math.e) ** ((-1) * alpha * (d_steps - 1))
    num, beep = random.uniform(0, 1), False

    if num <= prob_beep:
        beep = True
    prob_array_sample = [[0 for i in range(D)] for j in range(D)]

    if beep:
        for (nx, ny) in potential_leaks:
            prob_array_sample[nx][ny] = probIsBeep(ship, curr_x, curr_y, nx, ny, potential_leaks, prob_array)
            if prob_array_sample[nx][ny] == 0:
                potential_leaks.remove((nx, ny))
    else:

        for (nx, ny) in potential_leaks:
            prob_array_sample[nx][ny] = probNoBeep(ship, curr_x, curr_y, nx, ny, potential_leaks, prob_array)
            if prob_array_sample[nx][ny] == 0:
                potential_leaks.remove((nx, ny))

    for (i, j) in potential_leaks:
        prob_array[i][j] = prob_array_sample[i][j]


'''
'''
def move(ship, curr_x, curr_y, leak1, leak2, potential_leaks, prob_array):
    if (curr_x, curr_y) == leak1 or (curr_x, curr_y) == leak2:
        return 0
        
    num_moves = 0
    while True:
        if (curr_x, curr_y) == leak1 or (curr_x, curr_y) == leak2:
            return num_moves
        else:
            if prob_array[curr_x][curr_y] != 0:
                potential_leaks.remove((curr_x, curr_y))
                
                prob_array[curr_x][curr_y] = 0
                prob_array_sample_two = [[0 for i in range(D)] for j in range(D)]

                for (nx,ny) in potential_leaks:
                    prob_array_sample_two[nx][ny] = updateProb(nx, ny, prob_array, potential_leaks)

                for (i,j) in potential_leaks:
                     prob_array[i][j] = prob_array_sample_two[i][j]

        prob_array[curr_x][curr_y] = 0
        detect(ship, curr_x, curr_y, leak1, leak2, potential_leaks, prob_array)
        num_moves += 1
        _max, endX, endY = 0, 0, 0

        for i in range(D):
            for j in range(D):
                if prob_array[i][j] > _max:
                    _max = prob_array[i][j]
                    endX = i
                    endY = j

        path = bfs(ship, curr_x, curr_y, (endX,endY))

        for (cell_x, cell_y) in path:
            if curr_x == cell_x and curr_y == cell_y:
                continue
            probArraySample3 = [[0 for i in range(D)] for j in range(D)]

            if (cell_x, cell_y) == leak1 or (cell_x, cell_y) == leak2:
                return num_moves
            else:
                if prob_array[cell_x][cell_y] != 0:
                    prob_array[cell_x][cell_y] = 0
                    potential_leaks.remove((cell_x, cell_y))
                    curr_x = cell_x
                    curr_y = cell_y
                    for(i, j) in potential_leaks:
                        probArraySample3[i][j] = updateProb(i, j, prob_array, potential_leaks)
                    for i in range(D):
                        for j in range(D):
                            prob_array[i][j] = probArraySample3[i][j]
            num_moves = num_moves + 1


def run_bot7():
    ship = [[1 for i in range(D)] for j in range(D)]
    start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)
    ship[start_x][start_y], open_cells = 0, set()
    blocked_one_window_cells = {(start_x, start_y)}
    create_ship(ship, blocked_one_window_cells, open_cells)
    K = (D // 2) - 1

    open_cells.remove((start_x, start_y))
    prob_array = [[0 for i in range(D)] for j in range(D)]
    equal_prob = 1 / len(open_cells)
    
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                prob_array[i][j] = equal_prob
    
    prob_array[start_x][start_y] = 0
    potential_leaks = open_cells.copy()
    leak_cell1, leak_cell2 = random.choice(list(potential_leaks)), random.choice(list(potential_leaks))
    num_moves = move(ship, start_x, start_y, leak_cell1, leak_cell2, potential_leaks, prob_array)

    return num_moves


if __name__ == '__main__':
    total_moves = 0
    for i in range(100):
        print("Trial: ", i)
        print(run_bot7())
