'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
import random
from collections import deque
import math
# from colorama import init, Back, Style
# init(autoreset=True)

DIRECTIONS = [0, 1, 0, -1, 0]
D = 5


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

            
            return path.__len__
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set:
                continue
            fringe.append((nx, ny))
            previous_state[(nx, ny)] = (curr_x, curr_y)
            closed_set.add((nx, ny))
        closed_set.add((curr_x, curr_y))
    return None

def probIsBeep(ship, curr_x, curr_y, potential_leaks, leak):#probability that leak in j given that beep in cell i
    fixedAlpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    dsteps = bfs(ship, curr_x, curr_y, leak)#dsteps
    probLeakinJ = 1/potential_leaks.__len__
    probbeepinigivenleakinj= math.e ** -1*fixedAlpha*(dsteps-1)
    
    probbeepini = 0
    
    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                probbeepini += math.e ** -1*fixedAlpha*(dsteps-1)
    
    probbeepini = probbeepini / potential_leaks
    prob = probLeakinJ*probbeepinigivenleakinj / probbeepini
    return prob

def probNoBeep(ship, curr_x, curr_y, potential_leaks, leak):#probability that leak in j given that no beep in cell i
    fixedAlpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    dsteps = bfs(ship, curr_x, curr_y, leak)#dsteps
    
    probLeakinJ = 1/potential_leaks.__len__
    probnotBeepinigivenleakincellj = 1-(math.e ** -1*fixedAlpha*(dsteps-1))
    
    probnoBeepini = 0
    
    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                probnoBeepini += 1-(math.e ** -1*fixedAlpha*(dsteps-1))
    
    probnoBeepini = probnoBeepini / potential_leaks
    prob = probLeakinJ*probnotBeepinigivenleakincellj / probnoBeepini
    return prob
    

    

def detect(ship, curr_x, curr_y, leak, potential_leaks, K, probArray):
    fixedAlpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    
    dsteps = bfs(ship, curr_x, curr_y, leak)#dsteps
    
    probBeep = math.e ** (fixedAlpha*(dsteps-1))#probability for beep
    
    num = random.randint(0,1)
    
    beep = False
    
    if num <= probBeep:
        beep = True
    
    if beep == True:#Probability of Leak Given that there is a beep
        for (nx, ny) in potential_leaks:
            probArray[nx][ny] = probIsBeep(ship, nx,ny,potential_leaks, leak)
            
            
    else:#Probability of Leak Given that there is not a beep
        for (nx, ny) in potential_leaks:
            probArray[nx][ny] = probNoBeep(ship, nx, ny, potential_leaks, leak)

    


def move(ship, curr_x, curr_y, leak, potential_leaks, probArray):
    
    if (curr_x, curr_y) == leak:#if your starting point is at the ending point
        return 0
    
    
    num_moves = 0
    visited_cells = set((curr_x, curr_y))
        
    while True: 
        detect(curr_x, curr_y, leak, potential_leaks, probArray)
        tempSet = set()
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i+1] + curr_y
            
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0:
                continue
            
            tempSet.add(probArray[nx][ny])
        
        _max = -1
        
        
        for (nx,ny) in tempSet:
            if probArray[nx][ny] > _max:
                _max = probArray[nx][ny]
                curr_x = nx
                curr_y = ny
                
        num_moves = num_moves + 1

def run_bot1():
    # ship = [[1 for i in range(D)] for j in range(D)]
    # start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)   # start coordinates for the bot
    # ship[start_x][start_y], open_cells = 0, set()
    # blocked_one_window_cells = {(start_x, start_y)}
    # create_ship(ship, blocked_one_window_cells, open_cells)
    ship = [[1,1,0,0,0], [0,0,0,0,0], [0,1,0,0,1], [0,0,0,0,0], [0,0,0,0,0]]
    start_x, start_y = 3, 3
    open_cells = set()
    K = (D // 2) - 1
    
    probArray = [[1 for i in range(D)] for j in range(D)]#probability array
    equalProb = 1.0 / open_cells.__len__#1 / open cells length is what all cells will have in the beginning
    
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:#if it is open cell
                probArray[i][j] = equalProb

    for i in range(D):
        print(ship[i])
    print()
    leak_cell = (0, 2)
    # leak_cell = random.choice(list(open_cells))
    potential_leaks = open_cells.copy()
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                potential_leaks.add((i, j))
    num_moves = move(ship, start_x, start_y, leak_cell, potential_leaks, K)

    return num_moves


if __name__ == '__main__':
    total_moves = 0
    print(run_bot1())
