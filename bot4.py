'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
from __future__ import division
import random
from collections import deque
import math
import heapq
# from colorama import init, Back, Style
# init(autoreset=True)

DIRECTIONS = [0, 1, 0, -1, 0]
D = 5
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



def minStepsLeak(ship, curr_x, curr_y, leak):# to leak
    
    finalX = abs(leak[0] - curr_x)
    finalY = abs(leak[1]-curr_y)
    
    final = finalX + finalY
    return final

def minStepsBot(bot_x, bot_y, curr_x, curr_y):# to bot
     finalX = abs(bot_x - curr_x)
     finalY = abs(bot_y - curr_y)
     
     final = finalX + finalY
     
     return final

# A* search to make a path that avoids low-probability cells when determining
# a path from bot to the cell we are considering. This will affect num_moves
# since a more optimized path will effect the number of cells in the path 
# from the bot to the cell under consideration.
def astar(ship, start_x, start_y, goal, probArray):
    # print(fire_tiles)
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

                    # x_fires, y_fires = 0, 0
                    # for fire_tile in fire_tiles:
                    #     x_fires += fire_tile[0]
                    #     y_fires += fire_tile[1]
                    # x_fires //= len(fire_tiles)
                    # y_fires //= len(fire_tiles)

                    # dist_button_to_neighbor = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    # dist_button_to_fire = abs(x_fires - goal[0]) + abs(y_fires - goal[1])
                    # heapq.heappush(pq, (distTo[neighbor] - dist_button_to_fire + dist_button_to_neighbor, neighbor))



            processed_cells.add(curr_cell)
    return None


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
    

def probIsBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, leak, probArray):#probability that leak in j given that beep in cell i
    # alpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    
    dsteps = minStepsBot(bot_x, bot_y, cellx, celly)#dsteps
    # print("Dsteps", dsteps)
    probLeakinJ = probArray[cellx][celly]
    probbeepinigivenleakinj = (math.e) ** ((-1)*alpha*(dsteps-1))
    probbeepini = 0.0
    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                dsteps = minStepsBot(bot_x, bot_y, i,j)
                probbeepini += probArray[i][j] * (math.e ** ((-1)*alpha*(dsteps-1)))
    
    prob = (probLeakinJ*probbeepinigivenleakinj) / probbeepini
    return prob

def probNoBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, leak, probArray):#probability that leak in j given that no beep in cell i
    # alpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    dsteps = minStepsBot(bot_x, bot_y, cellx, celly)#dsteps
    
    probLeakinJ = probArray[cellx][celly]
    probnotBeepinigivenleakincellj = (1-(math.e ** ((-1)*alpha*(dsteps-1))))
    
    
    probnoBeepini = 0.0
    
    
    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                dsteps = minStepsBot(bot_x, bot_y, i,j)
                probnoBeepini += probArray[i][j] * (1-(math.e ** ((-1)*alpha*(dsteps-1))))
    
    
    prob = (probLeakinJ*probnotBeepinigivenleakincellj) / probnoBeepini
    return prob
    
def updateProb(ship, curr_x, curr_y, probArray, potential_leaks):
    # print("Prob", probArray[curr_x][curr_y])
    num = probArray[curr_x][curr_y]
    dem = 0.0
    
    for (i,j) in potential_leaks:
        dem += probArray[i][j]
                
    # print("Denominator", dem)
    
   
    return num/dem

def detect(ship, curr_x, curr_y, leak, potential_leaks, K, probArray):
    #returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    
    dsteps = minStepsLeak(ship, curr_x, curr_y, leak)#dsteps
    # probBeep = 0.0
    # for i in range(D):
    #     for j in range(D):
    #         if (i,j) in potential_leaks:
    #             dsteps = minStepsBot(curr_x, curr_y, i,j)
    #             probBeep += probArray[i][j] * (math.e ** ((-1)*alpha*(dsteps-1)))
                
                
    probBeep = (math.e) ** ((-1)*alpha*(dsteps-1))#probability for beep
    num = random.uniform(0,1)
    
    beep = False
    
    if num <= probBeep:
        beep = True
        
    probArraySample = [[0 for i in range(D)] for j in range(D)]
    
    
    if beep:#Probability of Leak Given that there is a beep
        print("Beep")
        for (nx, ny) in potential_leaks:
            probArraySample[nx][ny] = probIsBeep(ship, curr_x, curr_y, nx,ny,potential_leaks, leak, probArray)
            
            
    else:#Probability of Leak Given that there is not a beep
        print("No beep")
        for (nx, ny) in potential_leaks:
            probArraySample[nx][ny] = probNoBeep(ship, curr_x, curr_y, nx, ny, potential_leaks, leak, probArray)

    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                probArray[i][j] = probArraySample[i][j]


def move(ship, curr_x, curr_y, leak, potential_leaks, K, probArray):
    if (curr_x, curr_y) == leak:#if your starting point is at the ending point
        return 0
    
    visited_cells = set((curr_x, curr_y))

        
    num_moves = 0
    while True:

        print("Currx, Curr_y: ",curr_x, curr_y)

        if(curr_x, curr_y) == leak:
            return num_moves
        
        else:
            if probArray[curr_x][curr_y] != 0:
                potential_leaks.remove((curr_x, curr_y))
                
                probArray[curr_x][curr_y] = 0#We do this because if we don't find the leak cell, we set it to 0
                # print(potential_leaks)
                
                #sum = 0.0
                #ctr = 0

                
                
                probArraySample2 = [[0 for i in range(D)] for j in range(D)]

                for (nx,ny) in potential_leaks:
                    probArraySample2[nx][ny] = updateProb(ship, nx, ny, probArray, potential_leaks)
                    
                    #sum += probArray[nx][ny]
                    #ctr = ctr + 1
                    # print(ctr)
                for i in range(D):
                    for j in range(D):
                         if (i,j) in potential_leaks:
                             probArray[i][j] = probArraySample2[i][j]
                             
                   
       
        sum = printProbArray(probArray)
        print("Sum for updating", sum)
        
        probArray[curr_x][curr_y] = 0
        detect(ship, curr_x, curr_y, leak, potential_leaks, K, probArray)
        num_moves += 1
        
        # sum = printProbArray(probArray)
        # print("Sum for updating", sum)
        
        sum = printProbArray(probArray)
        print("Sum for beep", sum)
        
        _max = 0
        
        endX = 0
        endY = 0
        
        for i in range(D):
            for j in range(D):
                if probArray[i][j] > _max:
                    _max = probArray[i][j]
                    endX = i
                    endY = j
                    
        

        # path = bfs(ship, curr_x, curr_y, (endX,endY))
        path = astar(ship, curr_x, curr_y, (endX,endY), probArray)
        
        print("Path: ", path)
        
        for (cellx,celly) in path:
            

            if curr_x == cellx and curr_y == celly:#ignore the beginning cell as we already updated
                continue
            
            probArraySample3 = [[0 for i in range(D)] for j in range(D)]

            
            if (cellx, celly) == leak:
                return num_moves
            
            else:

                if probArray[cellx][celly] != 0:
                    probArray[cellx][celly] = 0
                    potential_leaks.remove((cellx, celly))
                    curr_x = cellx
                    curr_y = celly
                    for(i,j) in potential_leaks:
                        probArraySample3[i][j] = updateProb(ship, i, j, probArray, potential_leaks)
            
                    for i in range(D):
                        for j in range(D):
                            probArray[i][j] = probArraySample3[i][j]
                    
            num_moves = num_moves + 1
                
                
        # tempSet = set()
        # for i in range(4):
        #     nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i+1] + curr_y
            
        #     if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0:
        #         continue
            
        #     tempSet.add((nx,ny))
        
        # _max = -1
        
        # isAll0 = True
        # for (nx,ny) in tempSet: 
        #     if probArray[nx][ny] != 0:
        #         isAll0 = False
        
        # # print("Before")
        # # printProbArray(probArray)
        
        # if isAll0:
        #     # print("Temp", tempSet)
        #     # print(isAll0)

        #     nx,ny = random.choice(list(tempSet))
        #     curr_x = nx
        #     curr_y = ny
        
            
        
        # else:   
        #     for (nx,ny) in tempSet:
        #         if probArray[nx][ny] > _max and (nx,ny) not in visited_cells:
        #             _max = probArray[nx][ny]
        #             curr_x = nx
        #             curr_y = ny
                
        # print("After")
        # printProbArray(probArray)

        
        
        

def printProbArray(probArray):
    _sum = 0
    # for i in range(len(probArray)):
    #     print(probArray[i])
        
    for i in range(len(probArray)):
        for j in range(len(probArray)):
            print((i,j), probArray[i][j])
    
    for i in range(len(probArray)):
        for j in range(len(probArray[i])):
            _sum += probArray[i][j]
            
    return _sum
        

def run_bot4():
    # ship = [[1 for i in range(D)] for j in range(D)]
    # start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)   # start coordinates for the bot
    # ship[start_x][start_y], open_cells = 0, set()
    # blocked_one_window_cells = {(start_x, start_y)}
    # create_ship(ship, blocked_one_window_cells, open_cells)
    ship = [[1,1,0,0,0], [0,0,0,0,0], [0,1,0,0,1], [0,0,0,0,0], [0,0,0,0,0]]
    start_x, start_y = 0,2
    open_cells = set()
    K = (D // 2) - 1
    
    
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                open_cells.add((i, j))
                

    open_cells.remove((start_x, start_y))
    probArray = [[0 for i in range(D)] for j in range(D)]#probability array
    equalProb = 1 / len(open_cells)#1 / open cells length is what all cells will have in the beginning
    
    #print("EqualProb", equalProb)
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:#if it is open cell and not equal to (start_x, start_y)
                probArray[i][j] = equalProb
    
    probArray[start_x][start_y] = 0         
    # print("Equal Prob: ", equalProb)
    for i in range(D):
        print(probArray[i])
    print()
    
    for i in range(D):
        print(ship[i])
    print()
    # leak_cell = (4,4)
    # leak_cell = random.choice(list(open_cells))
    potential_leaks = open_cells.copy()
    # print("Potential Leaks", potential_leaks)
    leak_cell = random.choice(list(potential_leaks))
    num_moves = move(ship, start_x, start_y, leak_cell, potential_leaks, K, probArray)

    return num_moves


if __name__ == '__main__':
    total_moves = 0
    print(run_bot4())
