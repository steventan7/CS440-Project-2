'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
from __future__ import division
import random
from collections import deque
import math
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


def minSteps(ship, curr_x, curr_y, leak):#DO NOT IGNORE BLOCKED CELLS
    
    finalX = abs(leak[0] - curr_x)
    finalY = abs(leak[1]-curr_y)
    
    final = finalX + finalY
    return final
    
    
    

def probIsBeep(ship, curr_x, curr_y, potential_leaks, leak, probArray):#probability that leak in j given that beep in cell i
    # alpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    dsteps = minSteps(ship, curr_x, curr_y, leak)#dsteps
    # print("Dsteps", dsteps)
    probLeakinJ = probArray[curr_x][curr_y]
    probbeepinigivenleakinj= math.e ** ((-1)*alpha*(dsteps-1))
    probbeepini = 0.0
    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                dsteps = minSteps(ship, i, j, leak)
                probbeepini += probArray[i][j] * (math.e ** ((-1)*alpha*(dsteps-1)))
    
    prob = probLeakinJ*probbeepinigivenleakinj / probbeepini
    return prob

def probNoBeep(ship, curr_x, curr_y, potential_leaks, leak, probArray):#probability that leak in j given that no beep in cell i
    # alpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    dsteps = minSteps(ship, curr_x, curr_y, leak)#dsteps
    
    probLeakinJ = probArray[curr_x][curr_y]
    probnotBeepinigivenleakincellj = 1-(math.e ** ((-1)*alpha*(dsteps-1)))
    
    probnoBeepini = 0.0
    
    for i in range(D):
        for j in range(D):
            if (i,j) in potential_leaks:
                dsteps = minSteps(ship, i, j, leak)
                probnoBeepini += probArray[i][j] * 1-(math.e ** ((-1)*alpha*(dsteps-1)))
    
    prob = probLeakinJ*probnotBeepinigivenleakincellj / probnoBeepini
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
    
    dsteps = minSteps(ship, curr_x, curr_y, leak)#dsteps
    
    probBeep = math.e ** ((-1)*alpha*(dsteps-1))#probability for beep
    num = random.uniform(0,1)
    
    beep = False
    
    if num <= probBeep:
        beep = True
    
    if beep:#Probability of Leak Given that there is a beep
        for (nx, ny) in potential_leaks:
            probArray[nx][ny] = probIsBeep(ship, nx,ny,potential_leaks, leak, probArray)
            
            
    else:#Probability of Leak Given that there is not a beep
        for (nx, ny) in potential_leaks:
            probArray[nx][ny] = probNoBeep(ship, nx, ny, potential_leaks, leak, probArray)

    


def move(ship, curr_x, curr_y, leak, potential_leaks, K, probArray):
    if (curr_x, curr_y) == leak:#if your starting point is at the ending point
        return 0
        
    num_moves = 0
    while True:

        print("Currx, Curr_y: ",curr_x, curr_y)

        if(curr_x, curr_y) == leak:
            return num_moves
        
        else:
            if probArray[curr_x][curr_y] != 0:
                potential_leaks.discard((curr_x, curr_y))
                
                probArray[curr_x][curr_y] = 0#We do this because if we don't find the leak cell, we set it to 0
                # print(potential_leaks)
                
                sum = 0.0
                ctr = 0
                for (nx,ny) in potential_leaks:
                    probArray[nx][ny] = updateProb(ship, nx, ny, probArray, potential_leaks)
                    sum += probArray[nx][ny]
                    ctr = ctr + 1
                    print(ctr)
                   
       
        sum = printProbArray(probArray)
        print("Sum for updating", sum)
        
        probArray[curr_x][curr_y] = 0
        detect(ship, curr_x, curr_y, leak, potential_leaks, K, probArray)
        num_moves += 1
        
        # sum = printProbArray(probArray)
        # print("Sum for updating", sum)
        
        sum = printProbArray(probArray)
        print("Sum for beep", sum)
        
        
        tempSet = set()
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i+1] + curr_y
            
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0:
                continue
            
            tempSet.add((nx,ny))
        
        _max = -1
        
        isAll0 = True
        for (nx,ny) in tempSet: 
            if probArray[nx][ny] != 0:
                isAll0 = False
        
        if isAll0: 
            # print("Temp", tempSet)
            # print(isAll0)

            nx,ny = random.choice(list(tempSet))
            curr_x = nx
            curr_y = ny
                
        else:   
            for (nx,ny) in tempSet:
                if probArray[nx][ny] > _max:
                    _max = probArray[nx][ny]
                    curr_x = nx
                    curr_y = ny
                
       
        num_moves = num_moves + 1
        
        

def printProbArray(probArray):
    _sum = 0
    for i in range(len(probArray)):
        print(probArray[i])
    
    for i in range(len(probArray)):
        for j in range(len(probArray[i])):
            _sum += probArray[i][j]
            
    return _sum
        

def run_bot1():
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
    leak_cell = (2,2)
    # leak_cell = random.choice(list(open_cells))
    potential_leaks = open_cells.copy()
    # print("Potential Leaks", potential_leaks)
    
    
    num_moves = move(ship, start_x, start_y, leak_cell, potential_leaks, K, probArray)

    return num_moves


if __name__ == '__main__':
    total_moves = 0
    print(run_bot1())
