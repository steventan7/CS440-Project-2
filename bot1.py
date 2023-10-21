'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
import random
from collections import deque

#Goal of Bot1 is to find the leak
#In order to win, Bot1 must make sure to enter a cell that is considered "leaked"

DIRECTIONS = [0, 1, 0, -1, 0]
D = 50#Set dimensions to 50

def has_one_open_neighbor(ship, r, c):#This is basically saying that the cell has exactly one open neighbor
    num_open_neighbors = 0
    for i in range(4):
        nx, ny = DIRECTIONS[i] + r, DIRECTIONS[i + 1] + c
        if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] == 1:
            continue
        num_open_neighbors += 1
    return num_open_neighbors <= 1

# Finds all the deadends located in the ship
def find_deadends(ship, open_cells, deadends):#Basically if an open cell has one neighbor, those are considered deadends and basically adds that to deadends set
    for curr_x, curr_y in open_cells:
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or not has_one_open_neighbor(ship, nx, ny):
                continue
            deadends.add((nx, ny))


# Creates the ship with logic in correspondence to the assignment write-up
def create_ship(ship, blocked_one_window_cells, open_cells):
    while blocked_one_window_cells:
        curr_x, curr_y = random.choice(list(blocked_one_window_cells))#get a random point from blocked cells
        blocked_one_window_cells.remove((curr_x, curr_y))#remove it
        if not has_one_open_neighbor(ship, curr_x, curr_y):#if it doesn't have one open neighbor then iterate through the next thing
            continue
        ship[curr_x][curr_y] = 0
        open_cells.add((curr_x, curr_y))#Consider this point an open cell if it does have exactly one open neighbor
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] == 0:
                continue
            if has_one_open_neighbor(ship, nx, ny):
                blocked_one_window_cells.add((nx, ny))#Get the next blocked cell that has one open neighbor


    deadends = set()
    find_deadends(ship, open_cells, deadends)

    length = len(deadends)
    for i in range(length // 2):
        deadend_x, deadend_y = random.choice(list(deadends))#deadendx is 
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


def search(ship, curr_x, curr_y, goal, leakSet):
    k = 0
        
    while k % 2 == 0 or k < 1:#makes sure that k is odd
        k =random.randint(0,D-1)#0-49

    temp_x = curr_x-k#-1
    length = 2*k+1


    for i in range(length):
        temp_y = curr_y-k#-1

        for j in range(length):
            if ship[temp_x][temp_y] == None or ship[temp_x][temp_y] == 1:#make sure that it is not out of bounds or closed
                temp_y += 1

            elif (temp_x, temp_y) == goal:
                return leakSet

            else:
                leakSet.add((temp_x, temp_y))
                temp_y += 1

        temp_x = temp_x+1

    leakSet.clear()#clear the leakset because we did not get the goal
    return leakSet

def move(ship, start_x, start_y, goal):
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
        
        leakset = set()

        leakset = search(ship, curr_x, curr_y, goal, leakset)

        if len(leakset) != 0:#if we realize that hey there is a cell in a (2k+1) * (2k+1) square that might have the leak
            for i in range(4):

                nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
                if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set or (nx, ny) not in leakset:
                    continue
                fringe.append((nx, ny))
                previous_state[(nx, ny)] = (curr_x, curr_y)
                closed_set.add((nx, ny))
            closed_set.add((curr_x, curr_y))
                
            

        else:#if we don't recognize any cell that has leak, then just keep doing the normal thing that Bot1 would do in project1
            for i in range(4):
                nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
                if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set:
                    continue
                fringe.append((nx, ny))
                previous_state[(nx, ny)] = (curr_x, curr_y)
                closed_set.add((nx, ny))
            closed_set.add((curr_x, curr_y))
    return None


    

   


def main():
    ship = [[1 for i in range(D)] for j in range(D)]
    start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)   # start coordinates for the bot
    ship[start_x][start_y], open_cells = 0, set()
    blocked_one_window_cells = {(start_x, start_y)}
    leakset = set()

    create_ship(ship, blocked_one_window_cells, open_cells)

    leakcell = random.choice(list(open_cells))# start coordinates for the leak

    path = move(ship, start_x, start_y, leakcell, leakset)

    
    print(path)

    




if __name__ == '__main__':
    main()
