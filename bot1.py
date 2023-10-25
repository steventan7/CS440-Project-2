'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
import random
from collections import deque
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


def detect(ship, curr_x, curr_y, leak, potential_leaks, K):
    found_leak = False
    cells_detected = set()
    for r in range(curr_x - K, curr_x + K + 1):
        for c in range(curr_y - K, curr_y + K + 1):
            if r <= -1 or r >= D or c <= -1 or c >= D or ship[r][c] == 1 or (r, c) not in potential_leaks:
                continue

            if (r, c) == leak:
                found_leak = True
            cells_detected.add((r, c))

    if found_leak:
        potential_leaks.clear()
        potential_leaks.update(cells_detected)
        print(potential_leaks)
    else:
        for coordinate in cells_detected:
            potential_leaks.remove(coordinate)
    return potential_leaks, found_leak


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


def move(ship, curr_x, curr_y, leak, potential_leaks, K):
    
    num_moves = 0
    
    visited_cells = set((curr_x, curr_y))
    
    
    justincase_cells = set()
    
    while True: 
        #print(curr_x, curr_y)
        potential_leaks, found_leak = detect(ship, curr_x, curr_y, leak, potential_leaks, K)
        ctr = 0


        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i+1] + curr_y
            #print("nx ny", (nx,ny))
            
            if found_leak:
                #print("hello")
                if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in visited_cells or (nx, ny) not in potential_leaks:
                    continue
                
                num_moves = num_moves + 1
                curr_x = nx
                curr_y = ny
                
                print(curr_x, curr_y)
                
                if (curr_x, curr_y) == leak:
                    return num_moves
                
                visited_cells.add((curr_x, curr_y))
                break
                
            else: 
                #print("Hello")
                ctr = ctr + 1
                justincase_cells.add((nx,ny))
                
                if ctr == 4:
                    # print("Just in case cells", justincase_cells)
                    # print("hello")
                    #print("Just in case cells:", justincase_cells)
                    for cellx, celly in justincase_cells:
                        if ship[cellx][celly] == 0:
                            curr_x = cellx
                            curr_y = celly
                            num_moves += 1
                            print((curr_x, curr_y))
                            visited_cells.add((cellx, celly))
                            justincase_cells.clear()
                            #print("Hello")
                            break
                    break
                
                if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in visited_cells:
                    continue
                
                num_moves = num_moves + 1
                curr_x = nx#update curr x 
                curr_y = ny#update curr y

                visited_cells.add((curr_x, curr_y))
                justincase_cells.clear()
                print(curr_x, curr_y)

                break

                
                # bfs(ship, curr_x, curr_y, leak)

            
    # leakset = search(ship, curr_x, curr_y, goal, leakset, K)
    #
    # if len(leakset) != 0:#if we realize that hey there is a cell in a (2k+1) * (2k+1) square that might have the leak
    #     for i in range(4):
    #
    #         nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
    #         if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set or (nx, ny) not in leakset:
    #             continue
    #         fringe.append((nx, ny))
    #         previous_state[(nx, ny)] = (curr_x, curr_y)
    #         closed_set.add((nx, ny))
    #     closed_set.add((curr_x, curr_y))
    # else:#if we don't recognize any cell that has leak, then just keep doing the normal thing that Bot1 would do in project1
    #     for i in range(4):
    #         nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
    #         if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set:
    #             continue
    #         fringe.append((nx, ny))
    #         previous_state[(nx, ny)] = (curr_x, curr_y)
    #         closed_set.add((nx, ny))
    #     closed_set.add((curr_x, curr_y))
    # return None


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

    for i in range(D):
        print(ship[i])
    print()
    leak_cell = (0,4)
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
