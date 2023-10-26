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
    else:
        for coordinate in cells_detected:
            potential_leaks.remove(coordinate)
    return potential_leaks, found_leak


def move(ship, curr_x, curr_y, leak, potential_leaks, K):
    num_moves = 0
    visited_cells = set()
    curr_path = [(curr_x, curr_y)]
    previous_state = {(curr_x, curr_y): None}
    while True:
        potential_leaks, leak_detected = detect(ship, curr_x, curr_y, leak, potential_leaks, K)
        num_moves += 1

        print(curr_x, curr_y)
        # check = input()
        potential_moves = set()

        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y

            if leak_detected:
                if nx <= -1 or nx >= D or ny <= -1 or ny >= D or ship[nx][ny] != 0 or (
                        (nx, ny) in visited_cells or (nx, ny) not in potential_leaks or (nx, ny) in curr_path):
                    continue

                if (nx, ny) == leak:
                    return num_moves
                potential_moves.add((nx, ny))
            else:
                if nx <= -1 or nx >= D or ny <= -1 or ny >= D or ship[nx][ny] != 0 or (nx, ny) in visited_cells or (
                        (nx, ny) in curr_path):
                    continue
                potential_moves.add((nx, ny))

        if not potential_moves:
            visited_cells.add((curr_x, curr_y))
            curr_x, curr_y = previous_state[(curr_x, curr_y)]
        else:
            next_x, next_y = random.choice(list(potential_moves))
            previous_state[(next_x, next_y)] = (curr_x, curr_y)
            curr_x, curr_y = next_x, next_y
        num_moves += 1


    return 0


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
    potential_leaks = set()
    # potential_leaks = open_cells.copy()
    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                potential_leaks.add((i, j))
    num_moves = move(ship, start_x, start_y, leak_cell, potential_leaks, K)
    return num_moves


if __name__ == '__main__':
    total_moves = 0
    print(run_bot1())
