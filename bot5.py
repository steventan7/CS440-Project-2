'''
Implementation for Bot5
@author Steven Tan
'''
import random
from colorama import init, Back, Style
from collections import deque, defaultdict
init(autoreset=True)

DIRECTIONS = [0, 1, 0, -1, 0]
D = 10


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
            return len(path)
        for i in range(4):
            nx, ny = DIRECTIONS[i] + curr_x, DIRECTIONS[i + 1] + curr_y
            if nx in [-1, D] or ny in [-1, D] or ship[nx][ny] != 0 or (nx, ny) in closed_set:
                continue
            fringe.append((nx, ny))
            previous_state[(nx, ny)] = (curr_x, curr_y)
            closed_set.add((nx, ny))
        closed_set.add((curr_x, curr_y))
    return 0


# Used to visualize the difference components of the game
def visualize_grid(ship, start, leak1, leak2, bot):
    for i in range(D):
        curr_row = ""
        for j in range(D):
            if (i, j) == start:
                curr_row += (Style.RESET_ALL + Back.BLUE + Style.BRIGHT + "S_")
            elif (i, j) == leak1:
                curr_row += (Style.RESET_ALL + Back.LIGHTBLACK_EX + Style.BRIGHT + "L1")
            elif(i, j) == leak2:
                curr_row += (Style.RESET_ALL + Back.LIGHTBLACK_EX + Style.BRIGHT + "L2")
            elif (i, j) == bot:
                curr_row += (Style.RESET_ALL + Back.GREEN + Style.BRIGHT + "B_")
            elif ship[i][j] == 0:
                curr_row += (Style.RESET_ALL + Back.YELLOW + "__")
            elif ship[i][j] == 1:
                curr_row += (Style.RESET_ALL + Back.WHITE + "__")
            else:
                curr_row += (Style.RESET_ALL + "__")
        print(curr_row)
    print()
    print()


# Determines whether a leak exists by iterating over a (2K + 1)(2K + 1) detection over the current location of the bot
def detect(ship, curr_x, curr_y, leak1, leak2, potential_leaks, K):
    found_leak = False
    cells_detected = set()

    for r in range(curr_x - K, curr_x + K + 1):
        if r <= -1 or r >= D:
            continue
        for c in range(curr_y - K, curr_y + K + 1):
            if c <= -1 or c >= D or ship[r][c] == 1 or (r, c) not in potential_leaks:
                continue

            if (r, c) == leak1 or (r, c) == leak2:
                found_leak = True
            cells_detected.add((r, c))

    if not found_leak:
        for coordinate in cells_detected:
            potential_leaks.remove(coordinate)
    return potential_leaks, found_leak


'''
If the sense action is performed and a leak is not found, then the cells outside the detection are checked to
determine the closest cells to the current location of the bot that potentially contain the leak
'''
def closest_moves_leak_not_found(ship, curr_x, curr_y, potential_leaks, K):
    curr_loop = 1
    distance_map = defaultdict(list)
    while not distance_map:
        for i in range(curr_y - K - curr_loop, curr_y + K + curr_loop + 1):
            if i <= -1 or i >= D:
                continue
            top, bottom = curr_x - K - curr_loop, curr_x + K + curr_loop
            if -1 < top < D and ship[top][i] == 0 and (top, i) in potential_leaks:
                distance_map[bfs(ship, top, i, (curr_x, curr_y))].append((top, i))
            if -1 < bottom < D and ship[bottom][i] == 0 and (bottom, i) in potential_leaks:
                distance_map[bfs(ship, bottom, i, (curr_x, curr_y))].append((bottom, i))
        for i in range(curr_x - K + 1 - curr_loop, curr_x + K + curr_loop - 1):
            if i <= -1 or i >= D:
                continue
            left, right = curr_y - K - curr_loop, curr_y + K + curr_loop
            if -1 < left < D and ship[i][left] == 0 and (i, left) in potential_leaks:
                distance_map[bfs(ship, i, left, (curr_x, curr_y))].append((i, left))
            if -1 < right < D and ship[i][right] == 0 and (i, right) in potential_leaks:
                distance_map[bfs(ship, i, right, (curr_x, curr_y))].append((i, right))
        curr_loop += 1
    return distance_map


'''
If the sense action is performed and a leak is found, then the cells inside the detection square are checked to
determine the cells closest to the current location of the bot that could potentially contain the leak
'''
def closest_moves_leak_found(ship, curr_x, curr_y, potential_leaks, K):
    distance_map = defaultdict(list)
    for r in range(curr_x - K, curr_x + K + 1):
        if r <= -1 or r >= D:
            continue
        for c in range(curr_y - K, curr_y + K + 1):
            if c <= -1 or c >= D or ship[r][c] == 1 or (r, c) not in potential_leaks:
                continue
            distance_map[bfs(ship, r, c, (curr_x, curr_y))].append((r, c))
    return distance_map


'''
Runs a simulation of the bot's actions until the leak is found. After each time the bot moves, it runs the sense action
and follows the guidelines as to whether a leak is found or not. Bfs is used to determine the length of the path of 
the bot's current location and the next location is must travel.
'''
def simulate (ship, start_x, start_y, leak1, leak2, potential_leaks, K):
    num_moves = 0
    curr_location = (start_x, start_y)
    leaks_found = []
    while (leak1 not in leaks_found or leak2 not in leaks_found):
        if curr_location in potential_leaks:
            potential_leaks.remove(curr_location)
        potential_leaks, leak_detected = detect(ship, curr_location[0], curr_location[1], leak1, leak2, potential_leaks, K)
        num_moves += 1

        # visualize_grid(ship, curr_location, leak1, leak2, curr_location)
        # check = input()

        if not leak_detected:
            distance_map = closest_moves_leak_not_found(ship, curr_location[0], curr_location[1], potential_leaks, K)
        else:
            distance_map = closest_moves_leak_found(ship, curr_location[0], curr_location[1], potential_leaks, K)

        closest_moves = distance_map[min(distance_map.keys())]
        next_location = random.choice(closest_moves) if len(closest_moves) > 1 else closest_moves[0]
        num_moves += bfs(ship, next_location[0], next_location[1], curr_location)
        curr_location = next_location
        if curr_location == leak1:
            leaks_found.append(leak1)
        elif curr_location == leak2:
            leaks_found.append(leak2)
    return num_moves


'''
Sets up the location of the robot's starting point, the button, the leak, and the ship itself. It first calls detect
on the current location of the bot and then randomly assigns the location of the leak, just to ensure that the location 
of the leak is not within the first sense action. It then runs the game for Bot 1. It returns the total number of moves 
computed to find where the leak is.
'''
def run_bot5():
    ship = [[1 for i in range(D)] for j in range(D)]
    start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)   # start coordinates for the bot
    ship[start_x][start_y], open_cells = 0, set()
    blocked_one_window_cells = {(start_x, start_y)}
    create_ship(ship, blocked_one_window_cells, open_cells)

    K = ((D // 2 - 1) + 1) // 2
    potential_leaks = open_cells.copy()
    detect(ship, start_x, start_y, (-1, -1),  (-1, -1), potential_leaks, K)
    leak_cell1, leak_cell2 = random.choice(list(potential_leaks)), random.choice(list(potential_leaks))
    while leak_cell2 == leak_cell1:
        leak_cell2 = random.choice(list(potential_leaks))
    num_moves = simulate(ship, start_x, start_y, leak_cell1, leak_cell2, potential_leaks, K)
    return num_moves


if __name__ == '__main__':
    total_moves = 0
    for i in range(100):
        print("Trial", i, "completed")
        total_moves += run_bot5()
    print(total_moves / 100)
