'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
import random
from colorama import init, Back, Style
init(autoreset=True)

DIRECTIONS = [0, 1, 0, -1, 0]
D = 100


# checks if the a cell has exactly one open neighbor
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


# Used to visualize the difference components of the game
def visualize_grid (ship, start, goal, bot, path):
    if path == None:
        print("NO PATH!")
    for i in range(D):
        curr_row = ""
        for j in range(D):
            if (i,j) == start:
                curr_row += (Style.RESET_ALL + Back.BLUE + Style.BRIGHT + "__")
            elif (i,j) == goal:
                curr_row += (Style.RESET_ALL + Back.LIGHTBLACK_EX + Style.BRIGHT + "__")
            elif (i,j) == bot:
                curr_row += (Style.RESET_ALL + Back.GREEN + Style.BRIGHT + "__")
            elif ship[i][j] == 2:
                curr_row += (Style.RESET_ALL + Back.RED + "__")
            elif (i,j) in path:
                curr_row += (Style.RESET_ALL + Back.BLUE + Style.BRIGHT + "__")
            elif ship[i][j] == 0:
                curr_row += (Style.RESET_ALL + Back.YELLOW + "__")
            elif ship[i][j] == 1:
                curr_row += (Style.RESET_ALL + Back.WHITE + "__")
            else:
                curr_row += (Style.RESET_ALL + "__")
        print(curr_row)
    print()
    print()


'''
Starts the game. The bot blindly follows the shortest path from performing BFS while the fire spreads until the fire or 
robot reaches the button or the fire reaches the bot.
'''
def run_bot2():
    return True

'''
Sets up the location of the robot's starting point, the button, the fire, and the ship itself. It first calls BFS on 
the starting point of the bot and fire to the button. It immediately returns true if the distance of the bot is less 
than the distance of fire to the button. Else, it runs the game for Bot 1.
'''
def main():
    ship = [[1 for i in range(D)] for j in range(D)]
    start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)   # start coordinates
    ship[start_x][start_y], open_cells = 0, set()
    blocked_one_window_cells = {(start_x, start_y)}

    create_ship(ship, blocked_one_window_cells, open_cells)


if __name__ == '__main__':
    wins, loses = 0, 0
    for i in range(100):
        print("Trial: ", i)
        if main():
            wins += 1
        else:
            loses += 1
    print("wins", wins)
    print("loses", loses)


