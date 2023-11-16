'''
Implementation for Bot9
@author Steven Tan
'''
from __future__ import division
import random
from collections import deque
import math
from copy import deepcopy
import heapq


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

'''
A* search to make a path that avoids low-probability cells when determining a path from bot to the cell we are 
considering. This will affect num_moves since a more optimized path will effect the number of cells in the path 
from the bot to the cell under consideration.
'''
def astar(ship, start_x, start_y, goal, combinationprobs):
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

                if neighbor not in distTo or temp_dist < distTo[neighbor] or (nx, ny) not in combinationprobs:
                    print(combinationprobs[(nx, ny)])
                    distTo[neighbor] = temp_dist
                    parent[neighbor] = curr_cell
                    prob_factor = D * combinationprobs[tuple({(nx, ny), (3,3)})]
                    heapq.heappush(pq, (distTo[neighbor] - prob_factor, neighbor))
            processed_cells.add(curr_cell)
    return None


def total_combinations_prob(combination_probs, potential_leaks):
    # total_probability holds the cumulative sum of all of the probabilities in combination_probs
    total_probability = 0

    # In move(), we set the probability of the bot location cell to be 0. Hence, we now have to
    # sum all of the probabilities in combination_probs to update the probabilities of all of the
    # other cells so they add up to 1.
    for potential_cell1 in potential_leaks:
        for potential_cell2 in potential_leaks:
            total_probability = total_probability + combination_probs[tuple({potential_cell1, potential_cell2})]

    return total_probability


# Parameters:
# potential_leaks -> set contaning cells that could have a leak
# combination_probs -> set containing probability of every combination of cells containing a leak
def update_probability(potential_leaks, combination_probs):
    # Create an empty compy of combination_probs to store temporary changes so that probability updates do NOT interfere with subsequent ones
    combination_probs_temp = deepcopy(combination_probs)

    # When we detect that there is no leak, we use the following formula:
    # P(((L(i),L(j))|A)|no L(bot_position)) = P(((L(i),L(j))|A)/SUM[all cell combinations (x,y) in potential_leaks](P(((L(x),L(y))|A))
    # We use total_combinations_prob() to calculate the denominator for this probability
    sum_of_probabilities = total_combinations_prob(combination_probs, potential_leaks)

    # For every combination of cells in potential_leaks, apply the aforementioned formula
    for potential_cell1 in potential_leaks:
        for potential_cell2 in potential_leaks:
            combination_probs_temp[(potential_cell1, potential_cell2)] = combination_probs_temp[tuple(
                {potential_cell1, potential_cell2})] / sum_of_probabilities

    # Update combination_probs to the values in combination_probs_sample
    combination_probs = deepcopy(combination_probs_temp)

    return combination_probs


'''
Formula: P((L(c1,c2)|A)|B(bot)) = P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
    NOTE -> in this function, we are only concerned with the numerator, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
Variables:
    c1,c2 -> cells containing candidate1, candidate2
    bot -> cell containing bot
    B(bot) -> beep heard at position of bot
    L(c1,c2) -> leak in c1 and c2
    (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
    A -> all previous given information
'''


def numerator_prob_when_beep(ship, bot_position, candidate1, candidate2, combination_probs):
    # Extract variable information
    (bot_x, bot_y) = bot_position
    (c1_x, c1_y) = candidate1
    (c2_x, c2_y) = candidate2

    # Compute distances from bot to candidate cells (candidate 1 and candidate2) for leak1 and leak2
    dist_to_candidate1 = len(bfs(ship, bot_x, bot_y, candidate1))
    dist_to_candidate2 = len(bfs(ship, bot_x, bot_y, candidate2))

    # Extract P(L(c1,c2)|A) from combination_probs using (c1,c2)
    current_prob_of_c1_c2 = combination_probs[tuple({candidate1, candidate2})]

    # Compute P(B(bot)|(L(c1)|A))
    prob_beep_given_leak_c1 = (math.e) ** ((-1) * alpha * (dist_to_candidate1 - 1))

    # Compute P(B(bot)|(L(c2)|A))
    prob_beep_given_leak_c2 = (math.e) ** ((-1) * alpha * (dist_to_candidate2 - 1))

    # Compute P(B(bot)|(L(c1,c2)|A))
    # This is equal to: 1-(1-(P(B(bot)|(L(c1)|A)))*(1-(P(B(bot)|(L(c2)|A)))
    prob_beep_given_any_leak = 1 - ((1 - prob_beep_given_leak_c1) * (1 - prob_beep_given_leak_c2))

    # Compute numerator of formula, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    numerator = (current_prob_of_c1_c2) * (prob_beep_given_any_leak)

    # Since we only return the numerator, just return this value
    return numerator


'''
Formula: P((L(c1,c2)|A)|B(bot)) = P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        NOTE -> we are only concerned with the denominator in this function, which is:
        SUM[P(B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        However, since we calculate the probability for every combination, we call the numerator function 
        as we iterate and SUM through all combinations of cells in potential_leaks
    Variables:
        c1,c2 -> cells containing candidate1, candidate2
        bot -> cell containing bot
        B(bot) -> beep heard at position of bot
        L(c1,c2) -> leak in c1 and c2
        (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        A -> all previous given information
'''


def denominator_prob_when_beep(ship, bot_position, potential_leaks, combination_probs):
    cumulative_sum_of_probs = 0

    for cell_1 in potential_leaks:
        for cell_2 in potential_leaks:
            # The numerator computes P(B(bot)|(L(x,y)|A))*P(L(x,y)|A) and adds it to the cumulative sum, where x = cell_1 and y = cell_2
            cumulative_sum_of_probs = cumulative_sum_of_probs + numerator_prob_when_beep(ship, bot_position, cell_1,
                                                                                         cell_2, combination_probs)

    denominator = cumulative_sum_of_probs
    return denominator


'''
Formula: P((L(c1,c2)|A)|no B(bot)) = P(no B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A)] 
for every x,y in potential_leaks
        NOTE -> in this function, we are only concerned with the numerator, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    Variables:
        c1,c2 -> cells containing candidate1, candidate2
        bot -> cell containing bot
        B(bot) -> beep heard at position of bot
        L(c1,c2) -> leak in c1 and c2
        (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        A -> all previous given information
'''


def numerator_prob_when_no_beep(ship, bot_position, candidate1, candidate2, combination_probs):
    (bot_x, bot_y) = bot_position
    (c1_x, c1_y) = candidate1
    (c2_x, c2_y) = candidate2

    dist_to_candidate1 = len(bfs(ship, bot_x, bot_y, candidate1))
    dist_to_candidate2 = len(bfs(ship, bot_x, bot_y, candidate2))

    # Extract P(L(c1,c2)|A) from combination_probs using (c1,c2)
    current_prob_of_c1_c2 = combination_probs[tuple({candidate1, candidate2})]

    # Compute P(B(bot)|(L(c1)|A))
    prob_beep_given_leak_c1 = 1 - ((math.e) ** ((-1) * alpha * (dist_to_candidate1 - 1)))

    # Compute P(B(bot)|(L(c2)|A))
    prob_beep_given_leak_c2 = 1 - ((math.e) ** ((-1) * alpha * (dist_to_candidate2 - 1)))

    # Compute P(no B(bot)|(L(c1,c2)|A)). This is equal to: (1-(P(B(bot)|(L(c1)|A)))*(1-(P(B(bot)|(L(c2)|A)))
    prob_no_beep_given_any_leak = ((1 - prob_beep_given_leak_c1) * (1 - prob_beep_given_leak_c2))

    # Compute numerator of formula, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    numerator = (current_prob_of_c1_c2) * (prob_no_beep_given_any_leak)

    return numerator


'''
Formula: P((L(c1,c2)|A)|B(bot)) = P(no B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        NOTE -> we are only concerned with the denominator in this function, which is:
        SUM[P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        However, since we calculate the probability for every combination, we call the numerator function 
        as we iterate and SUM through all combinations of cells in potential_leaks
    Variables:
        c1,c2 -> cells containing candidate1, candidate2
        bot -> cell containing bot
        B(bot) -> beep heard at position of bot
        L(c1,c2) -> leak in c1 and c2
        (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        A -> all previous given information
'''


def denominator_prob_when_no_beep(ship, bot_position, potential_leaks, combination_probs):
    cumulative_sum_of_probs = 0

    for cell_1 in potential_leaks:
        for cell_2 in potential_leaks:
            # The numerator computes P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A) and adds it to the cumulative sum, where x = cell_1 and y = cell_2
            cumulative_sum_of_probs = cumulative_sum_of_probs + numerator_prob_when_no_beep(ship, bot_position, cell_1,
                                                                                            cell_2, combination_probs)

    denominator = cumulative_sum_of_probs
    return denominator


'''
The beep in come from either leak, so we use the formula:
    # P(beep from leak1 OR beep from leak2) = P(beep from leak1) + P(beep from leak2) - P(beep from leak1)*P(beep from leak2)
    # This is basically since the beeps are not necessarily disjoint; in other words, beeps can come from ANY of the leak
'''


def detect(ship, bot_position, leak1, leak2, potential_leaks, combination_probs):
    (curr_x, curr_y) = bot_position

    # Compute distance from current bot cell to leak cells
    distance_to_leak1 = len(bfs(ship, curr_x, curr_y, leak1))
    distance_to_leak2 = len(bfs(ship, curr_x, curr_y, leak2))

    # Compute probability of beep using the distances to the leak cells
    prob_beep_leak1 = (math.e) ** ((-1) * alpha * (distance_to_leak1 - 1))
    prob_beep_leak2 = (math.e) ** ((-1) * alpha * (distance_to_leak2 - 1))
    prob_either_beep = (prob_beep_leak1 + prob_beep_leak2) - (prob_beep_leak1 * prob_beep_leak2)

    # Simulate the detection of a beep, where the beep must appear from one or both of the leaks
    num = random.uniform(0, 1)
    beep = False
    if num <= prob_either_beep:
        beep = True

    # Create an empty compy of combination_probs to store temporary changes so that probability updates do NOT interfere with subsequent ones
    combination_probs_sample = deepcopy(combination_probs)

    # If there is a beep, update combination_probs using numerator_prob_when_beep() and denominator_prob_when_beep()
    if beep:
        denominator_when_beep = denominator_prob_when_beep(ship, bot_position, potential_leaks, combination_probs)
        for candidate1 in potential_leaks:
            for candidate2 in potential_leaks:
                numerator_when_beep = numerator_prob_when_beep(ship, bot_position, candidate1, candidate2,
                                                               combination_probs)
                combination_probs_sample[(candidate1, candidate2)] = numerator_when_beep / denominator_when_beep

    # If there is a beep, update combination_probs using numerator_prob_when_no_beep() and denominator_prob_when_no_beep()
    else:
        denominator_when_no_beep = denominator_prob_when_no_beep(ship, bot_position, potential_leaks, combination_probs)
        for candidate1 in potential_leaks:
            for candidate2 in potential_leaks:
                numerator_when_no_beep = numerator_prob_when_no_beep(ship, bot_position, candidate1, candidate2,
                                                                     combination_probs)
                combination_probs_sample[(candidate1, candidate2)] = numerator_when_no_beep / denominator_when_no_beep

    # Update combination_probs to the values in combination_probs_sample
    combination_probs = deepcopy(combination_probs_sample)


def move(ship, bot_position, leak1, leak2, potential_leaks, combination_probs):
    # Extract the starting bot position
    (curr_x, curr_y) = bot_position

    # Check if the starting position of the is in either of the leaks
    if (curr_x, curr_y) == leak1 or (curr_x, curr_y) == leak2:
        return 0

    # Initialize the number of bot movements as num_moves = 0
    num_moves = 0

    # Iterate until a leak is found
    while True:

        # Check if the current bot position contains any of the leaks
        if bot_position == leak1 or bot_position == leak2:
            return num_moves

        # If the current bot position is not one of the leaks, do the following:
        else:
            # Remove the bot position from potential_leaks since it is no longer a leak
            if bot_position in potential_leaks:
                potential_leaks.remove(bot_position)

                # Update combination_probs so that EVERY COMBINATION with bot_position has a probability of 0
                for potential_other_cell in potential_leaks:
                    combination_probs[tuple(potential_other_cell, bot_position)] = 0

                # Update combination_probs to make sure that all of the probabilities add up to 1 after the modification
                combination_probs = update_probability(potential_leaks, combination_probs)

        # Call detect() which will simulate the beep
        detect(ship, bot_position, leak1, leak2, potential_leaks, combination_probs)

        # Increment the number of moves by 1 since detect() counts as a move
        num_moves += 1

        # Store the probability that corresponds to the cell that has the highest MARGINAL
        # PROBABILITY in combination_probs
        max_cell_probability = 0

        # Store the coordinates of that cell in max_cell
        max_cell = (0, 0)

        # Find the cell with the highest marginal probability in combination_probs
        for potential_max_cell in potential_leaks:
            marginal_probability = 0

            # For marginal probability example: For P(cell1 containing leak | A), we have to do
            # P(cell1, cell2 containing leaks) + P(cell1, cell3 containing leaks) + ...
            # This would be the marginal probability
            for other_cell in potential_leaks:
                marginal_probability = marginal_probability + combination_probs[tuple({potential_max_cell, other_cell})]

            if (marginal_probability > max_cell_probability):
                max_cell_probability = marginal_probability
                max_cell = potential_max_cell

        # Store the path from the bot to the determined cell
        path = astar(ship, curr_x, curr_y, potential_max_cell, combination_probs)

        # Perform the bot movement and check if any of the cells in the path contain the leak
        for (cellx, celly) in path:

            # If we return to the original bot position in the path, then go to the next position in the path
            if curr_x == cellx and curr_y == celly:
                continue

            # Check if the cell in the path contains either leak. If it does, return the number of moves
            if (cellx, celly) == leak1 or (cellx, celly) == leak2:
                return num_moves

            # If not, remove the cell from potential_leaks and update the probability matrix using update_probability()
            else:
                # Remove the cell from potential_leaks
                if (cellx, celly) in potential_leaks:
                    potential_leaks.remove((cellx, celly))
                    curr_x = cellx
                    curr_y = celly

                    # Update combination_probs so that EVERY COMBINATION with bot_position has a probability of 0
                    for potential_other_cell in potential_leaks:
                        combination_probs[(potential_other_cell, bot_position)] = 0

                    # Update combination_probs to make sure that all of the probabilities add up to 1 after the modification
                    combination_probs = update_probability(potential_leaks, combination_probs)

            num_moves = num_moves + 1


def run_bot9():
    ship = [[1 for i in range(D)] for j in range(D)]
    start_x, start_y = random.randint(0, D - 1), random.randint(0, D - 1)
    bot_start = (start_x, start_y)
    ship[start_x][start_y], open_cells = 0, set()
    blocked_one_window_cells = {(start_x, start_y)}
    create_ship(ship, blocked_one_window_cells, open_cells)

    for i in range(D):
        for j in range(D):
            if ship[i][j] == 0:
                open_cells.add((i, j))

    open_cells.remove(bot_start)

    # Initialize combination_probs, which contains the probability of a COMBINATION
    # of cells containing leak1 and leak2, GIVEN all of the previous information.
    # This would be P((L(x),L(y))|A) where:
    # x and y are the combination of cells
    # A is all of the given information, which is initially nothing
    combination_probs = {}

    # Initialize potential_leaks
    potential_leaks = open_cells.copy()

    # Populate combination_probs by going through EVERY combination of cells in potential_leaks
    for potential_cell1 in potential_leaks:
        for potential_cell2 in potential_leaks:
            # Create a set() to make sure the order of potential_cell1 and potential_cell2 does NOT matter
            # This makes sure that (x,y) is recognized as (y,x)
            cell_combination = set((potential_cell1, potential_cell2))
            # Each value in combination_probs is initally the same at (1/(number of open cells))*(1/(number of open cells -1))
            # This is the probability of (potential_cell1, potential_cell2) containing (leak1, leak2)
            combination_probs[tuple(cell_combination)] = (1 / len(open_cells)) * (1 / (len(open_cells) - 1))

    leak_cell1 = random.choice(list(potential_leaks))
    leak_cell2 = random.choice(list(potential_leaks))

    num_moves = move(ship, bot_start, leak_cell1, leak_cell2, potential_leaks, combination_probs)
    return num_moves


if __name__ == '__main__':
    total_moves = 0
    for i in range(100):
        print("Trial: ", i)
        total_moves += run_bot9()


