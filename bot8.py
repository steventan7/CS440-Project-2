'''
Implementation for Bot1
@author Steven Tan, Ajay Anand
'''
from __future__ import division
import random
from collections import deque
import math
from copy import deepcopy
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
    
# PROBABILITY GIVEN THAT WE GOT A BEEP
# def probIsBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, leak, probArray):#probability that leak in j given that beep in cell i
#     # alpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    
#     # dsteps = minStepsBot(bot_x, bot_y, cellx, celly)#dsteps
#     dsteps = len(bfs(ship, bot_x, bot_y, (cellx,celly)))
#     # print("Dsteps", dsteps)
#     probLeakinJ = probArray[cellx][celly]
#     probbeepinigivenleakinj = (math.e) ** ((-1)*alpha*(dsteps-1))
#     probbeepini = 0.0
#     for i in range(D):
#         for j in range(D):
#             if (i,j) in potential_leaks:
#                 # dsteps = minStepsBot(bot_x, bot_y, i,j)
#                 dsteps = len(bfs(ship, bot_x, bot_y, (i,j)))
#                 probbeepini += probArray[i][j] * (math.e ** ((-1)*alpha*(dsteps-1)))
    
#     prob = (probLeakinJ*probbeepinigivenleakinj) / probbeepini
#     return prob

# PROBABILITY GIVEN THAT WE DID NOT GET A BEEP
# def probNoBeep(ship, bot_x, bot_y, cellx, celly, potential_leaks, leak, probArray):#probability that leak in j given that no beep in cell i
#     # alpha = random.uniform(0,1)#returns a float between 0 and 1(T.A. said it must be between 0 and 1)
#     dsteps = len(bfs(ship, bot_x, bot_y, (cellx,celly)))
    
#     probLeakinJ = probArray[cellx][celly]
#     probnotBeepinigivenleakincellj = (1-(math.e ** ((-1)*alpha*(dsteps-1))))
    
    
#     probnoBeepini = 0.0
    
    
#     for i in range(D):
#         for j in range(D):
#             if (i,j) in potential_leaks:
#                 dsteps = len(bfs(ship, bot_x, bot_y, (i,j)))
#                 probnoBeepini += probArray[i][j] * (1-(math.e ** ((-1)*alpha*(dsteps-1))))
    
    
#     prob = (probLeakinJ*probnotBeepinigivenleakincellj) / probnoBeepini
#     return prob
    
# PROBABILITY GIVEN THAT CURRENT CELL DOES NOT HAVE A LEAK
# def updateProb(ship, curr_x, curr_y, probArray, potential_leaks):
#     # print("Prob", probArray[curr_x][curr_y])
#     num = probArray[curr_x][curr_y]
#     dem = 0.0
    
#     for (i,j) in potential_leaks:
#         dem += probArray[i][j]
                
#     # print("Denominator", dem)
    
   
#     return num/dem


def updateProb(ship, curr_x, curr_y, probArray, potential_leaks):
    # print("Prob", probArray[curr_x][curr_y])
    num = probArray[curr_x][curr_y]
    dem = 0.0
    
    for (i,j) in potential_leaks:
        dem += probArray[i][j]
                
    # print("Denominator", dem)
    
   
    return num/dem


# Parameters: 
    # ship -> matrix of cells
    # bot_position = (bot_x, bot_y) -> bot position
    # candiate1 = (c1_x, c1_y) -> 1st candidate cell for leak
    # candiate2 = (c2_x, c2_y) -> 2nd candidate cell for the leak
    # potential_leaks -> set contaning cells that could have a leak
    # leak1 -> first leak cell
    # leak2 -> second leak cell
    # combination_probs -> set containing probability of every combination of cells containing a leak
def numerator_prob_when_beep (ship, bot_position, candidate1, candidate2, combination_probs):    
    # Extract variable information
    (bot_x, bot_y) = bot_position
    (c1_x, c1_y) = candidate1
    (c2_x, c2_y) = candidate2

    # Compute distances from bot to candidate cells (candidate 1 and candidate2) for leak1 and leak2
    dist_to_candidate1 = len(bfs(ship, bot_x, bot_y, candidate1))
    dist_to_candidate2 = len(bfs(ship, bot_x, bot_y, candidate2))

    # Formula: P((L(c1,c2)|A)|B(bot)) = P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        # NOTE -> in this function, we are only concerned with the numerator, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    # Variables:
        # c1,c2 -> cells containing candidate1, candidate2
        # bot -> cell containing bot
        # B(bot) -> beep heard at position of bot
        # L(c1,c2) -> leak in c1 and c2
        # (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        # A -> all previous given information

    # Extract P(L(c1,c2)|A) from combination_probs using (c1,c2)
    current_prob_of_c1_c2 = combination_probs[(candidate1, candidate2)]

    # Compute P(B(bot)|(L(c1)|A))
    prob_beep_given_leak_c1 = (math.e) ** ((-1)*alpha*(dist_to_candidate1-1))

    # Compute P(B(bot)|(L(c2)|A))
    prob_beep_given_leak_c2 = (math.e) ** ((-1)*alpha*(dist_to_candidate2-1))

    # Compute P(B(bot)|(L(c1,c2)|A))
    # This is equal to: 1-(1-(P(B(bot)|(L(c1)|A)))*(1-(P(B(bot)|(L(c2)|A)))
    prob_beep_given_any_leak = 1-((1-prob_beep_given_leak_c1)*(1-prob_beep_given_leak_c2))

    # Compute numerator of formula, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    numerator = (current_prob_of_c1_c2)*(prob_beep_given_any_leak)

    # Since we only return the numerator, just return this value
    return numerator


# Parameters: 
    # ship -> matrix of cells
    # bot_position = (bot_x, bot_y) -> bot position
    # candiate1 = (c1_x, c1_y) -> 1st candidate cell for leak
    # candiate2 = (c2_x, c2_y) -> 2nd candidate cell for the leak
    # potential_leaks -> set contaning cells that could have a leak
    # leak1 -> first leak cell
    # leak2 -> second leak cell
    # combination_probs -> set containing probability of every combination of cells containing a leak
def denominator_prob_when_beep(ship, bot_position, potential_leaks, combination_probs):
    
    # Formula: P((L(c1,c2)|A)|B(bot)) = P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        # NOTE -> we are only concerned with the denominator in this function, which is:
        # SUM[P(B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        # However, since we calculate the probability for every combination, we call the numerator function 
        # as we iterate and SUM through all combinations of cells in potential_leaks
    # Variables:
        # c1,c2 -> cells containing candidate1, candidate2
        # bot -> cell containing bot
        # B(bot) -> beep heard at position of bot
        # L(c1,c2) -> leak in c1 and c2
        # (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        # A -> all previous given information
    
    # We start the cumulative sum at 0
    cumulative_sum_of_probs = 0
    
    # For every COMBINATION of cells (cell_x, cell_y) in potential_leaks, compute the numerator function
    for cell_1 in potential_leaks:
        for cell_2 in potential_leaks:
            # The numerator computes P(B(bot)|(L(x,y)|A))*P(L(x,y)|A) and adds it to the cumulative sum, where x = cell_1 and y = cell_2
            cumulative_sum_of_probs = cumulative_sum_of_probs + numerator_prob_when_beep(ship, bot_position, cell_1, cell_2, potential_leaks, combination_probs)

    # The cumulative sum is basically the denominator
    denominator = cumulative_sum_of_probs

    # We return the denominator
    return denominator

# Parameters: 
    # ship -> matrix of cells
    # bot_position = (bot_x, bot_y) -> bot position
    # candiate1 = (c1_x, c1_y) -> 1st candidate cell for leak
    # candiate2 = (c2_x, c2_y) -> 2nd candidate cell for the leak
    # potential_leaks -> set contaning cells that could have a leak
    # leak1 -> first leak cell
    # leak2 -> second leak cell
    # combination_probs -> set containing probability of every combination of cells containing a leak
def numerator_prob_when_no_beep (ship, bot_position, candidate1, candidate2, combination_probs):    
    # Extract variable information
    (bot_x, bot_y) = bot_position
    (c1_x, c1_y) = candidate1
    (c2_x, c2_y) = candidate2

    # Compute distances from bot to candidate cells (candidate 1 and candidate2) for leak1 and leak2
    dist_to_candidate1 = len(bfs(ship, bot_x, bot_y, candidate1))
    dist_to_candidate2 = len(bfs(ship, bot_x, bot_y, candidate2))

    # Formula: P((L(c1,c2)|A)|no B(bot)) = P(no B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        # NOTE -> in this function, we are only concerned with the numerator, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    # Variables:
        # c1,c2 -> cells containing candidate1, candidate2
        # bot -> cell containing bot
        # B(bot) -> beep heard at position of bot
        # L(c1,c2) -> leak in c1 and c2
        # (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        # A -> all previous given information

    # Extract P(L(c1,c2)|A) from combination_probs using (c1,c2)
    current_prob_of_c1_c2 = combination_probs[(candidate1, candidate2)]

    # Compute P(B(bot)|(L(c1)|A))
    prob_beep_given_leak_c1 = 1-((math.e) ** ((-1)*alpha*(dist_to_candidate1-1)))

    # Compute P(B(bot)|(L(c2)|A))
    prob_beep_given_leak_c2 = 1-((math.e) ** ((-1)*alpha*(dist_to_candidate2-1)))

    # Compute P(no B(bot)|(L(c1,c2)|A))
    # This is equal to: (1-(P(B(bot)|(L(c1)|A)))*(1-(P(B(bot)|(L(c2)|A)))
    prob_no_beep_given_any_leak = ((1-prob_beep_given_leak_c1)*(1-prob_beep_given_leak_c2))

    # Compute numerator of formula, which is P(B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)
    numerator = (current_prob_of_c1_c2)*(prob_no_beep_given_any_leak)

    # Since we only return the numerator, just return this value
    return numerator

# Parameters: 
    # ship -> matrix of cells
    # bot_position = (bot_x, bot_y) -> bot position
    # candiate1 = (c1_x, c1_y) -> 1st candidate cell for leak
    # candiate2 = (c2_x, c2_y) -> 2nd candidate cell for the leak
    # potential_leaks -> set contaning cells that could have a leak
    # leak1 -> first leak cell
    # leak2 -> second leak cell
    # combination_probs -> set containing probability of every combination of cells containing a leak
def denominator_prob_when_no_beep(ship, bot_position, potential_leaks, combination_probs):
    
    # Formula: P((L(c1,c2)|A)|B(bot)) = P(no B(bot)|(L(c1,c2)|A))*P(L(c1,c2)|A)/SUM[P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        # NOTE -> we are only concerned with the denominator in this function, which is:
        # SUM[P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A)] for every x,y in potential_leaks
        # However, since we calculate the probability for every combination, we call the numerator function 
        # as we iterate and SUM through all combinations of cells in potential_leaks
    # Variables:
        # c1,c2 -> cells containing candidate1, candidate2
        # bot -> cell containing bot
        # B(bot) -> beep heard at position of bot
        # L(c1,c2) -> leak in c1 and c2
        # (x,y) -> any combination of cells x,y in potential_leaks (where x and y are in potential leaks)
        # A -> all previous given information
    
    # We start the cumulative sum at 0
    cumulative_sum_of_probs = 0
    
    # For every COMBINATION of cells (cell_x, cell_y) in potential_leaks, compute the numerator function
    for cell_1 in potential_leaks:
        for cell_2 in potential_leaks:
            # The numerator computes P(no B(bot)|(L(x,y)|A))*P(L(x,y)|A) and adds it to the cumulative sum, where x = cell_1 and y = cell_2
            cumulative_sum_of_probs = cumulative_sum_of_probs + numerator_prob_when_no_beep(ship, bot_position, cell_1, cell_2, combination_probs)

    # The cumulative sum is basically the denominator
    denominator = cumulative_sum_of_probs

    # We return the denominator
    return denominator

# def detect(ship, curr_x, curr_y, leak, potential_leaks, K, probArray):
#     #returns a float between 0 and 1(T.A. said it must be between 0 and 1)
    
#     # dsteps = minStepsLeak(ship, curr_x, curr_y, leak)#dsteps
#     dsteps = len(bfs(ship, curr_x, curr_y, leak))
#     # probBeep = 0.0
#     # for i in range(D):
#     #     for j in range(D):
#     #         if (i,j) in potential_leaks:
#     #             dsteps = minStepsBot(curr_x, curr_y, i,j)
#     #             probBeep += probArray[i][j] * (math.e ** ((-1)*alpha*(dsteps-1)))
                
                
#     probBeep = (math.e) ** ((-1)*alpha*(dsteps-1))#probability for beep
#     num = random.uniform(0,1)
    
#     beep = False
    
#     if num <= probBeep:
#         beep = True
        
#     probArraySample = [[0 for i in range(D)] for j in range(D)]
    
    
#     if beep:#Probability of Leak Given that there is a beep
#         print("Beep")
#         for (nx, ny) in potential_leaks:
#             probArraySample[nx][ny] = probIsBeep(ship, curr_x, curr_y, nx,ny,potential_leaks, leak, probArray)
            
            
#     else:#Probability of Leak Given that there is not a beep
#         print("No beep")
#         for (nx, ny) in potential_leaks:
#             probArraySample[nx][ny] = probNoBeep(ship, curr_x, curr_y, nx, ny, potential_leaks, leak, probArray)

#     for i in range(D):
#         for j in range(D):
#             if (i,j) in potential_leaks:
#                 probArray[i][j] = probArraySample[i][j]

# Parameters:
    # ship -> matrix of cells
    # bot_position = (bot_x, bot_y) -> bot position
    # candiate1 = (c1_x, c1_y) -> 1st candidate cell for leak
    # candiate2 = (c2_x, c2_y) -> 2nd candidate cell for the leak
    # potential_leaks -> set contaning cells that could have a leak
    # leak1 -> first leak cell
    # leak2 -> second leak cell
    # combination_probs -> set containing probability of every combination of cells containing a leak
def detect(ship, bot_position, leak1, leak2, potential_leaks, K, combination_probs):

    # Extract current bot positon
    (curr_x, curr_y) = bot_position

    # The beep in come from either leak, so we use the formula:
        # P(beep from leak1 OR beep from leak2) = P(beep from leak1) + P(beep from leak2) - P(beep from leak1)*P(beep from leak2)
        # This is basically since the beeps are not necessarily disjoint; in other words, beeps can come from ANY of the leak

    # Compute distance from current bot cell to leak cells
    distance_to_leak1 = len(bfs(ship, curr_x, curr_y, leak1))
    distance_to_leak2 = len(bfs(ship, curr_x, curr_y, leak2))
                
    # Compute probability of beep using the distances to the leak cells
    prob_beep_leak1 = (math.e) ** ((-1)*alpha*(distance_to_leak1-1))
    prob_beep_leak2 = (math.e) ** ((-1)*alpha*(distance_to_leak2-1))
    prob_either_beep = (prob_beep_leak1 + prob_beep_leak2) - (prob_beep_leak1 * prob_beep_leak2)

    # Simulate the detection of a beep, where the beep must appear from one or both of the leaks
    num = random.uniform(0,1)
    beep = False
    if num <= prob_either_beep:
        beep = True

    # Create an empty compy of combination_probs to store temporary changes so that probability updates do NOT interfere with subsequent ones
    combination_probs_sample = deepcopy(combination_probs) 
    
    # If there is a beep, update combination_probs using numerator_prob_when_beep() and denominator_prob_when_beep()
    if beep:
        print("Beep")
        denominator_when_beep = denominator_prob_when_beep(ship, bot_position, potential_leaks, combination_probs)
        for candidate1 in potential_leaks:
            for candidate2 in potential_leaks:
                numerator_when_beep = numerator_prob_when_beep (ship, bot_position, candidate1, candidate2, combination_probs)
                combination_probs_sample[(candidate1, candidate2)] = numerator_when_beep/denominator_when_beep
            
    # If there is a beep, update combination_probs using numerator_prob_when_no_beep() and denominator_prob_when_no_beep()    
    else:
        print("No Beep")
        denominator_when_no_beep = denominator_prob_when_no_beep(ship, bot_position, potential_leaks, combination_probs)
        for candidate1 in potential_leaks:
            for candidate2 in potential_leaks:
                numerator_when_no_beep = numerator_prob_when_no_beep(ship, bot_position, candidate1, candidate2, combination_probs)
                combination_probs_sample[(candidate1, candidate2)] = numerator_when_no_beep/denominator_when_no_beep

    # Update combination_probs to the values in combination_probs_sample
    combination_probs = deepcopy(combination_probs_sample)


def move(ship, bot_position, leak1, leak2, potential_leaks, K, combination_probs):

    # Extract the starting bot position
    (curr_x, curr_y) = bot_position

    # Check if the starting position of the is in either of the leaks
    if (curr_x, curr_y) == leak1 or (curr_x, curr_y) == leak2:
        return 0
        
    # Initialize the number of bot movements as num_moves = 0
    num_moves = 0

    # Iterate until a leak is found
    while True:

        print("Currx, Curr_y: ",curr_x, curr_y)

        # Check if the current bot position contains any of the leaks
        if (curr_x, curr_y) == leak1 or (curr_x, curr_y) == leak2:
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
                    
        

        path = bfs(ship, curr_x, curr_y, (endX,endY))
        
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
        

def run_bot3():
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
    combination_probs = {}
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
   
    # leak_cell = random.choice(list(open_cells))
    potential_leaks = open_cells.copy()
    # print("Potential Leaks", potential_leaks)
    leak_cell = random.choice(list(potential_leaks))
    

    num_moves = move(ship, start_x, start_y, leak_cell, potential_leaks, K, probArray)

    return num_moves


if __name__ == '__main__':
    total_moves = 0
    print(run_bot3())
