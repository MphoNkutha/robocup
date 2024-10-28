import numpy as np

def c_matrix(players, formation):

    num_player = len(players)
    num_pos = len(formation)
    costmatrix = np.zeros((num_player,num_pos))
    for i, player in enumerate(players):
        for j, position in enumerate(formation):
            costmatrix[i,j] = np.linalg.norm(np.array(player)-np.array(position))
    return costmatrix



def subtract_row_and_column_minima(cost_matrix):
    # Step 2: Subtract Row Minimums
    cost_matrix -= cost_matrix.min(axis=1)[:, np.newaxis]
    # Step 3: Subtract Column Minimums
    cost_matrix -= cost_matrix.min(axis=0)
    return cost_matrix

def cover_zeros_and_mask(cost_matrix, n, m):
    # Step 4: Cover All Zeros with a Minimum Number of Lines
    masked_zeros = np.zeros((n, m), dtype=bool)
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(m, dtype=bool)
    
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] == 0 and not covered_rows[i] and not covered_cols[j]:
                masked_zeros[i, j] = True
                covered_rows[i] = True
                covered_cols[j] = True

    covered_rows[:] = False
    covered_cols[:] = False
    
    return masked_zeros, covered_rows, covered_cols

def execute_hungarian_algorithm(cost_matrix):
    cost_matrix = cost_matrix.astype(float)
    n, m = cost_matrix.shape

    # Step 2 and 3: Subtract Row and Column Minima
    cost_matrix = subtract_row_and_column_minima(cost_matrix)

    # Step 4: Mask Initial Zeros and Create Cover Arrays
    masked_zeros, covered_rows, covered_cols = cover_zeros_and_mask(cost_matrix, n, m)
    unmasked_zeros = np.zeros((n, m), dtype=bool)

    while True:
        # Cover columns that have masked zeros
        for j in range(m):
            if np.any(masked_zeros[:, j]):
                covered_cols[j] = True
        
        # If all columns are covered, break out of the loop
        if np.sum(covered_cols) == n:
            break
        
        while True:
            zero_found = False
            for i in range(n):
                if covered_rows[i]:
                    continue
                for j in range(m):
                    if cost_matrix[i, j] == 0 and not covered_cols[j]:
                        unmasked_zeros[i, j] = True
                        
                        # If there are no masked zeros in this row
                        if not np.any(masked_zeros[i, :]):
                            path = [(i, j)]
                            while True:
                                row = np.argmax(masked_zeros[:, j])
                                if masked_zeros[row, j]:
                                    path.append((row, j))
                                else:
                                    break
                                
                                col = np.argmax(unmasked_zeros[row, :])
                                if unmasked_zeros[row, col]:
                                    path.append((row, col))
                                    i, j = row, col
                                else:
                                    break

                            for i, j in path:
                                masked_zeros[i, j] = ~masked_zeros[i, j]
                                unmasked_zeros[i, j] = False
                            
                            covered_rows.fill(False)
                            covered_cols.fill(False)
                            zero_found = True
                            break
                        else:
                            covered_rows[i] = True
                            covered_cols[np.argmax(masked_zeros[i, :])] = False
                if zero_found:
                    break
            
            if zero_found:
                break
            
            # Step 5: Adjust the Matrix  if Necessary
            min_val = np.min(cost_matrix[~covered_rows][:, ~covered_cols])
            cost_matrix[covered_rows] += min_val
            cost_matrix[:, ~covered_cols] -= min_val
    
    # Step 7: Return the Optimal Assignment
    assignments = []

# Step 1: Iterate over the rows of the array
    for i in range(masked_zeros.shape[0]):
        for j in range(masked_zeros.shape[1]):
            if masked_zeros[i, j] != 0:
                assignments.append([i, j])


    assignments = np.array(assignments)
    return assignments

def role_assignment(teammate_positions, formation_positions):
    # Step 1: Construct the Cost Matrix
    num_players = len(teammate_positions)
    num_positions = len(formation_positions)
    cost_matrix = c_matrix(teammate_positions,formation_positions)

    assignments = execute_hungarian_algorithm(cost_matrix)
    # point_preferences = {}
    # for i, j in assignments:
    #      point_preferences[i + 1] = formation_positions[j]
    
    player_assignments = {}
    assigned_positions = set()

    # Assign players to their optimal formation positions without duplicates
    for player, position in assignments:
        player_assignments[player + 1] = np.array(formation_positions[position])
        assigned_positions.add(position)

    # Handle cases where players may not have been assigned
    for i in range(num_players):
        if i + 1 not in player_assignments:
            for j in range(num_positions):
                if j not in assigned_positions:
                    player_assignments[i + 1] = np.array(formation_positions[j])
                    assigned_positions.add(j)
                    break

    return player_assignments
def pass_reciever_selector(player_unum, teammate_positions, final_target):
    
    # Input : Locations of all teammates and a final target you wish the ball to finish at
    # Output : Target Location in 2d of the player who is recieveing the ball
    #-----------------------------------------------------------#
        # #Calculate the distance between the player and all teammates
    distances = [np.linalg.norm(np.array(teammate_positions[i])-np.array(teammate_positions[player_unum])) for i in range(len(teammate_positions)-1)]
    
    #Disregard the player itself
    distances[player_unum] = np.inf

    #Find the closest player
    receiver= (np.argmin(distances))
    target = teammate_positions[receiver]





    # If the goal post is closer than the closest player, pass to the goal post
    if np.linalg.norm(np.array(final_target)-np.array(teammate_positions[player_unum])) < np.linalg.norm(np.array(teammate_positions[receiver])-np.array(teammate_positions[player_unum])):
        target = final_target

    
    return target

    # # Example
    # pass_reciever_unum = player_unum + 1                  #This starts indexing at 1, therefore player 1 wants to pass to player 2
    
    # if pass_reciever_unum != 12:
    #     target = teammate_positions[pass_reciever_unum-1] #This is 0 indexed so we actually need to minus 1 
    # else:
    #     target = final_target 
    
    # return target

# player_positions = [(-14,0),(-9,-5),(-9,0),(-9,5),(-5,-5),(-5,0),(-5,5),(-1,-6),(-1,-2.5),(-1,2.5),(-1,6)]
# formation_positions = [(-13,0),(-10, -2),(-11, 3),(-8, 0),(-3, 0),(0, 1),(2, 0), (3, 3),(8, 0),(9, 1),(12, 0)]
# print(role_assignment(player_positions, formation_positions))
