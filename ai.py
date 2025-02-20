from __future__ import absolute_import, division, print_function
import copy, random
from game import Game
from collections import deque
import math
import numpy as np



MOVES = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}
MAX_PLAYER, CHANCE_PLAYER = 0, 1 

# Tree node. To be used to construct a game tree. 
class Node: 
    # Recommended: do not modify this __init__ function
    def __init__(self, state, player_type):
        self.state = (state[0], state[1])

        # to store a list of (direction, node) tuples
        self.children = []

        self.player_type = player_type

    # returns whether this is a terminal state (i.e., no children)
    def is_terminal(self):
        #TODO: complete this

        return len(self.children) == 0
    


# AI agent. Determine the next move.
class AI:
    # Recommended: do not modify this __init__ function
    def __init__(self, root_state, search_depth=3, use_improved_heuristic=True): 
        self.root = Node(root_state, MAX_PLAYER)
        self.search_depth = search_depth
        self.simulator = Game(*root_state)
        self.use_improved_heuristic = use_improved_heuristic

    # (Hint) Useful functions: 
    # self.simulator.current_state, self.simulator.set_state, self.simulator.move

    # TODO: build a game tree from the current node up to the given depth
    def build_tree(self, node = None, depth = 0):
        
        dq = deque([ (node, depth) ])
        
        while dq:
            
            current_node, current_depth = dq.popleft()

            if current_depth == 0:
                continue

            if current_node.player_type == MAX_PLAYER:
                
                # get the valid moves
                valid_moves = []
                board_copy = copy.deepcopy( current_node.state[0] )
                score_copy = copy.deepcopy( current_node.state[1] )

                for dir in range(4):

                    self.simulator.set_state(board_copy, score_copy)
                    moved = self.simulator.move(dir)
                    if moved:
                        valid_moves.append(dir)
                
                # creat child nodes
                for dir in valid_moves:
                    self.simulator.set_state(board_copy, score_copy)
                    self.simulator.move(dir)
                    new_board = self.simulator.current_state()[0]
                    new_score = copy.deepcopy( self.simulator.current_state()[1] )
                    child_node = Node((new_board, new_score), CHANCE_PLAYER)
                    current_node.children.append((child_node, dir))
                    dq.append((child_node, current_depth - 1)) 

            elif current_node.player_type == CHANCE_PLAYER:
                
                # get valid move
                valid_cells = []
                board_copy = copy.deepcopy( current_node.state[0] )
                score_copy = copy.deepcopy( current_node.state[1] )
                for i in range(len(board_copy)):
                    for j in range(len(board_copy[0])):
                        if board_copy[i][j] == 0:
                            valid_cells.append((i, j))
                
                # create child nodes
                for i, j in valid_cells:
                    new_board = copy.deepcopy(board_copy)
                    new_score = copy.deepcopy(score_copy)
                    new_board[i][j] = 2
                    child_node = Node((new_board, new_score), MAX_PLAYER)
                    current_node.children.append((child_node, (i, j)))
                    dq.append(( child_node, current_depth - 1 ))

            else:
                assert False


    # using monotonicity and smoothness
    def build_tree_better(self, node = None, depth = 0):
        dq = deque([ (node, depth) ])
        
        while dq:
            
            current_node, current_depth = dq.popleft()

            if current_depth == 0:
                continue

            if current_node.player_type == MAX_PLAYER:
                
                # get the valid moves
                valid_moves = []
                board_copy = copy.deepcopy( current_node.state[0] )
                score_copy = copy.deepcopy( current_node.state[1] )

                for dir in range(4):

                    self.simulator.set_state(board_copy, score_copy)
                    moved = self.simulator.move(dir)
                    if moved:
                        valid_moves.append(dir)
                
                # creat child nodes
                for dir in valid_moves:
                    self.simulator.set_state(board_copy, score_copy)
                    self.simulator.move(dir)
                    new_board = self.simulator.current_state()[0]
                    new_score = self.monotonicity(new_board) + self.smoothness(new_board)
                    child_node = Node((new_board, new_score), CHANCE_PLAYER)
                    current_node.children.append((child_node, dir))
                    dq.append((child_node, current_depth - 1)) 

            elif current_node.player_type == CHANCE_PLAYER:
                
                # get valid move
                valid_cells = []
                board_copy = copy.deepcopy( current_node.state[0] )
                score_copy = copy.deepcopy( current_node.state[1] )
                for i in range(len(board_copy)):
                    for j in range(len(board_copy[0])):
                        if board_copy[i][j] == 0:
                            valid_cells.append((i, j))
                
                # create child nodes
                for i, j in valid_cells:
                    new_board = copy.deepcopy(board_copy)
                    new_board[i][j] = 2
                    new_score = self.monotonicity(new_board) + self.smoothness(new_board)
                    child_node = Node((new_board, new_score), MAX_PLAYER)
                    current_node.children.append((child_node, (i, j)))
                    dq.append(( child_node, current_depth - 1 ))

            else:
                assert False


    
    # measure the monotonicity of the board
    def monotonicity(self, board):
   
        totals = [0, 0, 0, 0]  # Scores for the four directions

        board = np.array(board)
        log_board = np.zeros_like(board, dtype=float)  
        nonzero_indices = board > 0  
        log_board[nonzero_indices] = np.log2(board[nonzero_indices])

        # Up/Down Monotonicity
        for x in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while next < 4 and log_board[next][x] == 0:  
                    next += 1
                if next >= 4:  
                    break  

                currentValue = log_board[current][x]
                nextValue = log_board[next][x]

                if currentValue > nextValue:  
                    totals[0] += nextValue - currentValue 
                elif nextValue > currentValue:
                    totals[1] += currentValue - nextValue 

                current = next
                next += 1

        # Left/Right Monotonicity
        for y in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while next < 4 and log_board[y][next] == 0:  
                    next += 1
                if next >= 4:  
                    break  

                currentValue = log_board[y][current]
                nextValue = log_board[y][next]

                if currentValue > nextValue:
                    totals[2] += nextValue - currentValue  
                elif nextValue > currentValue:
                    totals[3] += currentValue - nextValue 

                current = next
                next += 1

        return max(totals[0], totals[1]) + max(totals[2], totals[3])


    # measure the smoothness of the board
    # make sure that adjacent tiles have smaller difference
    # here, we only check right and down direction
    def smoothness(self, board):

        smoothness_score = 0

        board = np.array(board)
        log_board = np.zeros_like(board, dtype=float)  
        nonzero_indices = board > 0  
        log_board[nonzero_indices] = np.log2(board[nonzero_indices])

        directions = [(0, 1), (1, 0)]  

        for x in range(4):
            for y in range(4):
                if log_board[x, y] > 0:  
                    value = log_board[x, y]

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy 

                        while 0 <= nx < 4 and 0 <= ny < 4 and log_board[nx, ny] == 0:
                            nx += dx
                            ny += dy

                        if 0 <= nx < 4 and 0 <= ny < 4:
                            neighbor_value = log_board[nx, ny]
                            smoothness_score -= abs(value - neighbor_value) 

        return smoothness_score




    # TODO: expectimax calculation.
    # Return a (best direction, expectimax value) tuple if node is a MAX_PLAYER
    # Return a (None, expectimax value) tuple if node is a CHANCE_PLAYER
    def expectimax(self, node = None):
        # TODO: delete this random choice but make sure the return type of the function is the same

        if node.is_terminal():
            return None, node.state[1]
        
        if node.player_type == MAX_PLAYER:
            best_score = -math.inf
            best_move = None

            for child, move in node.children:
                _, value = self.expectimax(child)
                if value > best_score:
                    best_score = value
                    best_move = move
            
            return best_move, best_score


        if node.player_type == CHANCE_PLAYER:
            total = 0

            for child, _, in node.children:
                _, value = self.expectimax(child)
                total += value

            return None, total / len(node.children)


    # Return decision at the root
    def compute_decision(self):
        if not self.use_improved_heuristic:
            self.build_tree(self.root, self.search_depth)
        else:
            self.build_tree_better(self.root, self.search_depth)
        
        direction, _ = self.expectimax(self.root)
        return direction

    def compute_decision_ec(self):
        return random.randint(0, 3)

