
import sys
import puzz
import pdqpq


GOAL_STATE = puzz.EightPuzzleBoard("012345678")


def solve_puzzle(start_state, flavor):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state (EightPuzzleBoard): the start state for the search
        flavor (str): tag that indicate which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
        'path' - list of 2-tuples representing the path from the start to the goal state (both 
            included).  Each entry is a (str, EightPuzzleBoard) pair indicating the move and 
            resulting successor state for each action.  Omitted if the search fails.
        'path_cost' - the total cost of the path, taking into account the costs associated with 
            each state transition.  Omitted if the search fails.
        'frontier_count' - the number of unique states added to the search frontier at any point 
            during the search.
        'expanded_count' - the number of unique states removed from the frontier and expanded 
            (successors generated)

    """
    if flavor.find('-') > -1:
        strat, heur = flavor.split('-')
    else:
        strat, heur = flavor, None

    if strat == 'bfs':
        return BreadthFirstSolver().solve(start_state)
    elif strat == 'ucost':
        return UniformCostSearch().solve(start_state)
    elif strat == 'greedy':
        return GreedySearch().solve(start_state,heur)
    elif strat == 'astar':
        return AStar().solve(start_state,heur)
    else:
        raise ValueError("Unknown search flavor '{}'".format(flavor))

class AStar:
    def __init__(self):
        self.goal = GOAL_STATE
        self.parents = {}  # state -> parent_state
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        self.frontier_count = 0 
        self.expanded_count = 0
        self.heuristic = None

    def solve(self,start_state,heur):
        self.heuristic = heur
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()

            if node == self.goal:
                return self.get_results_dict(node)
            
            succs = self.expand_node(node)

            for move, succ in succs.items():
                
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ)
                elif (node in self.frontier) and (self.frontier.get(succ) > self.astar_cost(succ)):
                    self.add_to_frontier(succ)
        return self.get_results_dict(None)

    def astar_cost(self,node):
        total = self.get_cost(node) + self.heur_cost(node)
        return total

    def heur_cost(self, node):
        if self.heuristic == 'h1':
            return self.h1(node)
        elif self.heuristic == 'h2':
            return self.h2(node)["distance"]
        elif self.heuristic == 'h3':
            return self.h3(node)
        return None

    def h1(self,state):
        goal = GOAL_STATE._board
        cur = state._board
        miss = 0

        for i in range(0, len(goal)):
            if (cur[i] != goal[i]) and (int(cur[i]) != 0):
                miss += 1
        
        return miss

    def h2(self,state):
        positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
        goal = [(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
        cur = [None] * 8
        zero = state._board.index('0')
        i=0
        dist = 0
        vals = []
        for item in positions:

            if i != zero:        
                cur[int(state._board[i]) -1] = positions[i]
            i+=1
              
        for i in range(0,8):
            val = abs(goal[i][0] - cur[i][0]) + abs(goal[i][1] - cur[i][1])
            vals. append(val)
            dist += val
        
        return {"distance": dist, "list": vals }

    def h3(self, state):
        man2 = 0
        vals = self.h2(state)["list"]
        for i in range(0,len(vals)):
            n = i + 1
            newVal = (n*n) * vals[i]
            man2+=newVal
        return man2



    def add_to_frontier(self, node): # new priority
        if node not in self.frontier:
            self.frontier_count += 1
        self.frontier.add(node,self.astar_cost(node))

    def expand_node(self, node):
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def get_results_dict(self, state):

        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def get_path(self, state):
        
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost



class GreedySearch:
    def __init__(self):
        self.goal = GOAL_STATE
        self.parents = {}  # state -> parent_state
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        self.frontier_count = 0 
        self.expanded_count = 0
        self.heuristic = None 

    def solve(self,start_state,heur):
        self.heuristic = heur
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()

            if node == self.goal:
                return self.get_results_dict(node)
            
            succs = self.expand_node(node)

            for move, succ in succs.items():
                
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ)
                elif (node in self.frontier) and (self.frontier.get(succ) > self.heur_cost(succ)):
                    self.add_to_frontier(succ)
        return self.get_results_dict(None)

    def heur_cost(self, node):
        if self.heuristic == 'h1':
            return self.h1(node)
        elif self.heuristic == 'h2':
            return self.h2(node)["distance"]
        elif self.heuristic == 'h3':
            return self.h3(node)
        return None

    def h1(self,state):
        goal = GOAL_STATE._board
        cur = state._board
        miss = 0

        for i in range(0, len(goal)):
            if (cur[i] != goal[i]) and (int(cur[i]) != 0):
                miss += 1
        
        return miss

    def h2(self,state):
        positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
        goal = [(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
        cur = [None] * 8
        zero = state._board.index('0')
        i=0
        dist = 0
        vals = []
        for item in positions:

            if i != zero:        
                cur[int(state._board[i]) -1] = positions[i]
            i+=1
              
        for i in range(0,8):
            val = abs(goal[i][0] - cur[i][0]) + abs(goal[i][1] - cur[i][1])
            vals.append(val)
            dist += val
        
        return {"distance": dist, "list": vals }

    def h3(self, state):
        man2 = 0
        vals = self.h2(state)["list"]
        for i in range(0,len(vals)):
            n = i + 1
            newVal = (n*n) * vals[i]
            man2+=newVal
        return man2


    def add_to_frontier(self, node): # new priority
        if node not in self.frontier:
            self.frontier_count += 1
        self.frontier.add(node,self.heur_cost(node))

    def expand_node(self, node):
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def get_results_dict(self, state):

        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def get_path(self, state):
        
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost

class UniformCostSearch:

    def __init__(self):
        self.goal = GOAL_STATE
        self.parents = {}  # state -> parent_state
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        self.frontier_count = 0 
        self.expanded_count = 0  
    
    def solve(self, start_state):
        self.parents[start_state] = None # start stae has no parent
        self.add_to_frontier(start_state) # start state has  no priority !! we could add the get cost function to add_to_frontier function !!
        while not self.frontier.is_empty():
            
            node = self.frontier.pop() # gets item with higest priority
            

            if node == self.goal:
                return self.get_results_dict(node) # checks if current node is goal state

            succs = self.expand_node(node) # if not we expand


            for move, succ in succs.items(): # fro each successor from current node

                if (succ not in self.frontier) and (succ not in self.explored): # if we havent seen thsi state yet... 
                    self.parents[succ] = node # add the previous node as a parent 
                    
                    self.add_to_frontier(succ)# add new node to frontier with calculated priority

                elif (succ in self.frontier) and (self.frontier.get(succ) > self.get_cost(succ)): # if we already have it in frontier but the new cost of reaching state is lower than what we alrady have
                    self.parents[succ] = node
                    self.frontier.add(succ)# update the cost

        return self.get_results_dict(None)

    def add_to_frontier(self, node):
        if node not in self.frontier:
            self.frontier_count += 1
        self.frontier.add(node,self.get_cost(node))

    def expand_node(self, node):
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def get_results_dict(self, state):

        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def get_path(self, state):
        
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost

class BreadthFirstSolver:
    """Implementation of Breadth-First Search based puzzle solver"""

    def __init__(self):
        self.goal = GOAL_STATE
        self.parents = {}  # state -> parent_state
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        self.frontier_count = 0  # increment when we add something to frontier
        self.expanded_count = 0  # increment when we pull something off frontier and expand
    
    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node

                    # BFS checks for goal state _before_ adding to frontier
                    if succ == self.goal:
                        return self.get_results_dict(succ)
                    else:
                        self.add_to_frontier(succ)

        # if we get here, the search failed
        return self.get_results_dict(None) 

    def add_to_frontier(self, node):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def get_results_dict(self, state):
        """Construct the output dictionary for solve_puzzle()
        
        Args:
            state (EightPuzzleBoard): final state in the search tree
        
        Returns:
            A dictionary describing the search performed (see solve_puzzle())

        """
        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def get_path(self, state):
        """Return the solution path from the start state of the search to a target.
        
        Results are obtained by retracing the path backwards through the parent tree to the start
        state for the serach at the root.
        
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            A list of EightPuzzleBoard objects representing the path from the start state to the
            target state

        """
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 

        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path

        """
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost


def print_table(flav__results, include_path=False):
    """Print out a comparison of search strategy results.

    Args:
        flav__results (dictionary): a dictionary mapping search flavor tags result statistics. See
            solve_puzzle() for detail.
        include_path (bool): indicates whether to include the actual solution paths in the table

    """
    result_tups = sorted(flav__results.items())
    c = len(result_tups)
    na = "{:>12}".format("n/a")
    rows = [  # abandon all hope ye who try to modify the table formatting code...
        "flavor  " + "".join([ "{:>12}".format(tag) for tag, _ in result_tups]),
        "--------" + ("  " + "-"*10)*c,
        "length  " + "".join([ "{:>12}".format(len(res['path'])) if 'path' in res else na 
                                for _, res in result_tups ]),
        "cost    " + "".join([ "{:>12,}".format(res['path_cost']) if 'path_cost' in res else na 
                                for _, res in result_tups ]),
        "frontier" + ("{:>12,}" * c).format(*[res['frontier_count'] for _, res in result_tups]),
        "expanded" + ("{:>12,}" * c).format(*[res['expanded_count'] for _, res in result_tups])
    ]
    if include_path:
        rows.append("path")
        longest_path = max([ len(res['path']) for _, res in result_tups if 'path' in res ] + [0])
        print("longest", longest_path)
        for i in range(longest_path):
            row = "        "
            for _, res in result_tups:
                if len(res.get('path', [])) > i:
                    move, state = res['path'][i]
                    row += " " + move[0] + " " + str(state)
                else:
                    row += " "*12
            rows.append(row)
    print("\n" + "\n".join(rows), "\n")


def get_test_puzzles():
    """Return sample start states for testing the search strategies.
    
    Returns:
        A tuple containing three EightPuzzleBoard objects representing start states that have an
        optimal solution path length of 3-5, 10-15, and >=25 respectively.
    
    """ 
    # Note: test cases can be hardcoded, and are not required to be programmatically generated.
    #
    # fill in function body here
    #    

    small = puzz.EightPuzzleBoard("142358607")
    medium = puzz.EightPuzzleBoard("504168723")
    large = puzz.EightPuzzleBoard("671308245")

    return (small, medium, large) 


############################################

if __name__ == '__main__':
    
    # parse the command line args
    start = puzz.EightPuzzleBoard(sys.argv[1])
    if sys.argv[2] == 'all':
        flavors = ['bfs', 'ucost', 'greedy-h1', 'greedy-h2', 
                   'greedy-h3', 'astar-h1', 'astar-h2', 'astar-h3']
    else:
        flavors = sys.argv[2:]

    # run the search(es)
    results = {}
    for flav in flavors:
        print("solving puzzle {} with {}".format(start, flav))
        results[flav] = solve_puzzle(start, flav)

    print_table(results, include_path=True)  # change to True to see the paths!


