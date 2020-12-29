# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()  # Fringe (Stack) to store the nodes along with their paths
    discovered = set()  # A set to maintain all the visited nodes
    fringe.push((problem.getStartState(), []))  # Pushing (Node, [Path from start-node till 'Node']) to the fringe
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node = elem_pop[0], elem_pop[1]
    
        if problem.isGoalState(curr_node):  # Exit on encountering goal node
            break
        
        else:
            if curr_node not in discovered:   # Skipping already visited nodes
                discovered.add(curr_node)     # Adding newly encountered nodes to the set of visited nodes
                successors = problem.getSuccessors(curr_node)
                for successor in successors:

                    new_node, new_path = successor[0], successor[1]
                    fin_path = path_curr_node + [new_path]  # Computing path of child node from start node
                    fringe.push((new_node, fin_path)) # Pushing ('Child Node',[Full Path]) to the fringe

    return path_curr_node

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()   # Fringe (Queue) to store the nodes along with their paths
    discovered = set()   # A set to maintain all the visited nodes
    fringe.push((problem.getStartState(), []))   # Pushing (Node, [Path from start-node till 'Node']) to the fringe
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node = elem_pop[0], elem_pop[1]
        
        if problem.isGoalState(curr_node):   # Exit on encountering goal node
            break
        else:
            if curr_node not in discovered:  # Skipping already visited nodes
                discovered.add(curr_node)    # Adding newly encountered nodes to the set of visited nodes
                successors = problem.getSuccessors(curr_node)
                for successor in successors:
                    new_node, new_path = successor[0], successor[1]
                    fin_path = path_curr_node + [new_path]    # Computing path of child node from start node
                    fringe.push((new_node, fin_path))   # Pushing ('Child Node',[Full Path]) to the fringe

    return path_curr_node

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()   # Fringe (Priority Queue) to store the nodes along with their paths
    discovered = set()   # A set to maintain all the visited nodes
    fringe.push((problem.getStartState(), [], 0), 0)   # Pushing (Node, [Path from start-node till 'Node'], Culmulative backward cost till 'Node') to the fringe. In this case, we are using culmulative backward cost as a factor based on which priority is decided.
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node, cost_curr_node = elem_pop[0], elem_pop[1], elem_pop[2]
        if problem.isGoalState(curr_node):   # Exit on encountering goal node
            break
        else:
            if curr_node not in discovered:   # Skipping already visited nodes
                discovered.add(curr_node)     # Adding newly encountered nodes to the set of visited nodes
                successors = problem.getSuccessors(curr_node)
                for successor in successors:
                    new_node, new_path, new_cost = successor[0], successor[1], successor[2]
                    fin_path = path_curr_node + [new_path]  # Computing path of child node from start node
                    fin_cost = cost_curr_node + new_cost     # Computing culmulative backward cost of child node from start node
                    fringe.push((new_node, fin_path, fin_cost), fin_cost)   # Pushing (Node, [Path], Culmulative backward cost) to the fringe.

    return path_curr_node

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()    # Fringe (Priority Queue) to store the nodes along with their paths
    discovered = set()    # A set to maintain all the visited nodes
    fringe.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem) + 0)    # Pushing (Node, [Path from start-node till 'Node'], Culmulative backward cost till 'Node') to the fringe. In this case, we are using the sum of culmulative backward cost and the heutristic of the node as a factor based on which priority is decided.
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node, cost_curr_node = elem_pop[0], elem_pop[1], elem_pop[2]
        if problem.isGoalState(curr_node):    # Exit on encountering goal node
            break
        else:
            if curr_node not in discovered:    # Skipping already visited nodes
                discovered.add(curr_node)     # Adding newly encountered nodes to the set of visited nodes
                successors = problem.getSuccessors(curr_node)
                for successor in successors:
                    new_node, new_path, new_cost = successor[0], successor[1], successor[2]
                    fin_path = path_curr_node + [new_path]    # Computing path of child node from start node
                    fin_cost = cost_curr_node + new_cost    # Computing culmulative backward cost of child node from start node
                    fringe.push((new_node, fin_path, fin_cost), fin_cost + heuristic(new_node, problem))    # Pushing (Node, [Path], Culmulative backward cost) to the fringe.

    return path_curr_node


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
