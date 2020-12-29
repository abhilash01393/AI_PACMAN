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
from game import Actions
from util import PriorityQueueLAS, PriorityQueueWithFunction, manhattanDistance

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
    fringe = util.Stack()  
    discovered = set() 
    fringe.push((problem.getStartState(), []))
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node = elem_pop[0], elem_pop[1]
    
        if problem.isGoalState(curr_node):
            break
        
        else:
            if curr_node not in discovered:   
                discovered.add(curr_node)    
                successors = problem.getSuccessors(curr_node)
                for successor in successors:

                    new_node, new_path = successor[0], successor[1]
                    fin_path = path_curr_node + [new_path]  
                    fringe.push((new_node, fin_path)) 
    final_path = path_curr_node

    return final_path
"""
   Is the exploration order what you would have expected?
   Yes, given it was running dfs search algo, the algorith gave the expected order by dfs.
   Does Pacman actually go to all the explored squares on his way to the goal?
   No
   
   Is this a least cost solution? 
   No, this not the least cost solution becasue DFS works in Stack Last In First Out fashion, i.e.,
   searching the deepest nodes first.
   """

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()   
    discovered = set()   
    fringe.push((problem.getStartState(), []))   
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node = elem_pop[0], elem_pop[1]
        
        if problem.isGoalState(curr_node):   
            break
        else:
            if curr_node not in discovered:  
                discovered.add(curr_node)    
                successors = problem.getSuccessors(curr_node)
                for successor in successors:
                    new_node, new_path = successor[0], successor[1]
                    fin_path = path_curr_node + [new_path]    
                    fringe.push((new_node, fin_path))   
    final_path = path_curr_node

    return final_path
"""
    Does BFS find a least cost solution? If so explain why ?
    Yes, BFS found the least cost solution with a total cost of 68 and 269
    nodes expanded because bfs implements in Queue fashion, searching the
    shallowest nodes.
    """

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()   
    discovered = set()   
    fringe.push((problem.getStartState(), [], 0), 0)   
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node, cost_curr_node = elem_pop[0], elem_pop[1], elem_pop[2]
        if problem.isGoalState(curr_node):   
            break
        else:
            if curr_node not in discovered:   
                discovered.add(curr_node)     
                successors = problem.getSuccessors(curr_node)
                for successor in successors:
                    new_node, new_path, new_cost = successor[0], successor[1], successor[2]
                    fin_path = path_curr_node + [new_path]  
                    fin_cost = cost_curr_node + new_cost     
                    fringe.push((new_node, fin_path, fin_cost), fin_cost) 
    final_path = path_curr_node

    return final_path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()    
    discovered = set()    
    fringe.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem) + 0)   
    while True:
        elem_pop = fringe.pop()
        curr_node, path_curr_node, cost_curr_node = elem_pop[0], elem_pop[1], elem_pop[2]
        if problem.isGoalState(curr_node):    
            break
        else:
            if curr_node not in discovered:    
                discovered.add(curr_node)     
                successors = problem.getSuccessors(curr_node)
                for successor in successors:
                    new_node, new_path, new_cost = successor[0], successor[1], successor[2]
                    fin_path = path_curr_node + [new_path]    
                    fin_cost = cost_curr_node + new_cost    
                    fringe.push((new_node, fin_path, fin_cost), fin_cost + heuristic(new_node, problem))  
    final_path = path_curr_node

    return final_path
# Dynamic A* search algorithm
def modaStarSearch(problem, heuristic=nullHeuristic):
    # this is the normal astar search primer function that initializes the frontier
    def aStarPath():
        def priorityFunction(node):
            state, actions_sequence, path_cost = node
            heuristic_cost = heuristic(state, problem)
            return path_cost+heuristic_cost
        frontier = util.PriorityQueueWithFunction(priorityFunction)
        return commonSearch(frontier)
    # planning implementation with a generic frontier
    def commonSearch(frontier):
        # as the start location changes every time an obstacle is found, using a dynamic start state class variable
        root = problem.dynamicStartState
        explored_set = set()
        actions_sequence = list()
        path_cost = 0
        frontier.push((root, actions_sequence, path_cost))
        while not frontier.isEmpty():
            parent, actions_sequence, path_cost = frontier.pop()
            if parent not in explored_set:
                if problem.getGoalState() == parent:
                    return actions_sequence+[(parent,None)]
                explored_set.add(parent)
                for successor in problem.getASuccessors(parent):
                    state, action, step_cost = successor
                    new_actions_sequence = actions_sequence[:]
                    new_actions_sequence += [(parent, action)]
                    cost = path_cost+step_cost
                    frontier.push((state, new_actions_sequence, cost))

    def planning():
        # generate path
        path = aStarPath()
        if len(path) == 1 and path[0][0] == problem.getGoalState(): 
            return True
        for index in range(len(path)-1):
            currentState, currentAction = path[index]
            nextState, _ = path[index+1]
            problem.finalPath.append((currentState, currentAction))
            print "--> " + str(nextState),
            # found an obstacle
            if problem.isObstacle(nextState):
                print "\nObstacle @ "+ str(nextState)
                print "Replanning..."
                problem.insertObstacle(nextState)
                # update the new start state
                problem.dynamicStartState = currentState
                return False
            elif nextState == problem.getGoalState():
                return True

    def main():
        problem.finalPath = []
        problem.dynamicStartState = problem.getStartState()
        stop = False
        print 'The goal position is', problem.getGoalState()
        print "The path is: "
        print problem.dynamicStartState, 
        # do planning till the pacman reach the goal state.
        while (problem.dynamicStartState != problem.getGoalState())  and not stop:
            stop = planning()  
        problem.finalPath.append((problem.getGoalState(), None))
        print "\nDone Planning"
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index+1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)
        problem.drawObstacles()
        problem.printPath(states)
        print('Size of the Layout: ', str(problem.height)+'x'+str(problem.width))
        print('Path Length: ', len(actions))
        print('Number of obstacles: ', len(problem.obstacles))
        return actions

    return main()

# The life long A* search implementation
def lifeLongAStarSearch(problem, heuristic):
    
    # function directly implemented from the paper   
    def calculateKey(state):
        g_rhs = min(problem.g[state], problem.rhs[state])
        return (g_rhs + heuristic(state, problem), g_rhs)
    
    # function directly implemented from the paper   
    def initialize():
        for state in problem.getStates():
            problem.rhs[state] = float('inf')
            problem.g[state] = float('inf')
        problem.rhs[problem.dynamicStartState] = 0
        problem.U.insert(problem.dynamicStartState, calculateKey(problem.dynamicStartState))
    
    # function directly implemented from the paper   
    def updateVertex(u):
        if u != problem.dynamicStartState:
            prevKeys = [float('inf')]
            for successor, _, cost in problem.getASuccessors(u):
                prevKeys.append(problem.g[successor]+cost)
            problem.rhs[u] = min(prevKeys)
        problem.U.remove(u)
        if problem.g[u] != problem.rhs[u]:
            problem.U.insert(u, calculateKey(u))
    
    # function directly implemented from the paper   
    def computeShortestPath():
        goal = problem.getGoalState()
        while problem.U.topKey() < calculateKey(goal) or problem.rhs[goal] != problem.g[goal]:
            u = problem.U.pop()
            if problem.g[u] > problem.rhs[u]:
                problem.g[u] = problem.rhs[u]
                # the successor function produces a tuple of state, action, cost values
                for successor, _, _ in problem.getASuccessors(u):
                    updateVertex(successor)
            else:
                problem.g[u] = float('inf')
                updateVertex(u)
                for successor, _, _ in problem.getASuccessors(u):
                    updateVertex(successor)
    
    # After computing the shortest path the g values are updated.
    # From goal to start we will follow the least g value among 
    # the successors and get the shortest path. 
    def shortestPath():
        path = []
        state = (problem.getGoalState(), None)
        path.append(state)
        while state[0] != problem.dynamicStartState:
            minimum = float('inf')
            for successor, action, _ in problem.getASuccessors(state[0]):
                if minimum > problem.g[successor]:
                    minimum = problem.g[successor]
                    # since we are going in reverse direction, we need to reverse the actions.
                    state = (successor, Actions.reverseDirection(action))
            path.append(state)
        # reversing the direction path from start to goal
        return path[::-1]

    def planning():
        path = shortestPath()
        if len(path) == 1 and path[0][0] == problem.getGoalState(): 
            return True
        for index in range(len(path)-1):
            currentState, currentAction = path[index]
            nextState, _ = path[index+1]
            problem.finalPath.append((currentState, currentAction))
            print "--> " + str(nextState),
            if problem.isObstacle(nextState):
                print "\nObstacle @ "+ str(nextState)
                print "Replanning..."
                problem.insertObstacle(nextState)
                updateVertex(nextState)
                problem.dynamicStartState = currentState
                return False
            elif nextState == problem.getGoalState():
                return True

    def main():
        # initializing 
        problem.U = util.PriorityQueueLAS()
        problem.g = {}
        problem.rhs = {}
        problem.finalPath = []
        problem.dynamicStartState = problem.getStartState()
        initialize()
        stop = False
        print 'The goal position is', problem.getGoalState()
        print "The path is: "
        print problem.dynamicStartState, 
        while (problem.dynamicStartState != problem.getGoalState())  and not stop:
            initialize()
            computeShortestPath()
            stop = planning()  
        problem.finalPath.append((problem.getGoalState(), None))
        print "\nDone Planning"
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index+1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)
        problem.drawObstacles()
        problem.printPath(states)
        print('Path Length: ', len(actions))
        print('Size of the Layout: ', str(problem.height)+'x'+str(problem.width))
        print('Number of obstacles: ', len(problem.obstacles))
        return actions
    return main()

def dStarSearch(problem, heuristic):
    
    # function directly implemented from the paper   
    def calculateKey(state):
        g_rhs = min(problem.g[state], problem.rhs[state])
        return (g_rhs + util.manhattanDistance(state, problem.s['start']) + problem.k['m'], g_rhs)
    
    # function directly implemented from the paper   
    def initialize():
        problem.U = util.PriorityQueueLAS()
        problem.g = {}
        problem.rhs = {}
        problem.k = {}
        problem.k['m'] = 0
        problem.s = {}
        problem.s['start'] = problem.getStartState()
        problem.s['goal'] = problem.getGoalState()
        for state in problem.getStates():
            problem.rhs[state] = float('inf')
            problem.g[state] = float('inf')
        problem.rhs[problem.s['goal']] = 0
        problem.U.insert(problem.s['goal'], calculateKey(problem.s['goal']))
    
    # function directly implemented from the paper   
    def updateVertex(u):
        if u != problem.s['goal']:
            prevKeys = [float('inf')]
            for successor, _, cost in problem.getASuccessors(u):
                prevKeys.append(problem.g[successor]+cost)
            problem.rhs[u] = min(prevKeys)
        problem.U.remove(u)
        if problem.g[u] != problem.rhs[u]:
            problem.U.insert(u, calculateKey(u))
    
    # function directly implemented from the paper   
    def computeShortestPath():
        while problem.rhs[problem.s['start']] != problem.g[problem.s['start']] or problem.U.topKey() < calculateKey(problem.s['start']):
            problem.k['old'] = problem.U.topKey()
            u = problem.U.pop()
            if problem.k['old'] < calculateKey(u):
                problem.U.insert(u, calculateKey(u))
            elif problem.g[u] > problem.rhs[u]:
                problem.g[u] = problem.rhs[u]
                # the successor function produces a tuple of state, action, cost values
                for successor, _, _ in problem.getASuccessors(u):
                    updateVertex(successor)
            else:
                problem.g[u] = float('inf')
                updateVertex(u)
                for successor, _, _ in problem.getASuccessors(u):
                    updateVertex(successor)

    def main():
        initialize()
        problem.finalPath = []
        problem.s['last'] = problem.s['start']
        problem.dynamicAction = None
        computeShortestPath()
        print 'The goal position is', problem.getGoalState()
        print "The path is: "
        print problem.s['start'], 
        while (problem.s['start'] != problem.s['goal']):
            if problem.g[problem.s['start']] == float('inf'):
                return []
            minimum = float('inf')
            problem.s['successor'] = None
            for successor, action, cost in problem.getASuccessors(problem.s['start']):
                updatedCost = problem.g[successor]+cost
                if updatedCost < minimum:
                    minimum = updatedCost
                    problem.s['successor'] = successor
                    problem.dynamicAction = action
            print "--> " + str(problem.s['successor']),
            if problem.isObstacle(problem.s['successor']):
                print "\nObstacle @ "+ str(problem.s['successor'])
                print "Replanning..."
                problem.insertObstacle(problem.s['successor'])
                problem.k['m'] += util.manhattanDistance(problem.s['last'], problem.s['start'])
                problem.s['last'] = problem.s['start']
                updateVertex(problem.s['successor'])
                computeShortestPath()
            else:
                problem.finalPath.append((problem.s['start'], problem.dynamicAction))
                problem.s['start'] = problem.s['successor']
        problem.finalPath.append((problem.s['goal'], None))
        print "\nDone Planning"
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index+1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)
        problem.drawObstacles()
        problem.printPath(states)
        print('Size of the Layout: ', str(problem.height)+'x'+str(problem.width))
        print('Path Length: ', len(actions))
        print('Number of obstacles: ', len(problem.obstacles))
        return actions

    return main()

 # Abbreviations
astar = aStarSearch
modastar = modaStarSearch
lastar = lifeLongAStarSearch
dstar = dStarSearch

"""
    What happens on openMaze for the various search strategies?
    Search              Total Cost           # of Nodes Expanded
    DFS                  298                     576
    BFS                  54                      682
    A*                   54                      682
    UCS                  54                      682
    Mod A*
    LifeLong A*
    D* Lit
"""

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
