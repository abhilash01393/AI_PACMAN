# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from game import Agent
import random,util
from heuristics import *
import math

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()
        min_distance_f = -1
        for food in newFood:
            distance = util.manhattanDistance(newPos, food)
            if min_distance_f >= distance or min_distance_f == -1:
                min_distance_f = distance        
        g_distance = 1
        prox_ghost = 0
        for g_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, g_state)
            g_distance += distance
            if distance <= 1:
                prox_ghost += 1
        newScore = successorGameState.getScore() + (1 / float(min_distance_f)) - (1 / float(g_distance)) - prox_ghost
        return newScore
"""
The Pacman fared with 0.9 win rate and average score = 1012 on n=10 games with 1 default ghost
and 0.9 win rate with average score = 1311 on n=10 games with 1 Directional ghost.

The Pacman fared with 0.7 win rate and average score = 979 on n=10 games with 2 default ghosts
and never won, i.e., 0 win rate with average score = 41 on n=10 games with 2 Directional ghosts.
"""
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def Mini_Max(agent, depth, gameState):
            agent_max = []
            agent_min = []
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                for newState in gameState.getLegalActions(agent):
                    agent_max.append(Mini_Max(1, depth, gameState.generateSuccessor(agent, newState)))
                return max(agent_max)
            else:
                newAgent = agent + 1
                if gameState.getNumAgents() == newAgent:
                    newAgent = 0
                if newAgent == 0:
                   depth = depth + 1
                for newState in gameState.getLegalActions(agent):
                    agent_min.append(Mini_Max(newAgent, depth, gameState.generateSuccessor(agent, newState)))
                return min(agent_min )
        maxi = float("-inf")
        action = Directions.WEST
        for state_agent in gameState.getLegalActions(0):
            utility = Mini_Max(1, 0, gameState.generateSuccessor(0, state_agent))
            if utility > maxi or maxi == float("-inf"):
                maxi = utility
                action = state_agent
            new_action = action
        return new_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_val(agent, depth, game_state, alpha, beta):
            v = float("-inf")
            for newState in game_state.getLegalActions(agent):
                value = pruning_alpha_beta(1, depth, game_state.generateSuccessor(agent, newState), alpha, beta)
                v = max(v, value)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_val(agent, depth, game_state, alpha, beta):
            v = float("inf")

            new_agent = agent + 1
            if game_state.getNumAgents() == new_agent:
                new_agent = 0
            if new_agent == 0:
                depth = depth + 1

            for newState in game_state.getLegalActions(agent):
                value = pruning_alpha_beta(new_agent, depth, game_state.generateSuccessor(agent, newState), alpha, beta)
                v = min(v, value)
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def pruning_alpha_beta(agent, depth, game_state, alpha, beta):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)

            if agent == 0:
                return max_val(agent, depth, game_state, alpha, beta)
            else:
                return min_val(agent, depth, game_state, alpha, beta)

       
        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")
        for state_agent in gameState.getLegalActions(0):
            value_g = pruning_alpha_beta(1, 0, gameState.generateSuccessor(0, state_agent), alpha, beta)
            if value_g > utility:
                utility = value_g
                action = state_agent
            if utility > beta:
                return utility
            alpha = max(alpha, utility)
            new_action = action
        return new_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expecti_Max(agent, depth, gameState):
            agent_max = []
            sum_expect = 0
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                for newState in gameState.getLegalActions(agent):
                    agent_max.append(expecti_Max(1, depth, gameState.generateSuccessor(agent, newState)))
                return max(agent_max)
            else: 
                newAgent = agent + 1
                if gameState.getNumAgents() == newAgent:
                    newAgent = 0
                if newAgent == 0:
                    depth += 1
                for newState in gameState.getLegalActions(agent):
                    sum_expect = sum_expect + expecti_Max(newAgent, depth, gameState.generateSuccessor(agent, newState))
                avg_expect = sum_expect  / float(len(gameState.getLegalActions(agent)))
                return avg_expect

        
        maximum = float("-inf")
        action = Directions.WEST
        for state_agent in gameState.getLegalActions(0):
            utility = expecti_Max(1, 0, gameState.generateSuccessor(0, state_agent))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = state_agent
            new_action = action
        return new_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFood = newFood.asList()
    min_distance_f = -1
    for food in newFood:
        distance = util.manhattanDistance(newPos, food)
        if min_distance_f >= distance or min_distance_f == -1:
            min_distance_f = distance

    g_distance = 1
    prox_ghost = 0
    for g_state in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, g_state)
        g_distance += distance
        if distance <= 1:
            prox_ghost += 1
    newCapsule = currentGameState.getCapsules()
    numCapsules = len(newCapsule)

    newScore = currentGameState.getScore() + (1 / float(min_distance_f)) - (1 / float(g_distance)) - prox_ghost - numCapsules
    return newScore
	
## Node class with the attributes of each node
class Node:
	def __init__(self, state, parent, actionList, children,visited,score,action):
		self.parent = parent
		self.actionList = actionList
		self.children = children
		self.state = state
		self.visited=visited
		self.score=score
		self.action=action


class MCTSAgent(Agent):
	
	# Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    total=0


    def traverse_choice(self,node):
        len_actionList=len(node.actionList)
        r=random.randint(0, len_actionList-1)
        random_choice=node.actionList[r]
        node.actionList.pop(r)
        child_state=node.state[0].generatePacmanSuccessor(random_choice)
        if child_state is None:
            return None
        if child_state.isWin() or child_state.isLose():
            child_actionList=[]
    	else:
        	child_actionList=child_state.getLegalPacmanActions()
        s=node.state
        s.append(random_choice)    #always storing the root and all actions leading to this node

        #child.score value is initially assigned 1 and later updated
        child=Node(s, node,child_actionList,[],1,1,random_choice)   #visited node here is assigned 1 and is not considered during update
        child.score=self.rollout(child,child_state)    #child.score value is updated correctly
        
        self.update(child)
        node.children.append(child)
        return;

    def rollout(self,node,current_state):
        possible=node.actionList
        maximum_value=-1 * float('inf')
        for i in range(5):
        	if len(possible)>1:
        		current_state=current_state.generatePacmanSuccessor(possible[random.randint(0, len(possible)-1)])
        	elif len(possible)==1:
        		current_state=current_state.generatePacmanSuccessor(possible[0])

        	else:
        		try:
        			current_state=current_state.generatePacmanSuccessor(possible[random.randint(0, len(possible)-1)])
        		except:
        			return 0

        	if current_state is None or current_state.isLose():
        		return 0
        	else:
        		possible=current_state.getLegalPacmanActions()
        		score=normalizedScoreEvaluation(node.state[0], current_state)
        return score

#backpropagation and updation of nodes
    def update(self,node):
    	while node.parent is not None:
    		node = node.parent
    		node.visited+=1
    		node.score+=node.score
    		self.total=node.visited
    		
#To compare children of nodes to give back the best node
    def compare(self,node):
    	c=1
    	maximum_value=-1 * float('inf')                #-(infinity) is the initial value of maximum_value
    	if len(node.children)>0:
	    	for i in node.children:
	    		val = (i.score/i.visited) + 1 * math.sqrt(2*math.log(self.total)/float(i.visited))
	    		if val>maximum_value:
	    			maximum_value=val
	    			selected_child=i
	    	return selected_child
    	return None




    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        node=Node([state], None, state.getLegalPacmanActions(),[],0,0,Directions.STOP)
        root=node
    	flag=0

    	#Proper traversal of nodes
    	while True:
    		if node is None:
    			flag=1
    			break
    		if len(node.actionList)>0:
    			while len(node.actionList)>0:
    				self.traverse_choice(node)
    			flag=2
    		else:
	    		node=self.compare(node)
	    	if flag==2:
	    		node=root
	    		flag=0

	    #To give most visited node
        max_value=0
        for i in root.children:
        	if i.visited>=max_value:
        		max_value=i.visited

        ans=[]
        for i in root.children:
        	if i.visited==max_value:
        		ans.append(i.action)

        if len(ans)>0:
            return ans[random.randint(0,len(ans)-1)]

        return Directions.STOP
    
    def normalizedScoreEvaluation(rootState, currentState):
        rootEval = betterEvaluationFunction(rootState);
        currentEval = betterEvaluationFunction(currentState);
        return (currentEval - rootEval) / 1000.0;

# Abbreviation
better = betterEvaluationFunction
