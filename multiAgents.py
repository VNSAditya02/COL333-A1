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
import random, util


from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        newcap = successorGameState.getCapsules()
        Food2 = currentGameState.getFood()
        "*** YOUR CODE HERE ***"
        # print(newPos)
        # print(newFood.asList())
        food_locations = newFood.asList()
        food_locations2 = Food2.asList()
        #caplist = newcap.asList()
        min_distance = float('inf')
        max_distance = float('-inf')
        Min = float('-inf')
        Max = float('inf')
        index = -1
        distance = 0
        mind =  float('-inf')
        for i in range(len(food_locations2)):
            food = food_locations2[i]
            temp = manhattanDistance(newPos, food)
            temp2 = manhattanDistance(currentGameState.getPacmanPosition(),food)
            distance += temp
            # distance += (food[0] - newPos[0])**2 + (food[1] - newPos[1])**2
            min_distance = min(temp, min_distance)
            mind = min(temp2,mind)
            max_distance = max(temp, min_distance)

        ghost_distance = [0]*len(newGhostStates)
        for i in range(len(newGhostStates)):
            ghost = newGhostStates[i].getPosition()
            ghost_distance[i] = manhattanDistance(newPos, ghost)
            # ghost_distance[i] = (ghost[0] - newPos[0])**2 + (ghost[1] - newPos[1])**2
        # print(10*(1/(min(ghost_distance) + 0.0001)))
        #score = 100/(len(food_locations) + 1) + 5*len(food_locations)*(1/(distance + 1)) + (5/(max_distance + 1)) + (5/(min_distance + 1 - 10*(min(ghost_distance)) - (sum(ghost_distance)/len(ghost_distance) + 1)))
        #score = 2**(-len(food_locations)/(100*((min(ghost_distance) + sum(ghost_distance)/len(ghost_distance) + 1)))) #+ 2**((min_distance + 1))
        
        score = (9/(1+min_distance))  - (10/(1+min(ghost_distance))) #- (1/(0.001+max(ghost_distance)))
        if(min(ghost_distance)<=1 and max(newScaredTimes)!=0):
            score = Min
            #print("Hello")
        if(max(newScaredTimes)!=0):
            maxcap = max(newScaredTimes)
            #print(maxcap)
            if(max!=0):
                score = score + (2**(maxcap))*(9/(1+min_distance))
        
        score = successorGameState.getScore() + score 
        #print(newcap)

        return score 

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
    def minimax(self, gameState, agentIndex, numAgents, isMax, depth, isStart):
        if(depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if(isMax):
            max_val = float('-inf')
            idx = -1
            for i in range(len(gameState.getLegalActions(agentIndex))):
                temp = self.minimax(gameState.generateSuccessor(agentIndex, gameState.getLegalActions(agentIndex)[i]), agentIndex + 1, numAgents, False, depth, False)
                if(max_val < temp):
                    max_val = temp
                    idx = i
            if(isStart):
                return gameState.getLegalActions(agentIndex)[idx]
            else:
                return max_val
        else:
            if(agentIndex == numAgents - 1):
                min_val = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    min_val = min(min_val, self.minimax(gameState.generateSuccessor(agentIndex, action), 0, numAgents, True, depth + 1, False))
                return min_val
            else:
                min_val = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    min_val = min(min_val, self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, numAgents, False, depth, False))
                return min_val

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        x = self.minimax(gameState, 0, gameState.getNumAgents(), True, 0, True)
        return x
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, gameState, agentIndex, numAgents, isMax, depth, isStart, alpha, beta):
        if(depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if(isMax):
            max_val = float('-inf')
            idx = -1
            for i in range(len(gameState.getLegalActions(agentIndex))):
                temp = self.alphaBeta(gameState.generateSuccessor(agentIndex, gameState.getLegalActions(agentIndex)[i]), agentIndex + 1, numAgents, False, depth, False, alpha, beta)
                if(max_val < temp):
                    max_val = temp
                    idx = i
                if(max_val > beta):
                    return max_val
                alpha = max(alpha, max_val)
            if(isStart):
                return gameState.getLegalActions(agentIndex)[idx]
            else:
                return max_val
        else:
            if(agentIndex == numAgents - 1):
                min_val = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    min_val = min(min_val, self.alphaBeta(gameState.generateSuccessor(agentIndex, action), 0, numAgents, True, depth + 1, False, alpha, beta))
                    if(min_val < alpha):
                        return min_val
                    beta = min(beta, min_val)
                return min_val
            else:
                min_val = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    min_val = min(min_val, self.alphaBeta(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, numAgents, False, depth, False, alpha, beta))
                    if(min_val < alpha):
                        return min_val
                    beta = min(beta, min_val)
                return min_val

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        x = self.alphaBeta(gameState, 0, gameState.getNumAgents(), True, 0, True, float('-inf'), float('inf'))
        return x
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, agentIndex, numAgents, isMax, depth, isStart):
        if(depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        if(isMax):
            max_val = float('-inf')
            idx = -1
            for i in range(len(gameState.getLegalActions(agentIndex))):
                temp = self.expectimax(gameState.generateSuccessor(agentIndex, gameState.getLegalActions(agentIndex)[i]), agentIndex + 1, numAgents, False, depth, False)
                if(max_val < temp):
                    max_val = temp
                    idx = i
            if(isStart):
                return gameState.getLegalActions(agentIndex)[idx]
            else:
                return max_val
        else:
            if(agentIndex == numAgents - 1):
                min_val = 0
                for action in gameState.getLegalActions(agentIndex):
                    min_val += self.expectimax(gameState.generateSuccessor(agentIndex, action), 0, numAgents, True, depth + 1, False)
                return min_val/len(gameState.getLegalActions(agentIndex))
            else:
                min_val = 0
                for action in gameState.getLegalActions(agentIndex):
                    min_val += self.expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, numAgents, False, depth, False)
                return min_val/len(gameState.getLegalActions(agentIndex))

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, gameState.getNumAgents(), True, 0, True)
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
