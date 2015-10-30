# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
import resource
import sys
import time

"Global Data"
nodes_count=0

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

        "Counting Number Of Nodes Expanded"
        global nodes_count
        nodes_count=nodes_count+1

        #Parameter 1
        newPos = successorGameState.getPacmanPosition()

        #Parameter 2
        newFood = successorGameState.getFood()
        new_FSet = newFood.asList()
        closestFood = new_FSet and min([util.manhattanDistance(newPos, foodPos) for foodPos in new_FSet]) or 0
        food_Score = closestFood and 1.0 / float(closestFood)

        #Parameter 3
        newGhostStates = successorGameState.getGhostStates()
        ghost_Dist = min([util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])

        #Parameter 4
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghost_Scared_Score = sum(newScaredTimes)

        "*** YOUR CODE HERE ***"

        ################Code Changes For Question 1 Starts################
        "Logic::For Reflex Agent Focus is on food_score and ghost_distance"
        "And Not on ghost_Scared_Score"

        #Modification 1
        ret_val=(food_Score * ghost_Dist)

        #Modification 2
        ret_val=ret_val+ghost_Scared_Score

        #Modification 3
        ret_val=(ret_val+successorGameState.getScore())

        return ret_val
        ################Code Changes For Question 1 Ends################

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

        ###############Question 2 Answer Starts###############
        #Find Total Number Of Agents
        total_agents = gameState.getNumAgents()

        #Packman Agent Number/Index
        packman_agent = 0

        #Ghost Agent Number/Index
        ghost_agent = 1

        #If following function returns true means agent is packman
        def check_pacman_agent(agent_number):
            if((agent_number%total_agents==packman_agent)):
                return True
            else:
                return False

        #Max Function::For PacMan Agent
        def find_max_val(depth, gameState, agentNumber=packman_agent):

            #Get All The Legal Moves For a Packman Agent
            legal = gameState.getLegalActions(packman_agent)

            #If you have reached the last node or leaf node in a tree then return
            if(leaf_node(depth,legal,gameState)):
                return self.evaluationFunction(gameState), None

            #Aim:Returning The Maximum Value Of The Minimum Value Returned By The Ghost Agent
            min_val_set=[(find_min_val(depth, gameState.generateSuccessor(packman_agent, action),ghost_agent)[0], action) for action in legal]
            ret_val = max(min_val_set)

            #Return Max Of The Minimum Values
            return ret_val

        #Function which returns true when you have reached leaf node
        def leaf_node(depth,legal,game_State):
            if(depth == 0 or not(legal) or game_State.isWin()):
                return True
            else:
                return False

        #Function which returns true when you have reached leaf node
        def ghost_leaf_node(legal,game_State):
            if(not(legal) or (gameState.isLose())):
                return True
            else:
                return False

        #Min Function::For Ghosts Agent
        def find_min_val(depth, gameState,agent_num):

            #Get All The Legal Moves For a Ghost Agent
            legal_ghost_actions = gameState.getLegalActions(agent_num)

            if(ghost_leaf_node(legal_ghost_actions,gameState)):
                return self.evaluationFunction(gameState), None
            else:

                #Action Scores List
                game_scores = []

                #Iterate through ghost legal actions
                for action in legal_ghost_actions:
                    #Get all the legal actions for ghost agent
                    new_game_state = gameState.generateSuccessor(agent_num, action)

                    #Call to Anonymous Function
                    if(check_pacman_agent(agent_num + 1)):
                        #Finding The Maximum Value For Pacman Agent
                        game_score, _ = find_max_val(depth - 1,new_game_state, packman_agent)
                    else:
                        #Finding The Minimum Value For Ghost Agent
                        game_score, _ = find_min_val(depth, new_game_state, agent_num + 1)
                    game_scores.append((game_score, action))

                #Returns Minimum Score And Action Values
                return min(game_scores)

        #Global nodes_count variable for counting number of nodes expanded
        global nodes_count
        nodes_count=nodes_count+1

        return find_max_val(self.depth, gameState, 0)[1]
        ##############Question 2 Answer Ends##############

class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        ##########################Question 3 Answer Starts##########################
        #Finding Total Number Of Agents::Pacman+Ghost Agents
        total_agents = gameState.getNumAgents()

        #Indexes For Agents
        packman_Agent=0
        ghost_Agent=1

        #Initial Values Of Alpha, Beta and stop action
        stop_action = Directions.STOP
        init_alpha = -float('inf')
        init_beta = float('inf')

        "Following method finds maximum value based on current game state,depth,alpha and beta"
        def find_max_val(gamestate,depth,alpha,beta):

            #Initial Value For Pacman Agent
            ialpha = float('-inf')

            #Get All Legal Actions Of a Pacman Agent
            legal_actions_pacman = gamestate.getLegalActions(0)

            if((depth > self.depth) or check_legal_pacman_action(gamestate,legal_actions_pacman)):
                return self.evaluationFunction(gamestate), Directions.STOP

            #Iterate Through All The Legal Actions Of Pacman Agent
            for action in legal_actions_pacman:

                #Find Next State Of Pacman
                next_State = gamestate.generateSuccessor(0, action)

                #Find Cost Correspondong To The State
                game_score = find_min_val(next_State,ghost_Agent,depth, alpha, beta)[0]

                #If found score is greater then store it in ialpha
                if(game_score > ialpha):
                    ialpha = game_score
                    stop_action = action
                if(ialpha > beta):
                    return ialpha,stop_action

                #Return maximum
                alpha = max(alpha, ialpha)

            #Return Maximum Score and Action Pair
            return ialpha, stop_action

        "Following method checks whether packman action is legal or not"
        def check_legal_pacman_action(gamestate,legal_actions_pacman):
            if(gamestate.isWin() or not(legal_actions_pacman)):
                return True
            else:
                return False

        "Following method checks whether ghost action is legal or not"
        def check_legal_ghost_action(legal_actions_ghost,gamestate):
            if(not(legal_actions_ghost) or gamestate.isLose()):
                return True
            else:
                return False

        "Following method finds minimum value based on current game state,depth,alpha and beta"
        def find_min_val(gamestate,agent_index,depth,alpha,beta):

            #Initial Value For Pacman Agent
            ibeta = float('inf')

            #Get All Legal Actions Of a Ghost Agent
            legal_actions_ghost = gamestate.getLegalActions(agent_index)

            if(check_legal_ghost_action(legal_actions_ghost,gamestate)):
                return self.evaluationFunction(gamestate), Directions.STOP

            #Iterate Through All The Legal Actions Of Ghost Agent
            for action in legal_actions_ghost:

                #Find Next State Of Ghost
                next_State = gamestate.generateSuccessor(agent_index,action)

                if(check_packman(agent_index)):
                    game_score = find_max_val(next_State,(depth+1),alpha,beta)[0]
                else:
                    game_score = find_min_val(next_State,(agent_index+1),depth,alpha,beta)[0]

                #If you get the less value then store it in ibeta
                if(game_score<ibeta):
                    ibeta = game_score
                    stop_action = action

                #If alpha value is greater than beta then return
                if(ibeta<alpha):
                    return ibeta, stop_action

                #Return minimum of these 2
                beta = min(beta,ibeta)

            #Return Minimum Score and Action Pair
            return ibeta, stop_action

        "Following method checks whether agent_index belongs to packman or not"
        def check_packman(agent_index):
            if(agent_index == (total_agents - 1)):
                return True
            else:
                return False

        "To keep track of number of nodes expanded"
        global nodes_count
        nodes_count=nodes_count+1

        return find_max_val(gameState, 1, init_alpha, init_beta)[1]
        ##########################Question 3 Answer Ends##########################

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
        util.raiseNotDefined()

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

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
