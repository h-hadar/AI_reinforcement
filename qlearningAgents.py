# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import math
import random

import util
from learningAgents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
    
      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - update
    
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions
          for a state
    """
    
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()
    
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        return self.q_values[(state, action)]
    
    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        max_next = -math.inf
        for a in self.getLegalActions(state):
            max_next = max(max_next, self.getQValue(state, a))
        return max_next if max_next != -math.inf else 0
    
    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_next = -math.inf
        best_actions = []
        for a in self.getLegalActions(state):
            cur_val = self.getQValue(state, a)
            if max_next < cur_val:
                max_next = cur_val
                best_actions = [a]
            elif max_next == cur_val:
                best_actions.append(a)
        if not best_actions:
            return None
        return random.choice(best_actions)
    
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
    
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        dont_follow_policy = util.flipCoin(self.epsilon)
        if not legalActions:
            return None
        if dont_follow_policy:
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
    
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        prev_val = self.getQValue(state, action)
        max_next = self.getValue(nextState)
        next_val = reward + self.discount * max_next
        self.q_values[(state, action)] = (1-self.alpha) * prev_val + self.alpha * next_val


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"
    
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
    
    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
  
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        
        # You might want to initialize weights here.
        self.weights = util.Counter()  # keys are features, values are weights
    
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        feature_values = self.featExtractor(state, action)  # counter for features
        total = 0
        for feature in self.weights:
            total += self.weights[feature] * feature_values[feature]
        return total
    
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        feature_vals = self.featExtractor(state, action)
        correction = reward + self.discount * self.getValue(nextState) \
                     - self.getQValue(state,action)
        for feature_i in feature_vals:
            self.weights[feature_i] += self.alpha * correction * feature_vals[feature_i]

    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
