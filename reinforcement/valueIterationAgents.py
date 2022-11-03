# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(1, self.iterations + 1):
            newUtility = util.Counter()
            for s in self.mdp.getStates():
                newUtility[s] = self.getQValue(s, self.getAction(s))
            self.values = newUtility

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0

        if action is None:
            return qValue

        probabilityAndNextStep = self.mdp.getTransitionStatesAndProbs(state, action)

        for move in probabilityAndNextStep:
            qValue += (move[1] * (self.mdp.getReward(state, action, move[0]) + (self.discount * self.values[move[0]])))
            if move[0] == 'TERMINAL_STATE':
                break

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        tempUtility = -1e9
        maxAction = None

        for action in self.mdp.getPossibleActions(state):
            utility = self.getQValue(state, action)
            if tempUtility < utility:
                maxAction = action
                tempUtility = utility

        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        s = self.mdp.getStates()
        for i in range(0, self.iterations):
            index = i % len(s)
            self.values[s[index]] = self.getQValue(s[index], self.getAction(s[index]))


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.pred = {}
        self.pq = util.PriorityQueue()
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def initData(self):
        for s in self.mdp.getStates():
            self.pred[s] = set()

        for s in self.mdp.getStates():
            tempUtility = -1e9
            for a in self.mdp.getPossibleActions(s):
                utility = self.getQValue(s, a)
                tempUtility = max(tempUtility, utility)
                for move in self.mdp.getTransitionStatesAndProbs(s, a):
                    if move[1] != 0:
                        self.pred[move[0]].add(s)
            if s != 'TERMINAL_STATE':
                self.pq.push(s, -1 * abs(tempUtility-self.values[s]))

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.initData()
        for i in range(self.iterations):
            if self.pq.isEmpty():
                break
            top = self.pq.pop()
            if top != 'TERMINAL_STATE':
                tempUtility = -1e9
                for a in self.mdp.getPossibleActions(top):
                    utility = self.getQValue(top, a)
                    tempUtility = max(tempUtility, utility)
                self.values[top] = tempUtility

            for p in self.pred[top]:
                tempUtility = -1e9
                if p == 'TERMINAL_STATE':
                    continue
                for a in self.mdp.getPossibleActions(p):
                    utility = self.getQValue(p, a)
                    tempUtility = max(tempUtility, utility)
                diff = abs(tempUtility-self.values[p])
                if diff > self.theta:
                    self.pq.update(p, -1 * diff)
