import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.t = 0
        

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        self.t = self.t + 1
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        if testing == True:
            self.epsilon = 0
            self.alpha = 0
        else:
            
            #self.epsilon = self.epsilon -0.02 # bue. a+,f
            #self.epsilon =  math.pow(0.98,self.t) # > 130 iter
            self.alpha = math.pow(0.98,self.t)
            #self.epsilon=float(1)/math.exp(float( .15 * self.t)) # 20 iter a+,f
            #self.alpha =  math.cos(0.04*self.t) # 35 iter decay bad to slow in one  d,f
            #self.alpha = 1 - math.exp(-0.1*self.t)
            print "-----epsilon----"
            print self.epsilon
            if self.t > 140:
                #self.alpha = self.alpha - 0.15
                self.epsilon = self.epsilon - 0.15
                print "-----epsilon----"
                print self.epsilon
            else:
                self.epsilon =  math.pow(0.98,self.t)
            
            #self.epsilon =  1/math.pow(self.t,2) # 20 iter not use decay fast. f,f
            #self.epsilon =  math.exp(-0.05*self.t) # 70 trials, a+,f 
            #self.alpha =  math.exp(-0.05*self.t)
            #self.epsilon = 0.4/(0.5+self.t) # 20. a+,f decrease to fast
            #self.epsilon =  1/(1 + 0.1 * self.t) # a+,d
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 
        
        #listdata = ['left','right','right','right']
        #listdata = ['left','left','right']
        #listdata = ['left']
        #listdata = ['left','oncoming']
        #del inputs['right']
        #del inputs['left']
        #del inputs['oncoming']
        #inputs["waypoint"] = waypoint
        
        #inputs["deadline"] = deadline
        #listdata = ['left','oncoming','deadline','deadline'] # law of right intersection u.s
        #listdata = ['left','oncoming']
        #del inputs['right']
        #inputs["waypoint"] = waypoint
        #del inputs[random.choice(listdata)]
        #self.state = tuple(inputs.values())
        #state = tuple(inputs.values())
        
        listdata = ['left','oncoming']
        #listdata = [waypoint,deadline]
        listdatados = ['left','oncoming']
        feature_random = random.choice(listdata) #HELPS TO REDUCE SPACE STATE
        #self.state = (inputs['light'],inputs['oncoming'],inputs['left'] ,waypoint)
        self.state = (inputs['light'],inputs[feature_random] ,waypoint)
        #self.state = (inputs['light'],inputs['oncoming'],inputs['left'] ,feature_random)
        #self.state = (inputs['light'],inputs[random.choice(listdatados)],feature_random)
        self.createQ(self.state)
        
        return self.state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        
        
        # Calculate the maximum Q-value of all actions for a given state
       
        maxQ = max(self.Q[state].values()) #  max(self.Q[state], key=self.Q[state].get)
        
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        
        if self.learning == True:
            if state not in self.Q:
                self.Q[state] = {}
                for i in range(len(self.valid_actions)):
                    self.Q[state][self.valid_actions[i]] =random.uniform(0.05, 0.08)
                
        
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        #action = random.choice(self.valid_actions)
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        
        if self.learning == True:
            
            if random.random() < self.epsilon :
                
                action = random.choice(self.valid_actions)
                print "-----random-----"
                print action
            else:
                action = max(self.Q[state], key=self.Q[state].get)
                print "--action---"
                print action
                print self.Q[state]
        else:
            
            
            action = random.choice(self.valid_actions)
            
       
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
       
        next_state = self.build_state()
        self.createQ(next_state)
        
        if self.learning:
           
            
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.get_maxQ(next_state)  - self.Q[state][action])
         
            
     #Q(s,a) = Q(s,a) + alpha * [R(s,a) + gamma * argmax(R(s', a')) - Q(s, a)]
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = False,num_dummies=50,grid_size=(4,4))
    #env = Environment(verbose = False,num_dummies=60,grid_size=(4,4))
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning=True,alpha=0.60,epsilon=1)
    #alpha 0.02,0.01 epsilon 1 = d,b . 170 iters
    #alpha 0.02 epsilon 1 = d,b . 170 iters
    #alpha 0.08 epsilon 1 = d,f . 170 iters
    #alpha 0.15 epsilon 1 = c,f . 170 iters
    #alpha 0.25 epsilon 1 = c,f . 170 iters
    #alpha 0.30,0.01 epsilon 1 = a+,f . 170 iters
    #alpha 0.35 epsilon 1 = a+,f . 170 iters
    #alpha 0.45 epsilon 1 = d,f . 170 iters
    #alpha 0.40 epsilon 1 = d,f . 170 iters
    #alpha 0.35 epsilon 1 = d,f . 170 iters
    #alpha 0.35 epsilon 1 = f,d . 70 iters 
    #alpha 0.25 epsilon 1 = d,f . 70 iters 
    #alpha 0.02 epsilon 1 = d,b . 70 iters 
    #alpha 0.01 epsilon 1 = a+,a . 170 iters listdata = ['left','right'] sometimes d,f
    #alpha 0.01 epsilon 1 = a+,f . 170 iters listdata = ['right'] 
    #alpha function epsilon function = a+,d . 70 iters listdata = ['left','oncoming'], num_dummies=50,grid_size=(5,5)
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline=True)
    #env.set_primary_agent(agent)
    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env,update_delay=0.001,log_metrics=True,optimized=True,display=False)
    #sim = Simulator(env,log_metrics=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=40,tolerance=0.05)


if __name__ == '__main__':
    run()
