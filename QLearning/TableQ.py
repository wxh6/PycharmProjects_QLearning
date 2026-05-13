import numpy as np
import pickle
from tqdm import tqdm
import time
from maze_env import Maze
import maze_env
import random
class TableQ():
    """
    Q Learning Algorithm
    """
    def __init__(self,
                 n_actions,
                 MAZE_H,
                 MAZE_W,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9
                 ):
        self.n_actions = n_actions
        self.Q = {}
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy

        self.epsilon=0.05
        # total learning step
        self.learn_step_counter = 0

 
        # 记录每一步的误差
        self.cost_his = []

    def get_Q(self,status):
        if status not in self.Q:
            self.Q[status]=4*[0]#np.random.randn(4)
        return self.Q[status]
    def __init_Q(self,env):
        self.Q={}
        for i in range(0,maze_env.MAZE_W):
            for j in range(0,maze_env.MAZE_W):
                status=(i,j)
                self.Q[status]={}
                for action in [0,1,2,3]: 
                    offset1,offset2=env.get_offset(action)
                    i2=i+offset1
                    j2=j+offset2
                    if i2<0 or i2>=maze_env.MAZE_W:
                        continue
                    if j2<0 or j2>=maze_env.MAZE_H:
                        continue
                    self.Q[status][action]=0

    def choose_action(self, observation,greedy=False):
        if np.random.uniform() < self.epsilon or greedy:
            
            action =sorted(self.Q[observation].items(),key=lambda s:s[1],reverse=True)[0][0] #[np.argmax(self.get_Q(status))][0]
        else:
            action =random.sample(list(self.Q[observation].keys()),1)[0]
        return action
    def learn(self,learn_observation,learn_action,learn_reward,reward):
        #学习目标：当前+目标
        self.Q[learn_observation][learn_action]= (1-self.lr)*self.Q[learn_observation][learn_action]+self.lr*(learn_reward+self.gamma*reward)
 
    def train(self,epochs=200,style="sarsa"):
        env = Maze()
        self.__init_Q(env)
        for episode in tqdm(range(epochs)):
            # initial observation
            observation  = (0,0)
            learn_observation=None
            learn_action=None
            learn_reward=None
            step=0
            while True:
                # refresh env
                env.render()
                # RL choose action based on observation
                action = self.choose_action(observation)
                offset1,offset2=env.get_offset(action)
                # RL take action and get next observation and reward
                #下一个状态
                observation_=(observation[0]+offset1,observation[1]+offset2)
                reward=0
                done=False
                #获胜
                if list(observation_) == env.end_center:
                    reward=env.end_reward
                    done=True
                #失败
                elif list(observation_) in env.hell_center_list: 
                    reward=env.hell_reward
                    done=True
                #observation_, reward, done = env.step(action)
                #print (reward,done)
  
                if learn_observation is not None and learn_action is not None and learn_reward is not None :
                    #status=str(observation)
                    if style=="sarsa":
                        #下一步动作价值Q
                        reward2=self.Q[observation][action]#self.get_Q(status)[action]
                    else:
                        #
                        reward2=max(self.Q[observation].values())
 
 
                    self.learn(learn_observation,learn_action,learn_reward,reward2)      
                    #print (self.Q)       
                learn_observation=observation
                learn_action=action
                learn_reward=reward
                if done:
                    self.learn(learn_observation,learn_action,learn_reward,0)
                    break
                observation = observation_
                step += 1
        print("game over")
        print (self.Q)
        env.destroy()
    def save(self,path):
        with open(path,"wb") as f:
            pickle.dump(self,f)
    def load(path):
        with open(path,"rb") as f:
            RL=pickle.load(f)
        return RL
    def walk_maze(self,env):
        step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
        observation = (0,0)
        env.reset()
        while True:
            env.render()
            action = self.choose_action(observation,greedy=True)
            print (observation,self.Q[observation])
            env.step(action)
            time.sleep(1)
            offset1,offset2=env.get_offset(action)
                # RL take action and get next observation and reward
            observation=(observation[0]+offset1,observation[1]+offset2)
            done=False
            #获胜
            if list(observation) == env.end_center:
                done=True
            #失败
            elif list(observation) in env.hell_center_list: 
                done=True
            if done:
                break
            step += 1
        print ("END")
