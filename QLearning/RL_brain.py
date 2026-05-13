"""
Deep Q Network off-policy
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import maze_env
import random
import time
from tqdm import tqdm
np.random.seed(42)
torch.manual_seed(2)


class Network(nn.Module):
    """
    Network Structure
    """
    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=10
                 ):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_features,32),
            nn.Linear(in_features=32, out_features=n_neuron, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_neuron, out_features=n_actions, bias=True),      
        ) 

    def forward(self, s):
        """

        :param s: s
        :return: q
        """
        q = self.net(s)
        return q


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 env:maze_env.Maze,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # 这里用pd.DataFrame创建的表格作为memory
        # 表格的行数是memory的大小，也就是transition的个数
        # 表格的列数是transition的长度，一个transition包含[s, a, r, s_]，其中a和r分别是一个数字，s和s_的长度分别是n_features
        self.memory =self.memory_size*[[]] #pd.DataFrame(np.zeros((self.memory_size, self.n_features*2+2)))

        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions).cuda()
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions).cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)     
        # 记录每一步的误差
        self.cost_his = []
        self.unlegal_action=self.get_unlegal_action(env)
    def get_unlegal_action(self,env):
        result={}
        for i in range(0,maze_env.MAZE_W):
            for j in range(0,maze_env.MAZE_W):
                status=i+10*j
                result[status]=set()
                for action in [0,1,2,3]: 
                    offset1,offset2=env.get_offset(action)
                    i2=i+offset1
                    j2=j+offset2
                    if i2<0 or i2>=maze_env.MAZE_W:
                        result[status].add(action)
                        continue
                    if j2<0 or j2>=maze_env.MAZE_H:
                        result[status].add(action)
                        continue
 
        return result          
    def store_transition(self, s, a, r, s_,done):
        if not hasattr(self, 'memory_counter'):
            # hasattr用于判断对象是否包含对应的属性。
            self.memory_counter = 0

        transition = [s,a,r,s_,done]#np.hstack((s, [a,r], s_))

        # replace the old memory with new memory
        #print ("conter",self.memory_counter)
        index = self.memory_counter % self.memory_size
        #print (index)
        self.memory[index] = transition

        self.memory_counter += 1

    def choose_action(self, observation,greedy=False):
        #print ("b",observation,self.unlegal_action[observation])
        if np.random.uniform() < self.epsilon or greedy:
            status = torch.tensor([observation],dtype=torch.int64).cuda()
            actions_value = self.eval_net(status).cpu()
            actions_value=actions_value.detach().numpy()[0]
            actions_value=[[i,s] for i,s in enumerate(actions_value) if i  not in self.unlegal_action[observation]]
            #print (actions_value)
            #print (status,actions_value)
            action =sorted(actions_value,key=lambda s:s[1],reverse=True)[0][0]
            #print ("aaa",action,self.unlegal_action[observation])
        else:
            action = random.sample([i for i in range(0,self.n_actions) if i not in self.unlegal_action[observation]],1)[0]#np.random.randint(0, self.n_actions)
        #print ("aaa",action,self.unlegal_action[observation])
 
        return action

    def _replace_target_params(self):
        # 复制网络参数
        self.target_net.load_state_dict(self.eval_net.state_dict())
    def train(self,env:maze_env):
        step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
        all_step=50
        for episode in tqdm(range(all_step)):
            # initial observation
            observation = (0,0)
            while True:
                status=observation[0]+10*observation[1]
                action = self.choose_action(status)           
                offset1,offset2=env.get_offset(action)
                i1,j1=observation[0]+offset1,observation[1]+offset2
                observation_=(i1,j1)
                status_=i1+j1*10
                reward=0
                done=False
                #获胜
                if list(observation_) == env.end_center:
                    print ("win")
                    done=True
                    reward=env.end_reward
                #失败
                elif list(observation_) in env.hell_center_list:
                    print ("fail")
                    done=True
                    reward=env.hell_reward
                step += 1 
                # !! restore transition
                self.store_transition(status, action, reward, status_,done)
                # 超过200条transition之后每隔5步学习一次
                if (step > 200) and (step % 5 == 0):
                    self.learn()

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    break
                step += 1
 
        # end of game
        print("game over")

    def walk_maze(self,env):
        step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
        observation = (0,0)
        env.reset()
        while True:
            env.render()
            status=observation[0]+10*observation[1]
            action = self.choose_action(status,greedy=True)
            #print (action)
            #print (observation,self.Q[observation])
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




    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            #print ("xxxxxxxxxxxxx")
            self._replace_target_params()
            #print('\ntarget params replaced\n')

        # sample batch memory from all memory
        sub_memory=[s for s in self.memory if len(s)>0]
        if  len(sub_memory)<self.batch_size:
            batch_memory=sub_memory
        else:
            batch_memory=random.sample(sub_memory,self.batch_size)

        s,a,r,s_,done=zip(*batch_memory)
        # run the nextwork
        #transition = [s,a,r,s_]
        s = torch.tensor(s).cuda()
        s_ =torch.tensor(s_).cuda()
        q_eval = self.eval_net(s)
        q_next = self.target_net(s_).cpu()
         
        next_score=q_next.max(dim=1).values.detach().numpy()
        next_score=np.array([ score if not done[i] else 0 for i,score in enumerate(next_score)])
        q_target =q_eval.clone().cpu().detach().numpy()

        # 更新值
        #batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = a
        reward = np.array(r)
 
        q_target[:, eval_act_index] = reward + self.gamma * next_score
 
        # train eval network
        q_target=torch.tensor(q_target).cuda()
        self.optimizer.zero_grad()
        loss = self.loss_function(q_target, q_eval.cuda())
        loss.backward()
        self.optimizer.step()

        #self.cost_his.append(loss.detach().numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


































































































