from maze_env import Maze
#from RL_brain import DeepQNetwork
 
from TableQ import TableQ
import time
from tqdm import tqdm
import pickle
import maze_env




    #env.destroy()
 



if __name__ == "__main__":

    #训练学习
    # env = Maze()
    # RL = TableQ(4,maze_env.MAZE_W,maze_env.MAZE_H,learning_rate=0.2,reward_decay=0.9,e_greedy=0.8 )
    # RL.train(style="Qlearing",epochs=30000)
    # RL.save("Qlearing")
    #学习的效果
    env = Maze()
    RL=TableQ.load("Qlearing")
    RL.walk_maze(env)
    env.mainloop()



    # RL = TableQ(4,maze_env.MAZE_W,maze_env.MAZE_H,learning_rate=0.2,reward_decay=0.9,e_greedy=0.5 )
    # RL.train(style="sarsa",epochs=50000)
    # RL.save("sarsa")

    # env = Maze()
    # RL=TableQ.load("sarsa")
    # RL.walk_maze(env)
    # env.mainloop() 




    # # maze game
    # env = Maze()
    # RL = DeepQNetwork(4,100,env,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=200,
    #                   memory_size=2000)
    # RL.train(env)
    # with open("DQN","wb") as f:
    #     pickle.dump(RL,f)
    # with open("DQN","rb") as f:
    #     RL=pickle.load(f)
    # RL.walk_maze(env)
    # env.mainloop()

    # env = Maze()
    # RL = TableQ(4,maze_env.MAZE_W,maze_env.MAZE_H,learning_rate=0.1,reward_decay=0.9,e_greedy=0.9 )
    # RL.train(env,style="DQN")
    # RL.save("DQN")

 
    # env = Maze()
    # RL=TableQ.load("DQN")
    # RL.walk_maze(env)
    # env.mainloop()