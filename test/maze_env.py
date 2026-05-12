import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.h=MAZE_H
        self.w=MAZE_W
        self.end_reward=100
        self.hell_reward=-1
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.init_P()
 
    def init_P(self):
        self.P={}
        for i in range(0,self.h):
            self.P[i]={}
            for j in range(0,self.w):
                self.P[i][j]={}
                if [i,j] == self.end_center or [i,j] in  self.hell_center_list:
                     self.P[i][j]=None 
                     continue
                for a in range(0,self.n_actions):
                    offset=self.get_offset(a)
                    next=[i+offset[0],j+offset[1]]
                    if next[0]<0 or next[0]>=self.w or next[1]<0 or next[1]>=self.h:
                        continue
                    if next == self.end_center:
                        reward=self.end_reward
                        done=True
                    elif next in self.hell_center_list:
                        reward=self.hell_reward
                        done=True
                    else:
                        reward=0
                        done=False
                    self.P[i][j][a]=next,reward,done


    def get_vaild_action(self,h,w):
        unleagl=set()       
        if h==self.h-1:
            unleagl.add(1)
        if h==0:
            unleagl.add(0)
        if w==self.w-1:
            unleagl.add(3)
        if w==0:
            unleagl.add(2)
        # if h==0 and w==0:
        #     print (unleagl)
        #     print ([ i for i in range(0,4) if i not in unleagl])
        #     asd
        return [ i for i in range(0,4) if i not in unleagl]
        

    def create_point(self,origin,w,h,color):
        center = origin + np.array([UNIT * w, UNIT*h])
        point = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=color)

        return point
    def create_point_list(self,origin,center_list,color):
        results=[]
        for w,h in center_list:
            point=self.create_point(origin,w,h,color)
            results.append(point)
        return results
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        self.origin = np.array([20, 20])
        #不能踩的点
        # hell
        self.hell_center_list=[[5,3],[3,5],[4,3],[3,4],[0,2],[2,0],[0,4],[2,1],[4,1],[3,3],[8,5],[9,7],[6,5],[7,9],[6,3]]
        #self.hell_center_list=[]#[[5,3],[3,5]]
 
        self.hell_list=self.create_point_list(self.origin,self.hell_center_list,"black")
        #终点
        # create oval
        self.end_center=[5,5]
        self.end=self.create_point(self.origin,self.end_center[0],self.end_center[1],"yellow")
 
        self.start_center=[0,0]
        # create red rect
        self.start=self.create_point(self.origin,self.start_center[0],self.start_center[1],"red")

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.start)
        self.start = self.create_point(self.origin,self.start_center[0],self.start_center[1],"red")
  
        return  np.array(self.canvas.coords(self.start)[:2])/(MAZE_H*UNIT)
    def get_offset(self,action):
        if action == 0:   # up
            result=[0,-1]
        elif action == 1:   # down
            result= [0,1]
        elif action == 2:   # right
            result= [1,0]
        elif action == 3:   # left
            result= [-1,0]
        return np.array(result)
    def step(self, action):
        #['u', 'd', 'l', 'r']
        s = self.canvas.coords(self.start)
        base_action = np.array([0, 0])
        offset=self.get_offset(action)
        if action == 0:   # up
            if s[1] > UNIT:
                base_action=base_action+offset* UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action=base_action+offset* UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action=base_action+offset* UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action=base_action+offset* UNIT
        self.canvas.move(self.start, base_action[0], base_action[1])  # move agent
        next_coords = self.canvas.coords(self.start)  # next state
        if next_coords == self.canvas.coords(self.end):
            reward = self.end_reward
            done = True
        elif next_coords in [self.canvas.coords(hell) for hell in self.hell_list]:
            reward = self.hell_reward
            done = True
        else:
            reward = 0
            done = False
        #s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.end)[:2]))/(MAZE_H*UNIT)
        s_ = np.array(next_coords[:2])/(MAZE_H*UNIT)
        
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


