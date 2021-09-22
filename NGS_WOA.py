# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 02:03:07 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1109/ACCESS.2020.2989445
"""

import numpy as np
import matplotlib.pyplot as plt

class NGS_WOA():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
    def opt(self):
        # 初始化
        tao = (np.sqrt(5)-1)/2
        m1 = -2*np.pi + (1-tao)*2*np.pi
        m2 = -2*np.pi + tao*2*np.pi
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            a = self.a_max - (self.a_max-self.a_min)*(g/self.G)

            if g<0.5*self.G:
                C1 = 0.5*(1 + np.cos(np.pi*g/self.G))**0.5 # (8)
            else:
                C1 = 0.5*(1 - np.cos(np.pi+(np.pi*g/self.G)))**0.5 # (8)
                
            for i in range(self.P):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                A = 2*a*r1 - a
                C = 2*r2
                l = np.random.uniform(low=self.l_min, high=self.l_max)
                
                if p<0.5:
                    if np.abs(A)<1:
                        self.X[i, :] = self.gbest_X - C1*A*np.abs(C*self.gbest_X-self.X[i, :]) # (9)
                    else:
                        X_rand = self.X[np.random.randint(low=0, high=self.P, size=self.D), :]
                        X_rand = np.diag(X_rand).copy()
                        self.X[i, :] = X_rand - C1*A*np.abs(C*X_rand-self.X[i, :]) # (10)
                else:
                    D = np.abs(self.gbest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(self.b*l)*C1*np.cos(2*np.pi*l) + self.gbest_X # (11)

            r3 = np.random.uniform()
            r4 = np.random.uniform()
            self.X = self.X*np.abs(np.sin(r3)) + r4*np.sin(r3)*np.abs(m1*self.gbest_X-m2*self.X) # (20)
  
            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.loss_curve[-1], 3))+']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
            