# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:46:51 2022

@author: Ana
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

data = pd.read_csv('results.csv', names=['Algorithm', 'Score'])

plt.rcParams["figure.figsize"] = (4.5, 7)
sb.boxplot(x='Algorithm',
           y='Score', 
           data=data, 
           width=0.4, 
           palette=[(174/235, 231/235, 180/235), 
                    (231/235, 174/235, 223/235)],
           saturation=1)
plt.grid(axis='y', linestyle='--')
plt.xticks([0, 1], labels=['Basic', 'A*'])
plt.yticks(range(25, 300, 25), labels=range(25, 300, 25))
plt.title('Basic and A* algorithm comparison')
plt.savefig('comparison.png')