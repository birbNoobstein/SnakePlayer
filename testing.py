# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:46:51 2022

@author: Ana
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

def compare_score(data):
    plt.rcParams["figure.figsize"] = (4.5, 7)
    sb.boxplot(x='Algorithm',
               y='Score', 
               data=data,
               order=['basic', 'astar', 'hamiltonian'],
               width=0.4, 
               palette=[(174/235, 231/235, 180/235), 
                        (231/235, 174/235, 223/235)],
               saturation=1)
    plt.grid(axis='y', linestyle='--')
    plt.xticks([0, 1, 2], labels=['Basic', 'A*', 'Hamiltonian'])
    plt.savefig('comparison.png')
    plt.show()
    
def win_lose(data):
    algs = ['basic', 'astar', 'hamiltonian']
    reasons = data['Reason'].unique()
    
    reason_count = {}
    for reason in reasons:
        reason_count[reason] = [len(data.loc[(data['Reason'] == reason) & (data['Algorithm'] == a)]) for a in algs]
        
    reason_count = pd.DataFrame(reason_count, index=['Basic', 'A*', 'Hamiltonian'])
    sb.set(style='white')
    reason_count.plot(kind='bar', stacked=True, color=[(150/235, 231/235, 180/235), 
             (231/235, 160/235, 223/235), (111/235, 111/235, 222/235)], edgecolor='black')
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)
    plt.xticks(rotation=0)



if __name__ == '__main__':
    data = pd.read_csv('./test_results/results.csv', names=['Algorithm', 'Score', 'Reason'])
    
    compare_score(data)
    win_lose(data)
    
    basic = data.loc[data['Algorithm'] == 'basic']
    astar = data.loc[data['Algorithm'] == 'astar']
    hamilton = data.loc[data['Algorithm'] == 'hamiltonian']
    
    print('Means\n', 
          '   Basic: ', 
          basic['Score'].mean(), 
          '\n    A*: ', 
          astar['Score'].mean(), 
          '\n    Hamilton: ', 
          hamilton['Score'].mean())
    print('\nSD\n', 
          '   Basic: ', 
          basic['Score'].std(), 
          '\n    A*: ', 
          astar['Score'].std(), 
          '\n    Hamilton*: ', 
          hamilton['Score'].std())
    print('\nMean grid coverages in %\n', 
          '   Basic: ', 
          100*basic['Score'].mean()/399, 
          '\n    A*: ', 
          100*astar['Score'].mean()/399, 
          '\n    Hamilton: ', 
          100*hamilton['Score'].mean()/399)
    
    data['Score'] = 100 * data['Score']/(20*20-1)

    hamilton_wins = len(hamilton.loc[hamilton['Score'] == 399].index)
    print(f'Hamilton cycle won {hamilton_wins} out of 100 games')