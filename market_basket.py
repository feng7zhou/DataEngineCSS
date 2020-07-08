# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:39:27 2020

@author: FengZhou
"""


import pandas as pd
import numpy as np
from efficient_apriori import apriori

dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header=None)
print(dataset.shape)

transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0,20):
        if str(dataset.values[i,j])!='nan':
            temp.append(str(dataset.values[i,j]))
    transactions.append(temp)

'''
itemsets, rules = apriori(transactions, min_support=0.05, min_confidence=0.3)
print('频繁项集',itemsets)
print('关联规则',rules)
'''
#以上是利用apriori做的频繁项集和关联规则

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_itemsets= apriori(transactions, min_support=0.05, use_colnames=False)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.3)
frequent_itemsets = frequent_itemsets.sort_value(by='support', ascending=False)
print('频繁项集',frequent_itemsets)

transactions.options.display.max_columns=100
rules = rules.sort_values(by='lift',ascending=False)
print('关联规则',rules)
#以上利用mlxtend，但是最后报错'list' object has no attribute 'size'是什么原因？