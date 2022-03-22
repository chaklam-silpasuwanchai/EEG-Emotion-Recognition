# https://github.com/ruteee/TransferEntropy

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import random
import time
import math
import itertools

from IPython.display import clear_output


# In[2]:


def joint_probability_new_version(k,l,h, a, b, lbl_a, lbl_b):
    '''
    k B time horizon
    l A time horizon
    h instant in the future of serie B

    a, b array type

    lbl_a - label of a series
    lbl_b - label of b series
    '''
        
    
    numStates=2**(k+l+1)
    combinations = list(map(list, itertools.product([0, 1], repeat=k+l+1)))
    prob_cnjt = np.zeros(numStates)

    matrix_nova = np.matrix([b[1:],b[:-1],a[:-1]]).T
    df = pd.DataFrame(matrix_nova, columns = ['b_ftr', lbl_b, lbl_a])
    gpd = df.groupby(['b_ftr', lbl_b, lbl_a], as_index=False).size().reset_index(name='Count')
    total = sum(gpd['Count'])
    
    for i in np.arange(0,gpd.shape[0]):
        comb = [e for e in gpd.iloc[i][0:3].values.tolist()]
        idx = combinations.index(comb)
        prob_cnjt[idx] = gpd.iloc[i]['Count']/total


# In[3]:


def joint_probability(k,l,h, a, b):
    '''
        k B time horizon
        l A time horizon
        h instant in the future of serie B
        
        a, b array type'''

    #Alarm Series A (cause), B (effect), same len
    #tested
    sizeSeries = a.size
    transEntropy = 0
    numStates = 2**(k + l  + 1)
    combinations = list(map(list, itertools.product([0, 1], repeat=k+l+1)))
    counting = np.zeros(numStates)
    prob_cnjt = np.zeros(numStates)
    a_prob_ind = []
    b_prob_ind = []
    #joitn probability p(i_sub_t+1), i_sub_t**k, j_sub_t**l)
    inicio = np.max([k,l]) - 1
    for i in np.arange(inicio, sizeSeries - h):
        for hk in np.arange(0,k):
                b_prob_ind.append(b[i - hk])
        for hl in np.arange(0,l):
                a_prob_ind.append(a[i - hl])

        #print(a.size, b.size, a.size -1)     
        ab = [b[i + h]] + b_prob_ind + a_prob_ind 
        index_comb = combinations.index(ab)
        counting[index_comb] = counting[index_comb] + 1

        a_prob_ind = []
        b_prob_ind = []

    total = sum(counting)
  
    prob_cnjt = counting/total
     
    return prob_cnjt


# In[4]:


#Joint probability evaluation p(i_t+h, i_t**k)
#tested
def joint_prob_ih_ik(k,l, joint_prob_ih_ik_jl):
    states_ith_ik = list(map(list, itertools.product([0, 1], repeat=k + 1)))
    combinations = list(map(list, itertools.product([0, 1], repeat=k+l+1))) 
    p_jnt_ith_ik = np.zeros(2**(k+1))
    
    for i, state in enumerate(states_ith_ik):
        for j, comb in enumerate(combinations):
            if comb[0:k+1] == state:
                p_jnt_ith_ik[i] = p_jnt_ith_ik[i] + joint_prob_ih_ik_jl[j]
    return p_jnt_ith_ik


# In[5]:


def conditional_prob(k,l,joint_prob):
    states = list(map(list, itertools.product([0, 1], repeat=k+l)))
    combinations = list(map(list, itertools.product([0, 1], repeat=k+l+1)))

    size = int(joint_prob.size/2)
    conditional = np.zeros(2**(k+l+1))

    for i,state in enumerate(states):
        index_zero = combinations.index([0] + state)
        prob_zero = joint_prob[index_zero]

        index_one = combinations.index([1] + state)
        prob_one = joint_prob[index_one]

        if(prob_zero + prob_one != 0):
            conditional[i] = prob_zero/(prob_zero+ prob_one)
            conditional[i + 2**(k+l)] = prob_one/(prob_zero+ prob_one)
    return conditional


# In[6]:


#Division of the conditionals in log2 
#tested
def conditional_div(k,l,conditional_num, conditional_den):
    combinations = list(map(list, itertools.product([0, 1], repeat=k+l+1)))
    conditional_division = np.zeros(conditional_num.size)
    states_den = list(map(list, itertools.product([0, 1], repeat=1+k)))
    for j, comb in enumerate(combinations):
        if(conditional_den[states_den.index(comb[0:k+1])] != 0):
            conditional_division[j] = conditional_num[j]/conditional_den[states_den.index(comb[0:k+1])]            
    return conditional_division


# In[7]:


#Transfer entropy final evaluation
def te(k,l,h,a,b):
    '''
        transentropy a->b
        te(k,l,h,a,b)
        k - dimension of b, number of samples of the past of b
        l - dimension of a, number of samples of the passt of a
        h -> instant in the future of b
    '''
    joint_p_ih_ik_jl = joint_probability(k,l,h,a,b)
    
    joint_p_ih_ik = joint_prob_ih_ik(k,l, joint_p_ih_ik_jl)
    conditional_num = conditional_prob(k,l,joint_p_ih_ik_jl)
    conditional_den = conditional_prob(k,0, joint_p_ih_ik)    
    div = conditional_div(k,l,conditional_num, conditional_den)
    
    #log2 from the division of the conditionals -> #p(i_sub_t+h|i_sub_t**k, j_sub_t**l) /p(i_sub_t+h|i_t**k)
    
    log2_div_cond = np.log2(div[div!=0])
    te = np.sum(joint_p_ih_ik_jl[div!=0]*log2_div_cond)
    return te


# In[8]:


def transferEntropy_case(df, k, l, h):
    
    '''Evaluate Transfer entropy for a dataframe of variables'''
    start = time.clock()
    transEntropy = np.zeros([df.columns.size,df.columns.size])
    sigValues =  np.zeros([df.columns.size,df.columns.size])
    for i in np.arange(0, df.columns.size):
        for j in np.arange(0, df.columns.size):
            print('trans ', df.columns[i], df.columns[j])
            if(j != i + df.columns.size/2 and j!=i and j != i - df.columns.size/2):
                transEntropy[i][j] = te(k,l,h,df[df.columns[i]],
                                         df[df.columns[j]])
            clear_output()
    end = time.clock()   
    
    print(end - start)
    return transEntropy  


# In[ ]:





# In[9]:


#joint probablity for functions test
#joint_p_ih_ik_jl = np.array([0.97322404,0.00546448,0.00491803,0,0,0.00546448, 0.00546448, 0.00546448])

#aproximate results for this test

#p(ith, ik)
#jnt_p_ih_ik = [0.97868852,0.00491803,0.00546448,0.0109286] 

#p(i_t+h|i**k, j**l)
#cond_p_ih_ik_jl =  [1,0.5,0.4736841094123,0,0,0.5, 0.52631589085076,1]

#p(i_th|i_k)
#cond_p_ih_ik = [0.994711793480152,0.31035179088550,0.0552469991962,0.68964820911449]
