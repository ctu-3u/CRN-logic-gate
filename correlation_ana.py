import numpy as np
import numpy.matlib
import numpy.linalg

import scipy.stats as st
import numba
import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import ness_static as sts

m_dict_connection = {0:[1,2,4],1:[0,3,5],2:[0,3,6],3:[1,2,7],\
4:[0,5,6],5:[1,4,7],6:[2,4,7],7:[3,5,6]} # connected states

m_update = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=int) # label the position of changed variable

# compute propensity
def cal_propensity(ini,mat_k):
    prpn = np.zeros(8)
    for i in range(0,8):
        prpn[i] = mat_k[ini,i]
    return prpn

# draw out which reaction to happen according to discrete distribution
def sample_discrete(probs):
    q = np.random.rand()
    i = 0
    prob_cumu = 0
    while prob_cumu<q:
        prob_cumu = prob_cumu+probs[i]
        i = i+1
    return i-1

# draw out change in one single step
def gillespie_draw(c_cal_propensity,index,mat_k,args=()):
    propensity = c_cal_propensity(index,mat_k)
    propen_sum = np.sum(propensity)
    propensity = propensity/propen_sum
    # choose reaction interval time
    dtime = np.random.exponential(1/propen_sum)
    # choose what reaction to happen
    index = sample_discrete(propensity)
    return index, dtime

# Gillespie simulation algorithm
def gillespie_ssa(c_cal_propensity,timepoints,initial_index,mat_k,args=()):
    size = len(timepoints)
    record = np.matlib.zeros((size,3)) # record index of state, x label and y label
    record[0,:] = [initial_index,sts.m_x_label[initial_index],sts.m_y_label[initial_index]]
    t = 0
    i = 0
    i_time = 1 # indexing the latter bound of a time interval
    index = initial_index
    while i<size:
        while t<=timepoints[i_time]:
            # draw one single change
            (index,dtime) = gillespie_draw(c_cal_propensity,index,mat_k,args)
            t = t+dtime
            #
            print(index,dtime,t)
        i = np.searchsorted(timepoints,t,side='left')
        for j in range(i_time,i):
            record[j,:] = [index,sts.m_x_label[index],sts.m_y_label[index]]
        i_time = i
    return record

###################################### MAIN ##################################################

# modificating the system
m_x1 = 1
m_x2 = 1
m_g = 15
m_h = 7

# computing jumping rates (propensities)
m_mat_k = sts.cal_mat_k(m_x1,m_x2,m_g,m_h)

# running Gillespie simulation
timepoints = np.arange(0,1000,0.05)
initial_index = 7

np.random.seed(13)

m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,m_mat_k,args=(m_g,m_h))

# plotting trajectory
plt.figure()
plt.plot(timepoints,m_traj[:,1]+2,'r.--',label='(m,a)')
plt.plot(timepoints,m_traj[:,2],'b.--',label='y')
plt.xlabel('timestep')
plt.ylabel('trajectory')
plt.legend()
plt.show()