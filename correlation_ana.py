import numpy as np
import numpy.linalg

import scipy.stats as st
import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import ness_static as sts

m_dict_connection = {0:[1,2,4],1:[0,3,5],2:[0,3,6],3:[1,2,7],\
4:[0,5,6],5:[1,4,7],6:[2,4,7],7:[3,5,6]} # connected states
m_dict_xlabel = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}

# compute propensity
def cal_propensity(ini,*args):
    mat_k = sts.cal_mat_k(*args)
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
def gillespie_draw(c_cal_propensity,index,*args):
    propensity = c_cal_propensity(index,*args)
    propen_sum = np.sum(propensity)
    propensity = propensity/propen_sum
    # choose reaction interval time
    dtime = np.random.exponential(1/propen_sum)
    # choose what reaction to happen
    index = sample_discrete(propensity)
    return index, dtime

# Gillespie simulation algorithm
def gillespie_ssa(c_cal_propensity,timepoints,initial_index,*args):
    args1 = np.asarray(args).flatten()
    size = len(timepoints)
    record = np.zeros((size,4)) # record index of state, x label and y label
    record[0,:] = [initial_index,sts.m_x_label[initial_index],sts.m_y_label[initial_index],3]
    t = 0
    i = 0
    i_time = 1 # indexing the latter bound of a time interval
    index = initial_index
    while i<size:
        while t<=timepoints[i_time]:
            # draw one single change
            # args1[0] = np.random.randint(0,1)
            # args1[1] = np.random.randint(0,1)
            (index,dtime) = gillespie_draw(c_cal_propensity,index,*args1)
            t = t+dtime
        i = np.searchsorted(timepoints,t,side='left')
        for j in range(i_time,i):
            record[j,:] = [index,sts.m_x_label[index],sts.m_y_label[index],m_dict_xlabel[(args1[0],args1[1])]]
        i_time = i
    return record

###################################### MAIN ##################################################

## Console

# modificating the system
m_x1 = 1
m_x2 = 1
m_g = 15
m_h = 7

#  modificating sampling size
m_smpl_seed = 50
m_smpl_tau = 250
m_smpl_dt = 100

timepoints = np.arange(0,1000,0.05)

## Running

# running Gillespie simulation
initial_index = 7
time_middle_i = int(len(timepoints)/2)

m_rc_large = np.zeros((m_smpl_seed,2*m_smpl_tau+1,6))

for i in tqdm.tqdm(range(m_smpl_seed)):
    seed = np.random.seed(i+1)
    m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,m_x1,m_x2,m_g,m_h)
    m_rc = np.zeros((2*m_smpl_tau+1,6)) # record: xy_bar, x_bar, y_bar, xy_0, x_0, y_0
    filename = f".\\resl_corr\\trajctry_{i}.dat"
    with open(filename,'w') as f:
        np.savetxt(f,m_traj,header='index\tsensor\tresponsor')
    for j in range(-m_smpl_tau,m_smpl_tau+1): # 2*m_smpl_tau+1 samples in total
        t_xtytau = 0 # \bar(x(t)*y(t+\tau))
        t_xt = 0 # \bar(x(t))
        t_ytau = 0 # \bar(y(t+\tau))
        for k in range(-m_smpl_dt,m_smpl_dt+1): # 2*m_smpl_dt+1 samples in total
            if k==0:
                t_xt_0 = m_traj[time_middle_i,1]+1 # x(0)
                t_ytau_0 = m_traj[time_middle_i+20*j,2]+1 # y(0+\tau)
            t_xt = t_xt+m_traj[time_middle_i+20*k,1]+1
            t_ytau = t_ytau+m_traj[time_middle_i+20*k+20*j,2]+1
            t_xtytau = t_xtytau+(m_traj[time_middle_i+20*k,1]+1)*(m_traj[time_middle_i+20*k+20*j,2]+1)
        t_xtytau = t_xtytau/(2*m_smpl_dt+1)
        t_xt = t_xt/(2*m_smpl_dt+1)
        t_ytau = t_ytau/(2*m_smpl_dt+1)
        m_rc[j,:] = [t_xtytau,t_xt,t_ytau,t_xt_0*t_ytau_0,t_xt_0,t_ytau_0]
    m_rc_large[i,:,:] = m_rc
    filename = f".\\resl_corr\\correlation_{i}.dat"
    with open(filename,'w') as f:
        np.savetxt(f,m_rc,header='x*y_bar\tx_bar\ty_bar\tx*y_0\tx_0\ty_0')
m_rc_mean = np.mean(m_rc_large,axis=0)
filename = f".\\resl_corr\\correlation.dat"
with open(filename,'w') as f:
    np.savetxt(f,m_rc_mean,header='x*y_bar\tx_bar\ty_bar\tx*y_0\tx_0\ty_0')

# plotting correlation curve
plt.figure()
x_axis = range(-m_smpl_tau,m_smpl_tau+1)
m_corr = (m_rc_mean[:,0]-m_rc_mean[:,1]*m_rc_mean[:,2])/(m_rc_mean[:,1]*m_rc_mean[:,2])
m_corr_0 = (m_rc_mean[:,3]-m_rc_mean[:,4]*m_rc_mean[:,5])/(m_rc_mean[:,4]*m_rc_mean[:,5])
plt.plot(x_axis,m_corr,'red',label='sampled over a period of time')
plt.plot(x_axis,m_corr_0,'blue',label='sampled at a fixed time')
plt.legend()
plt.title("Time correlation function of sensor and responsor")
plt.xlabel('\u03C4')
plt.savefig('.\\resl_corr\\correlation_curve.png')
# plt.show()




# plotting trajectory
# plt.figure()
# plt.plot(timepoints,m_traj[:,1]+2,'r.--',label='(m,a)')
# plt.plot(timepoints,m_traj[:,2],'b.--',label='y')
# plt.xlabel('timestep')
# plt.ylabel('trajectory')
# plt.legend()
# plt.show()