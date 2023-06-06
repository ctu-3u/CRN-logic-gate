import numpy as np
import numpy.linalg

import scipy.stats as st
import tqdm
import numba

import matplotlib as mpl
import matplotlib.pyplot as plt

import ness_static as sts

m_dict_connection = {0:[1,2,4],1:[0,3,5],2:[0,3,6],3:[1,2,7],\
4:[0,5,6],5:[1,4,7],6:[2,4,7],7:[3,5,6]} # connected states
m_dict_xlabel = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}

m_reaction = np.array([[-1,1,0,0,0,0,0,0],[-1,0,1,0,0,0,0,0],[-1,0,0,0,1,0,0,0],\
[1,-1,0,0,0,0,0,0],[0,-1,0,1,0,0,0,0],[0,-1,0,0,0,1,0,0],\
[1,0,-1,0,0,0,0,0],[0,0,-1,1,0,0,0,0],[0,0,-1,0,0,0,1,0],\
[0,1,0,-1,0,0,0,0],[0,0,1,-1,0,0,0,0],[0,0,0,-1,0,0,0,1],\
[1,0,0,0,-1,0,0,0],[0,0,0,0,-1,1,0,0],[0,0,0,0,-1,0,1,0],\
[0,1,0,0,0,-1,0,0],[0,0,0,0,1,-1,0,0],[0,0,0,0,0,-1,0,1],\
[0,0,1,0,0,0,-1,0],[0,0,0,0,1,0,-1,0],[0,0,0,0,0,0,-1,1],\
[0,0,0,1,0,0,0,-1],[0,0,0,0,0,1,0,-1],[0,0,0,0,0,0,1,-1]]) # reaction prompts

# compute propensity
def cal_propensity(states,*args):
    mat_k = sts.cal_mat_k(*args)
    prpn = np.zeros(24)
    pt = 0
    for i in range(0,8):
        for j in range(0,8):
            if mat_k[i,j] > 0:
                prpn[pt] = mat_k[i,j]*states[i]/1000
                pt = pt+1
    return prpn

# draw out which reaction to happen according to discrete distribution
def sample_discrete(probs):
    q = np.random.rand()
    i = 0
    prob_cumu = 0
    while prob_cumu<=q:
        prob_cumu = prob_cumu+probs[i]
        i = i+1
    return i-1

# draw out change in one single step
def gillespie_draw(c_cal_propensity,states,*args):
    propensity = c_cal_propensity(states,*args)
    propen_sum = np.sum(propensity)
    propensity = propensity/propen_sum
    # choose reaction interval time
    dtime = np.random.exponential(1/propen_sum)
    # r = np.random.rand()
    # dtime = -1/propen_sum*np.log(r)
    # choose what reaction to happen
    index = sample_discrete(propensity) # index is the index of reactions to occur
    return index, dtime

# Gillespie simulation algorithm
def gillespie_ssa(c_cal_propensity,timepoints,initial_state,*args):
    args1 = np.asarray(args).flatten()
    size = len(timepoints)
    record = np.zeros((size,9)) # record time and state-index
    record[0,:] = np.append([0],initial_state)
    t = 0
    i = 0
    i_time = 1 # indexing the latter bound of a time interval
    state = initial_state
    while i<size:
        while t<=timepoints[i_time]:
            # index_now = index
            state_now = state/1000
            (index,dtime) = gillespie_draw(c_cal_propensity,state,*args1)
            t = t+dtime
            state = state + m_reaction[index]
        i = np.searchsorted(timepoints,t,side='left')
        for j in range(i_time,i):
            record[j,:] = np.append(timepoints[j],state_now)
        i_time = i
    return record

def test_gillespie_ssa(cal_propensity,timepoints,initial_index,*args): # arguments: x_1, x_2
    m_h = 7
    m_dist_sto = np.zeros((21,8))
    m_dist_num = np.zeros((21,8))
    half_time_i = int(len(timepoints)*0.9)

    for m_g in tqdm.tqdm(range(0,21)):
        np.random.seed(m_g)
        m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,*args,m_g,m_h)
        t_cnt = 0
        for i in range(half_time_i,len(timepoints)):
            m_dist_sto[m_g,:] = m_dist_sto[m_g,:]+m_traj[i,1:9]
        m_mat_k = sts.cal_mat_k(m_x1,m_x2,m_g,m_h)
        m_dist_num_t = np.transpose(sts.cal_ness_distribution(m_mat_k))
        m_dist_num[m_g,:] = m_dist_num_t
        # plt.figure()  # plotting trajectory
        # plt.plot(timepoints,m_traj[:,8])
        # plt.xlabel('timestep')
        # plt.ylabel('trajectory')
        # plt.legend()
        # plt.show()
    m_dist_sto = m_dist_sto/(len(timepoints)-half_time_i+1)
    # Plot result
    m_colorbar = ['k','b','c','g','y','r','m','violet']
    x_axis = range(21)
    plt.figure()
    for i in range(8):
        plt.plot(x_axis,m_dist_num[:,i],linestyle='dashed',color=m_colorbar[i])
        plt.plot(x_axis,m_dist_sto[:,i],color=m_colorbar[i],label='P(%d,%d,%d)'%(sts.m_m_v[i],sts.m_a_v[i],sts.m_y_v[i]))
    plt.legend()
    plt.xlabel('\u03B3')
    plt.ylabel('probability distribution')
    plt.show()

def cal_correlation_S_Y(cal_propensity,timepoints,initial_state,m_smpl_seed,t_end,t_interval,m_smpl_tau,m_smpl_dt,*args): # arguments: x_1, x_2, \gamma, h
    m_dtsample_interval = int(1/t_interval)
    m_tausample_interval = 1
    timepoints = np.arange(0,t_end,t_interval)
    # running Gillespie simulation
    bool_ = 0
    m_trajs = np.zeros((m_smpl_seed,len(timepoints),9))
    print('Running Gillespie SSA over %d samples:'%m_smpl_seed)
    for i in tqdm.tqdm(range(m_smpl_seed)):
        seed = np.random.seed(2*i)
        m_traj = gillespie_ssa(cal_propensity,timepoints,initial_state,*args)  
        filename = f".\\resl_corr\\trajctry_{i}.dat"
        with open(filename,'w') as f:
            np.savetxt(f,m_traj[:,1:9],header='g=%.0f\th=%.0f\ttend=%.0f\ttinterval=%.4f\nX\t\tindex'%(m_g,m_h,t_end,t_interval),\
            fmt="%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f")
        m_trajs[i,:,:] = m_traj
    # # Measuring correlation function for Sensor and Y
    time_middle_i = int(len(timepoints)*0.9)
    print("Measuring correlation function for Sensor and Y")
    x_label = [1,2,3,4,1,2,3,4]
    y_label = [1,1,1,1,2,2,2,2]
    m_corr_t_s = np.zeros((2*m_smpl_dt+1,2*m_smpl_tau+1)) # record correlation function for different t
    for i in tqdm.tqdm(range(-m_smpl_dt,m_smpl_dt+1)):
        t_st = 0
        t_ytau = np.zeros(2*m_smpl_tau+1)
        t_sy = np.zeros(2*m_smpl_tau+1)
        t_t_i = time_middle_i+m_dtsample_interval*i
        for j in range(m_smpl_seed):
            t_ds = np.dot(x_label,m_trajs[j,t_t_i,1:9])
            t_st = t_st+t_ds
            t_t_j = t_t_i-m_tausample_interval*m_smpl_tau # set the starting poing in \tau calculation
            for k in range(2*m_smpl_tau+1):
                t_t_j = t_t_j+m_tausample_interval
                t_dy = np.dot(y_label,m_trajs[j,t_t_j,1:9])
                t_ytau[k] = t_ytau[k]+t_dy
                for l in range(8):
                    for m in range(8):
                        t_sy[k] = t_sy[k]+x_label[l]*y_label[m]*m_trajs[j,t_t_i,l+1]*m_trajs[j,t_t_j,m+1]
        t_st = t_st/m_smpl_seed # sampling average of X
        t_ytau = t_ytau/m_smpl_seed # sampling average of Y
        t_sy = t_sy/m_smpl_seed # sampling average of X*Y
        t_corr_t_s = np.zeros(2*m_smpl_tau+1)
        for k in range(2*m_smpl_tau+1):
            t_corr_t_s[k] = (t_sy[k]-t_st*t_ytau[k])/(t_st*t_ytau[k])
        m_corr_t_s[i,:] = t_corr_t_s
    m_corr_s = np.mean(m_corr_t_s,axis=0)
    filename = f".\\resl_corr\\corr_s_%d.dat"%args[2]
    with open(filename,'w') as f:
        np.savetxt(f,np.transpose(m_corr_s))
    # plt.figure()    # plotting correlation curve
    # x_axis = np.arange(-m_smpl_tau*t_interval*m_tausample_interval,(m_smpl_tau+1)*t_interval*m_tausample_interval,t_interval*m_tausample_interval)
    # plt.plot(x_axis,m_corr_0_s,'blue',label='sampled at a fixed time')
    # plt.plot(x_axis,m_corr_s,'red',label='sampled over a period of time')
    # plt.legend()
    # plt.title("Time correlation function of sensor and responsor")
    # plt.xlabel('\u03C4')
    # #plt.savefig('.\\resl_corr\\correlation_curve_s.png')
    # plt.show()
    return m_corr_s

###################################### MAIN ##################################################

## Console

# modificating the system
m_x1 = 1
m_x2 = 1
m_g = 10
m_h = 7

# modificating sampling size
m_smpl_seed = 2

t_end = 30000
t_interval = 0.05

timepoints = np.arange(0,t_end,t_interval)

m_g_num = 5

# initialization
initial_state =np.array([1000,0,0,0,0,0,0,0]) #  initial distribution

# # testing Gillespie algorithm
# test_gillespie_ssa(cal_propensity,timepoints,initial_state,m_x1,m_x2)

# computing correlation function S-Y at different \gamma
m_colorbar = ['k','b','c','g','y','r','m','violet']
m_smpl_tau = 600
m_smpl_dt = 500

m_dtsample_interval = int(1/t_interval)
m_tausample_interval = 1
x_axis = np.arange(-m_smpl_tau*t_interval*m_tausample_interval,(m_smpl_tau+1)*t_interval*m_tausample_interval,t_interval*m_tausample_interval)

m_g_list = np.arange(m_g_num)*2
m_corr_list = np.zeros((m_g_num,2*m_smpl_tau+1))
plt.figure()
for i in range(m_g_num):
    m_corr_list[i,:] = \
    cal_correlation_S_Y(cal_propensity,timepoints,initial_state,m_smpl_seed,t_end,t_interval,m_smpl_tau,m_smpl_dt,m_x1,m_x2,m_g_list[i],m_h)
    plt.plot(x_axis,m_corr_list[i,:],color=m_colorbar[i],label='\u03B3=%.1f'%m_g_list[i])
plt.legend()
plt.show()