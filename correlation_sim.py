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
    while prob_cumu<=q:
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
    # r = np.random.rand()
    # dtime = -1/propen_sum*np.log(r)
    # choose what reaction to happen
    index = sample_discrete(propensity)
    return index, dtime

# Gillespie simulation algorithm
def gillespie_ssa(c_cal_propensity,timepoints,initial_index,bool_,*args):
    args1 = np.asarray(args).flatten()
    size = len(timepoints)
    record = np.zeros((size,2)) # record X and index
    record[0,:] = [3,initial_index]
    t = 0
    i = 0
    i_time = 1 # indexing the latter bound of a time interval
    index = initial_index
    if bool_==0: # args keep unchanged during simulation
        while i<size:
            while t<=timepoints[i_time]:
                index_now = index
                (index,dtime) = gillespie_draw(c_cal_propensity,index,*args1)
                t = t+dtime
            i = np.searchsorted(timepoints,t,side='left')
            for j in range(i_time,i):
                record[j,:] = [m_dict_xlabel[(args1[0],args1[1])],index_now]
            i_time = i
    if bool_==1:
        while i<size-1:
            while t<=timepoints[i_time]:
                index_now = index
                (index,dtime) = gillespie_draw(c_cal_propensity,index,*args1)
                if (t+dtime)<timepoints[i_time]:
                    t = t+dtime
                if (t+dtime)>=timepoints[i_time]:
                    # draw one single change
                    args1[0] = np.random.randint(2)
                    args1[1] = np.random.randint(2)
                    i = i+1
                    t = timepoints[i]
            for j in range(i_time,i):
                record[j,:] = [m_dict_xlabel[(args1[0],args1[1])],index_now]
            i_time = i
        record[size-1,:] = [m_dict_xlabel[(args1[0],args1[1])],index_now]
    return record

def test_gillespie_ssa(cal_propensity,timepoints,initial_index,*args): # arguments: x_1, x_2
    bool_ = 0
    m_h = 7
    m_dist_sto = np.zeros((21,8))
    m_dist_num = np.zeros((21,8))

    for m_g in tqdm.tqdm(range(0,21)):
        np.random.seed(m_g)
        m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,bool_,*args,m_g,m_h)
        t_cnt = 0
        for i in range(len(timepoints)):
            m_dist_sto[m_g,int(m_traj[i,1])] = m_dist_sto[m_g,int(m_traj[i,1])]+1
        m_mat_k = sts.cal_mat_k(m_x1,m_x2,m_g,m_h)
        m_dist_num_t = np.transpose(sts.cal_ness_distribution(m_mat_k))
        m_dist_num[m_g,:] = m_dist_num_t
        # plt.figure()  # plotting trajectory
        # plt.plot(timepoints,m_traj[:,1],'r.--',label='state label')
        # plt.xlabel('timestep')
        # plt.ylabel('trajectory')
        # plt.legend()
        # plt.show()
    m_dist_sto = m_dist_sto/len(timepoints)
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

def cal_correlation_S_Y(cal_propensity,timepoints,initial_index,m_smpl_seed,t_end,t_interval,*args): # arguments: x_1, x_2, \gamma, h
    m_smpl_tau = 250
    m_smpl_dt = 200

    m_dtsample_interval = int(1/t_interval)
    m_tausample_interval = 1
    timepoints = np.arange(0,t_end,t_interval)
    # running Gillespie simulation
    bool_ = 0
    m_trajs = np.zeros((m_smpl_seed,len(timepoints),2))
    print('Running Gillespie SSA over %d samples:'%m_smpl_seed)
    for i in tqdm.tqdm(range(m_smpl_seed)):
        seed = np.random.seed(2*i)
        m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,bool_,*args)  
        filename = f".\\resl_corr\\trajctry_{i}.dat"
        with open(filename,'w') as f:
            np.savetxt(f,m_traj,header='g=%.0f\th=%.0f\ttend=%.0f\ttinterval=%.4f\nX\t\tindex'%(m_g,m_h,t_end,t_interval),fmt="%d\t%d")
        m_trajs[i,:,:] = m_traj
    # # Measuring correlation function for Sensor and Y
    time_middle_i = int(len(timepoints)/2)
    print("Measuring correlation function for Sensor and Y")
    m_corr_t_s = np.zeros((2*m_smpl_dt+1,2*m_smpl_tau+1)) # record correlation function for different t
    for i in tqdm.tqdm(range(-m_smpl_dt,m_smpl_dt+1)):
        t_st = 0
        t_ytau = np.zeros(2*m_smpl_tau+1)
        t_sy = np.zeros(2*m_smpl_tau+1)
        t_t_i = time_middle_i+m_dtsample_interval*i
        for j in range(m_smpl_seed):
            t_ds = sts.m_x_label[int(m_trajs[j,t_t_i,1])]+1 # for clear calculation, add all the labels by 1
            t_st = t_st+t_ds
            t_t_j = t_t_i-m_tausample_interval*m_smpl_tau # set the starting poing in \tau calculation
            for k in range(2*m_smpl_tau+1):
                t_t_j = t_t_j+m_tausample_interval
                t_dy = sts.m_y_label[int(m_trajs[j,t_t_j,1])]+1 # for clear calculation, add all the labels by 1
                t_ytau[k] = t_ytau[k]+t_dy
                t_sy[k] = t_sy[k]+t_ds*t_dy
        t_st = t_st/m_smpl_seed # sampling average of X
        t_ytau = t_ytau/m_smpl_seed # sampling average of Y
        t_sy = t_sy/m_smpl_seed # sampling average of X*Y
        t_corr_t_s = np.zeros(2*m_smpl_tau+1)
        for k in range(2*m_smpl_tau+1):
            t_corr_t_s[k] = (t_sy[k]-t_st*t_ytau[k])/(t_st*t_ytau[k])
        m_corr_t_s[i,:] = t_corr_t_s
    t0_i = m_smpl_dt
    m_corr_0_s = m_corr_t_s[t0_i,:]
    m_corr_s = np.mean(m_corr_t_s,axis=0)
    filename = f".\\resl_corr\\corr_0_s_%d.dat"%args[2]
    with open(filename,'w') as f:
        np.savetxt(f,np.transpose(m_corr_0_s))
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
    
def cal_correlation_X_Y(cal_propensity,timepoints,initial_index,m_smpl_seed,t_end,t_interval,*args): # arguments: x_1, x_2, \gamma, h
    m_smpl_tau = 250
    m_smpl_dt = 200
    m_dtsample_interval = int(1/t_interval)
    m_tausample_interval = 1
    timepoints = np.arange(0,t_end,t_interval)
    # running Gillespis simulation
    bool_=1
    m_trajs = np.zeros((m_smpl_seed,len(timepoints),2))
    print('Running Gillespie SSA over %d samples:'%m_smpl_seed)
    for i in tqdm.tqdm(range(m_smpl_seed)):
        seed = np.random.seed(i+1)
        m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,bool_,*args)  
        filename = f".\\resl_corr\\trajctry_{i}.dat"
        with open(filename,'w') as f:
            np.savetxt(f,m_traj,header='g=%.0f\th=%.0f\ttend=%.0f\ttinterval=%.4f\nX\t\tindex'%(m_g,m_h,t_end,t_interval),fmt="%d\t%d")
        m_trajs[i,:,:] = m_traj
    # # Measuring correlation function for X and Y
    time_middle_i = int(len(timepoints)/2)
    print("Measuring correlation function for X and Y")
    m_corr_t = np.zeros((2*m_smpl_dt+1,2*m_smpl_tau+1)) # record correlation function for different t
    for i in tqdm.tqdm(range(-m_smpl_dt,m_smpl_dt+1)):
        t_xt = 0
        t_ytau = np.zeros(2*m_smpl_tau+1)
        t_xy = np.zeros(2*m_smpl_tau+1)
        for j in range(m_smpl_seed):
            t_t_i = time_middle_i+m_dtsample_interval*i
            t_dx = m_trajs[j,t_t_i,0]+1 # for clear calculation, add all the labels by 1
            t_xt = t_xt+t_dx
            t_t_i = time_middle_i+m_dtsample_interval*i-m_tausample_interval*m_smpl_tau # set the starting poing in \tau calculation
            for k in range(2*m_smpl_tau+1):
                t_t_i = t_t_i+m_tausample_interval
                t_dy = sts.m_y_label[int(m_trajs[j,t_t_i,1])]+1 # for clear calculation, add all the labels by 1
                t_ytau[k] = t_ytau[k]+t_dy
                t_xy[k] = t_xy[k]+t_dx*t_dy
        t_xt = t_xt/m_smpl_seed # sampling average of X
        t_ytau = t_ytau/m_smpl_seed # sampling average of Y
        t_xy = t_xy/m_smpl_seed # sampling average of X*Y
        t_corr_t = np.zeros(2*m_smpl_tau+1)
        for k in range(2*m_smpl_tau+1):
            t_corr_t[k] = (t_xy[k]-t_xt*t_ytau[k])/(t_xt*t_ytau[k])
        m_corr_t[i,:] = t_corr_t
    t0_i = m_smpl_dt
    m_corr_0 = m_corr_t[t0_i,:]
    m_corr = np.mean(m_corr_t,axis=0)
    filename = f".\\resl_corr\\corr_0.dat"
    with open(filename,'w') as f:
        np.savetxt(f,np.transpose(m_corr_0))
    filename = f".\\resl_corr\\corr.dat"
    with open(filename,'w') as f:
        np.savetxt(f,np.transpose(m_corr))
    plt.figure()    # plotting correlation curve
    x_axis = np.arange(-m_smpl_tau*t_interval*m_tausample_interval,(m_smpl_tau+1)*t_interval*m_tausample_interval,t_interval*m_tausample_interval)
    plt.plot(x_axis,m_corr_0,'blue',label='sampled at a fixed time')
    plt.plot(x_axis,m_corr,'red',label='sampled over a period of time')
    plt.legend()
    plt.title("Time correlation function of input and responsor")
    plt.xlabel('\u03C4')
    plt.savefig('.\\resl_corr\\correlation_curve.png')
    return m_corr

###################################### MAIN ##################################################

## Console

# modificating the system
m_x1 = 1
m_x2 = 1
m_g = 10
m_h = 7

# modificating sampling size
m_smpl_seed = 400

t_end = 1000 
t_interval = 0.05

timepoints = np.arange(0,t_end,t_interval)
initial_index = 4

m_g_num = 5

# testing Gillespie algorithm
test_gillespie_ssa(cal_propensity,timepoints,initial_index,m_x1,m_x2)

# # computing correlation function S-Y at different \gamma
# m_colorbar = ['k','b','c','g','y','r','m','violet']
# m_smpl_tau = 250
# m_smpl_dt = 200

# m_dtsample_interval = int(1/t_interval)
# m_tausample_interval = 1
# x_axis = np.arange(-m_smpl_tau*t_interval*m_tausample_interval,(m_smpl_tau+1)*t_interval*m_tausample_interval,t_interval*m_tausample_interval)

# m_g_list = np.arange(m_g_num)*2
# m_corr_list = np.zeros((m_g_num,2*m_smpl_tau+1))
# plt.figure()
# for i in range(m_g_num):
#     m_corr_list[i,:] = \
#     cal_correlation_S_Y(cal_propensity,timepoints,initial_index,m_smpl_seed,t_end,t_interval,m_x1,m_x2,m_g_list[i],m_h)
#     plt.plot(x_axis,m_corr_list[i,:],color=m_colorbar[i],label='\u03B3=%.1f'%m_g_list[i])
# plt.legend()
# plt.show()
