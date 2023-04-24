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
                (index,dtime) = gillespie_draw(c_cal_propensity,index,*args1)
                t = t+dtime
            i = np.searchsorted(timepoints,t,side='left')
            for j in range(i_time,i):
                record[j,:] = [m_dict_xlabel[(args1[0],args1[1])],index]
            i_time = i
    if bool_==1:
        while i<size:
            while t<=timepoints[i_time]:
                # draw one single change
                args1[0] = np.random.randint(2)
                args1[1] = np.random.randint(2)
                (index,dtime) = gillespie_draw(c_cal_propensity,index,*args1)
                t = t+dtime
            i = np.searchsorted(timepoints,t,side='left')
            for j in range(i_time,i):
                record[j,:] = [m_dict_xlabel[(args1[0],args1[1])],index]
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
m_smpl_seed = 200
m_smpl_tau = 250
m_smpl_dt = 100

t_end = 1000
t_interval = 0.05
timepoints = np.arange(0,t_end,t_interval)
initial_index = 7

bool_ = 0

## Testing Gillespie algorithm
m_crrness_sto = np.zeros(21)
for m_g in range(0,21):
    np.random.seed(m_g)
    m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,bool_,m_x1,m_x2,m_g,m_h) 
    # plt.figure()  # plotting trajectory
    # plt.plot(timepoints,m_traj[:,1]+2,'r.--',label='(m,a)')
    # plt.plot(timepoints,m_traj[:,2],'b.--',label='y')
    # plt.xlabel('timestep')
    # plt.ylabel('trajectory')
    # plt.legend()
    # plt.show()
    t_cnt = 0
    for i in range(len(timepoints)):
        if m_traj[i,1]>=4:
            t_cnt = t_cnt+1
    t_cnt = t_cnt/len(timepoints)
    m_crrness_sto[m_g] = t_cnt
m_crrness_num = np.zeros(21)
filename = f"..\\RESULTS\\contour_x1_1x2_0\\correctness.dat"
with open(filename,'r') as f:
    num_resl = np.loadtxt(f)
    for i in range(21):
        m_crrness_num[i] = num_resl[8*(i+1)-1,3]
plt.figure()
plt.plot(range(21),m_crrness_num,color='blue',label='numerical result')
plt.plot(range(21),m_crrness_sto,color='red',label='stochastical simulation result')
plt.ylabel("probability")
plt.xlabel("\u03B3")
plt.legend()
plt.title("Probability at state y=1, config h_0=7")
plt.show()

## Running

# running Gillespie simulation
bool_ = 1
m_trajs = np.zeros((m_smpl_seed,len(timepoints),2))

print('Running Gillespie SSA over %d samples:'%m_smpl_seed)
for i in tqdm.tqdm(range(m_smpl_seed)):
    seed = np.random.seed(i+1)
    m_traj = gillespie_ssa(cal_propensity,timepoints,initial_index,bool_,m_x1,m_x2,m_g,m_h)    
    filename = f".\\resl_corr\\trajctry_{i}.dat"
    with open(filename,'w') as f:
        np.savetxt(f,m_traj,header='g=%.0f\th=%.0f\ttend=%.0f\ttinterval=%.0f\nX\t\tindex'%(m_g,m_h,t_end,t_interval),fmt="%d%d")
    m_trajs[i,:,:] = m_traj

# Measuring correlation function
time_middle_i = int(len(timepoints)/2)

print("Measuring correlation function")
m_corr_t = np.zeros((2*m_smpl_dt+1,2*m_smpl_tau+1)) # record correlation function for different t
for i in tqdm.tqdm(range(-m_smpl_dt,m_smpl_dt+1)):
    t_xt = 0
    t_ytau = np.zeros(2*m_smpl_tau+1)
    t_xy = np.zeros(2*m_smpl_tau+1)
    for j in range(m_smpl_seed):
        t_t_i = time_middle_i+20*i
        t_dx = m_trajs[j,t_t_i,0]+1 # for clear calculation, add all the labels by 1
        t_xt = t_xt+t_dx
        t_t_i = time_middle_i+20*i-20*m_smpl_tau # set the starting poing in \tau calculation
        for k in range(2*m_smpl_tau+1):
            t_t_i = t_t_i+20
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
x_axis = range(-m_smpl_tau,m_smpl_tau+1)
plt.plot(x_axis,m_corr,'red',label='sampled over a period of time')
plt.plot(x_axis,m_corr_0,'blue',label='sampled at a fixed time')
plt.legend()
plt.title("Time correlation function of sensor and responsor")
plt.xlabel('\u03C4')
plt.savefig('.\\resl_corr\\correlation_curve.png')