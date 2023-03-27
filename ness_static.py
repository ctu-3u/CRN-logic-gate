import numpy as np
import numpy.matlib
import numpy.linalg

m_index_start_v = np.array([0,2,5,7,0,1,6,7,4,5,6,3])
m_index_end_v = np.array([1,3,4,6,2,3,4,5,0,1,2,7])
m_m_v = np.array([0,1,0,1,0,1,0,1]) # v means "vortex"
m_a_v = np.array([0,0,1,1,0,0,1,1])
m_y_v = np.array([0,0,0,0,1,1,1,1])


# vortex energy level
def cal_energy_level(h,i):
    # i is the index of vortex
    energy = h*np.square(m_y_v[i]-m_m_v[i]*m_a_v[i])
    return energy

# non-equilibrium driving force, positive direction
# i is the start vortex index, j is the end vortex index
def cal_driving(x1,x2,g,i,j):
    driving = -g/2*((x1-0.5)*(m_m_v[i]-0.5)+(x2-0.5)*(m_a_v[i]-0.5))
    return driving

# jumping rate
# i is the index of starting vortex, j is the index of ending vortex
def cal_k_ij(i,j,list_energy,mat_driving):
    connect = 0
    jumping_rate = 0
    for index in range(12):
        if m_index_start_v[index] == i and m_index_end_v[index] == j:
            connect = 1
        elif m_index_start_v[index] == j and m_index_end_v[index] == i:
            connect = 1
    if connect == 1:
        H_v = list_energy[j]
        d_e = mat_driving[i,j] # e means "edge"
        jumping_rate = np.exp(-H_v+d_e)
    return jumping_rate

# compute jumping rates' matrix
def cal_mat_k(x1,x2,g,h):
    mat_k = np.matlib.zeros((8,8))
    list_energy = np.zeros(8)
    mat_driving = np.matlib.zeros((8,8))
    for i in range(8):
        list_energy[i] = cal_energy_level(h,i)
    for i in range(8):
        for j in range(8):
            mat_driving[i,j] = cal_driving(x1,x2,g,i,j)
    for i in range(8):
        for j in range(8):
            mat_k[i,j] = cal_k_ij(i,j,list_energy,mat_driving)
    # print(mat_k)
    # print(list_energy)
    # print(mat_driving)
    return mat_k

# compute probability distribution at NESS
def cal_ness_distribution(mat_k):
    mat_master = np.matlib.ones((8,8)) # "transition" matrix
    k_ii = 0
    for i in range(7):
        for j in range(8):
            mat_master[i,j] = mat_k[j,i]
            k_ii = k_ii - mat_k[i,j]
        mat_master[i,i] = k_ii
        k_ii = 0
    mat_null = np.matlib.zeros((8,1))
    mat_null[7,0] = 1
    # print(mat_master)
    mat_master = np.linalg.inv(mat_master)
    mat_ness = np.matmul(mat_master,mat_null)
    return mat_ness

## compute mutual information flows
##

############################################ MAIN #########################################################

m_h = 1 # h_0
m_g = 10 # \gamma
m_x1 = 1
m_x2 = 1

for i in range(0,8):
    for j in range(0,31):
        m_h = 1+1.5*i
        m_g = j
        m_mat_k = cal_mat_k(m_x1,m_x2,m_g,m_h)
        m_ness = cal_ness_distribution(m_mat_k)
        filename = f".\\resu_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
        with open(filename,'w') as f:
            np.savetxt(f,m_ness)
        # print([i,j]) # just to monitor the process





