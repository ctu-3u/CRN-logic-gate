import numpy as np
import numpy.matlib
import numpy.linalg


#***#
m_index_start_v = np.array([0,2,5,7,0,1,6,7,4,5,6,3])
m_index_end_v = np.array([1,3,4,6,2,3,4,5,0,1,2,7])
m_m_v = np.array([0,1,0,1,0,1,0,1]) # v means "vortex"
m_a_v = np.array([0,0,1,1,0,0,1,1])
m_y_v = np.array([0,0,0,0,1,1,1,1])
#***#

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
    mat_master = np.linalg.inv(mat_master)
    mat_ness = np.matmul(mat_master,mat_null)
    return mat_ness

# compute mutual information flows
def cal_idot(mat_dist,mat_k):
    idot_x = 0 # information flux in x domain
    idot_y = 0 # information flux in y domain
    #***#
    x_cate = np.array([[0,4],[1,5],[2,6],[3,7]]) # categorize states with the same "x1,x2" keys. {(0,0),(1,0),(0,1),(1,1)}
    y_cate = np.array([[0,1,2,3],[4,5,6,7]]) # categorize states with the same "y" key. {0,1}
    x_label = np.array([0,1,2,3,0,1,2,3]) # category of "x1,x2" keys for each state
    y_label = np.array([0,0,0,0,1,1,1,1]) # category of "y" key for each state
    #***#
    num_x = x_cate.shape[0] # number of (x1,x2)
    num_incate_x = x_cate.shape[1] # number of states in any (x1,x2)
    num_y = y_cate.shape[0] # number of (y)
    num_incate_y = y_cate.shape[1] # number of states in any (y)
    prob_x = np.zeros(num_x) # p(x)
    prob_y = np.zeros(num_y) # p(y)
    for i in range(num_x): # compute p(x)
        for j in range(num_incate_x):
            prob_x[i] = prob_x[i]+mat_dist[x_cate[i][j]]
    for i in range(num_y): # compute p(y)
        for j in range(num_incate_y):
            prob_y[i] = prob_y[i]+mat_dist[y_cate[i][j]]
    for i in range(num_y): # compute idot_x
        for j in range(num_incate_y-1):
            for k in range(j+1,num_incate_y):
                s_xp = y_cate[i][j]
                s_x = y_cate[i][k]
                flux = mat_k[s_xp,s_x]*mat_dist[s_xp]-mat_k[s_x,s_xp]*mat_dist[s_x]
                stoc_entp = np.log(mat_dist[s_x]*prob_x[x_label[s_xp]]/(mat_dist[s_xp]*prob_x[x_label[s_x]]))
                idot_x = idot_x + flux*stoc_entp
    for i in range(num_x): # compute idot_x
        for j in range(num_incate_x-1):
            for k in range(j+1,num_incate_x):
                s_yp = x_cate[i][j]
                s_y = x_cate[i][k]
                flux = mat_k[s_yp,s_y]*mat_dist[s_yp]-mat_k[s_y,s_yp]*mat_dist[s_y]
                stoc_entp = np.log(mat_dist[s_y]*prob_y[y_label[s_yp]]/(mat_dist[s_yp]*prob_y[y_label[s_y]]))
                idot_y = idot_y + flux*stoc_entp
    return idot_x,idot_y

# compute effective intrinsic jumping rate \omega
def cal_intrin_rt(mat_dist,mat_k,mat_ene): # need to know distribution, "real" jumping rates and states' energy before coarse graining
    p_y0 = 0
    p_y1 = 0
    Eave_y0 = 0
    Eave_y1 = 0
    for i in range(4):
        p_y0 = p_y0+mat_dist[i]
        p_y1 = p_y1+mat_dist[i+4]
        Eave_y0 = Eave_y0+mat_dist[i]*mat_ene[i]
        Eave_y1 = Eave_y1+mat_dist[i+4]*mat_ene[i+4]
    Eave_y0 = Eave_y0/p_y0
    Eave_y1 = Eave_y1/p_y1
    Ebar_y0 = p_y1*Eave_y1+p_y0*Eave_y0-p_y1*np.log(p_y0/p_y1)
    Ebar_y1 = Ebar_y0+np.log(p_y0/p_y1)
    delta_effc = (Eave_y1-Eave_y0)-(Ebar_y1-Ebar_y0) # effective driving after coarse graining
    # compute \omega via "0 to 1" and "1 to 0" jummping respectively
    block_down = 0
    block_up = 0
    for i in range(4):
        block_down = block_down+mat_k[i,i+4]*mat_dist[i]
        block_up = block_up+mat_k[i+4,i]*mat_dist[i+4]
    omega_down = block_down/(p_y0*np.exp(-Ebar_y1))
    omega_up = block_up/(p_y1*np.exp(-Ebar_y0)) # theoretically up and down \omega should be equal. they actually are
    return delta_effc,(omega_down+omega_up)/2

############################################ MAIN #########################################################

## Analysis

m_x1 = 1
m_x2 = 1

# computing NESS distributions
for i in range(0,8):
    for j in range(0,31):
        m_h = 1+1.5*i # h_0
        m_g = j # \gamma
        m_mat_k = cal_mat_k(m_x1,m_x2,m_g,m_h)
        m_ness = cal_ness_distribution(m_mat_k)
        filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
        with open(filename,'w') as f:
            np.savetxt(f,m_ness)

# computing effective intrinsic jumping rate
filename_record = f".\\resl_ness_ana\\idot_flux.dat"
with open(filename_record,'w') as record:
    np.savetxt(record,[],header='h_0\tg\tiX\tiY\t')
    for i in range(0,8):
        for j in range(0,31):
            m_h = 1+1.5*i
            m_g = j
            m_mat_k = cal_mat_k(m_x1,m_x2,m_g,m_h)
            filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
            with open(filename,'r') as f:
                m_dist_rd = np.loadtxt(f,delimiter='\n') # distribution readout
            (m_idot_x,m_idot_y) = cal_idot(m_dist_rd,m_mat_k)
            #print("h:{:.1f}\tg:{:2.0f}\tiX:{:}\tiY:{:}\n".format(m_h,m_g,m_idot_x,m_idot_y)) # print to screen
            record_row = np.array([m_h,m_g,m_idot_x,m_idot_y])
            np.savetxt(record,[record_row],fmt="%.1f\t%d\t%.18e\t%.18e")

# computing mutual information flow
m_energy_nm = np.array([0,0,0,1,1,1,1,0])  # normailized state energies

filename_record = f".\\resl_ness_ana\\effc_rt.dat"
with open(filename_record,'w') as record:
    np.savetxt(record,[],header='h_0\tg\tdelta_effc\t omega')
    for i in range(0,8):
        for j in range(0,31):
            m_h = 1+1.5*i
            m_g = j
            m_mat_k = cal_mat_k(m_x1,m_x2,m_g,m_h)
            m_energy = m_h*m_energy_nm
            filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
            with open(filename,'r') as f:
                m_dist_rd = np.loadtxt(f,delimiter='\n') # distribution readout
            (delta_effc,omega) = cal_intrin_rt(m_dist_rd,m_mat_k,m_energy)
            #print("h:{:.1f}\tg:{:2.0f}\tiX:{:}\tiY:{:}\n".format(m_h,m_g,m_idot_x,m_idot_y)) # print to screen
            record_row = np.array([m_h,m_g,delta_effc,omega])
            np.savetxt(record,[record_row],fmt="%.1f\t%d\t%.7e\t%.7e")
            # print(record_row) # print to screen

## Plot results
