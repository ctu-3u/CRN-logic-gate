import numpy as np
import numpy.matlib
import numpy.linalg


#***# configure network structure
m_index_start_v = np.array([0,2,5,7,0,1,6,7,4,5,6,3])
m_index_end_v = np.array([1,3,4,6,2,3,4,5,0,1,2,7])
m_m_v = np.array([0,1,0,1,0,1,0,1]) # v means "vortex"
m_a_v = np.array([0,0,1,1,0,0,1,1])
m_y_v = np.array([0,0,0,0,1,1,1,1])
m_x_cate = np.array([[0,4],[1,5],[2,6],[3,7]]) # categorize states with the same "x1,x2" keys. {(0,0),(1,0),(0,1),(1,1)}
m_y_cate = np.array([[0,1,2,3],[4,5,6,7]]) # categorize states with the same "y" key. {0,1}
m_x_label = np.array([0,1,2,3,0,1,2,3]) # category of "x1,x2" keys for each state
m_y_label = np.array([0,0,0,0,1,1,1,1]) # category of "y" key for each state
m_kB = 1 # Boltzmann constant
m_T = 1 # temperature
m_beta = 1/(m_kB*m_T)
#***#

# vortex energy level, equilibrium model 1
def cal_energy_level_mdl1(x1,x2,h,i):
    # i is the index of vortex
    energy = h*np.square(m_y_v[i]-m_m_v[i]*m_a_v[i])
    return energy

# added external field, equilibrium model 1
# i is the start vortex index, j is the end vortex index
def cal_driving_mdl1(x1,x2,g,i,j):
    driving = -g/2*((x1-0.5)*(m_m_v[i]-0.5)+(x2-0.5)*(m_a_v[i]-0.5))
    return driving

# vortex energy level, non-equilibrium model 2
def cal_energy_level(x1,x2,h,i):
    # i is the index of vortex
    energy = -h*((x1-0.5)*(m_m_v[i]-0.5)+(x2-0.5)*(m_a_v[i]-0.5))
    return energy

# external driving, non-equilibrium model 2
# i is the start vortex index, j is the end vortex index
def cal_driving(x1,x2,g,i,j):
    driving = 0
    if i-j==4 or j-i==4:
        driving = g*(np.square(m_y_v[i]-m_m_v[i]*m_a_v[i])-0.5)
    # if i-j==4: # inhibitor
    #     driving = 2.1*g*(np.square(m_y_v[i]-m_m_v[i]*m_a_v[i])-0.5)
    # if j-i==4: # promotor
    #     driving = -0.1*g*(np.square(m_y_v[i]-m_m_v[i]*m_a_v[i])-0.5)
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
        H_v = list_energy[i]
        d_e = mat_driving[i,j] # e means "edge"
        jumping_rate = np.exp(m_beta*(H_v+d_e))
    return jumping_rate

# compute jumping rates' matrix
def cal_mat_k(x1,x2,g,h):
    mat_k = np.matlib.zeros((8,8))
    list_energy = np.zeros(8)
    mat_driving = np.matlib.zeros((8,8))
    for i in range(8):
        list_energy[i] = cal_energy_level(x1,x2,h,i)
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

# compute (decomposed) flux
def cal_J_flux(mat_dist,mat_k):
    mat_J = np.matlib.zeros((8,8))
    J_x = 0
    J_y = 0
    for j in range(7):
        for i in range(j+1,8): # only calculate jumping x > x'
            mat_J[i,j] = mat_k[i,j]*mat_dist[i]-mat_k[j,i]*mat_dist[j]
            if (i-j)>=4:
                J_y = J_y+mat_J[i,j]
            if (i-j)<4:
                J_x = J_x+mat_J[i,j]
    return mat_J,J_x,J_y

# compute mutual information
def cal_I_mutlinfo(mat_dist):
    i_mi = 0
    num_x = m_x_cate.shape[0] # number of (x1,x2)
    num_incate_x = m_x_cate.shape[1] # number of states in any (x1,x2)
    num_y = m_y_cate.shape[0] # number of (y)
    num_incate_y = m_y_cate.shape[1] # number of states in any (y)
    prob_x = np.zeros(num_x) # p(x)
    prob_y = np.zeros(num_y) # p(y)
    for i in range(num_x): # compute p(x)
        for j in range(num_incate_x):
            prob_x[i] = prob_x[i]+mat_dist[m_x_cate[i][j]]
    for i in range(num_y): # compute p(y)
        for j in range(num_incate_y):
            prob_y[i] = prob_y[i]+mat_dist[m_y_cate[i][j]]
    for i in range(8):
        i_mi = i_mi+mat_dist[i]*np.log(mat_dist[i]/(prob_x[m_x_label[i]]*prob_y[m_y_label[i]]))
    return i_mi

# compute mutual information flows
def cal_idot(mat_dist,mat_k):
    idot_x = 0 # information flux in x domain
    idot_y = 0 # information flux in y domain
    num_x = m_x_cate.shape[0] # number of (x1,x2)
    num_incate_x = m_x_cate.shape[1] # number of states in any (x1,x2)
    num_y = m_y_cate.shape[0] # number of (y)
    num_incate_y = m_y_cate.shape[1] # number of states in any (y)
    prob_x = np.zeros(num_x) # p(x)
    prob_y = np.zeros(num_y) # p(y)
    for i in range(num_x): # compute p(x)
        for j in range(num_incate_x):
            prob_x[i] = prob_x[i]+mat_dist[m_x_cate[i][j]]
    for i in range(num_y): # compute p(y)
        for j in range(num_incate_y):
            prob_y[i] = prob_y[i]+mat_dist[m_y_cate[i][j]]
    for i in range(num_y): # compute idot_x
        for j in range(num_incate_y-1):
            for k in range(j+1,num_incate_y):
                s_xp = m_y_cate[i][j]
                s_x = m_y_cate[i][k]
                flux = mat_k[s_xp,s_x]*mat_dist[s_xp]-mat_k[s_x,s_xp]*mat_dist[s_x]
                stoc_entp = np.log(mat_dist[s_x]*prob_x[m_x_label[s_xp]]/(mat_dist[s_xp]*prob_x[m_x_label[s_x]]))
                idot_x = idot_x + flux*stoc_entp
    for i in range(num_x): # compute idot_x
        for j in range(num_incate_x-1):
            for k in range(j+1,num_incate_x):
                s_yp = m_x_cate[i][j]
                s_y = m_x_cate[i][k]
                flux = mat_k[s_yp,s_y]*mat_dist[s_yp]-mat_k[s_y,s_yp]*mat_dist[s_y]
                stoc_entp = np.log(mat_dist[s_y]*prob_y[m_y_label[s_yp]]/(mat_dist[s_yp]*prob_y[m_y_label[s_y]]))
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
    Ebar_y0 = p_y1*Eave_y1+p_y0*Eave_y0-p_y1*np.log(p_y0/p_y1)/m_beta
    Ebar_y1 = Ebar_y0+np.log(p_y0/p_y1)/m_beta
    delta_effc = (Eave_y1-Eave_y0)-(Ebar_y1-Ebar_y0) # effective driving after coarse graining
    # compute \omega via "0 to 1" and "1 to 0" jummping respectively
    block_down = 0
    block_up = 0
    for i in range(4):
        block_down = block_down+mat_k[i,i+4]*mat_dist[i]
        block_up = block_up+mat_k[i+4,i]*mat_dist[i+4]
    omega_down = block_down/(p_y0*np.exp(-Ebar_y1*m_beta))
    omega_up = block_up/(p_y1*np.exp(-Ebar_y0*m_beta)) # theoretically up and down \omega should be equal. they actually are
    return delta_effc,(omega_down+omega_up)/2

# compute total entropy production
def cal_entropy_prod_total(mat_dist,mat_k):
    (mat_J,J_x,J_y) = cal_J_flux(mat_dist,mat_k)
    heat_dssp = 0
    entp_mic_prod = 0
    for i in range(8):
        for j in range (8):
            if mat_k[i,j]==0 or mat_k[j,i]==0: # not calculate non-connective edges
                continue
            heat_dssp = heat_dssp+m_kB*mat_J[i,j]*np.log(mat_k[i,j]/mat_k[j,i])/2
            entp_mic_prod = entp_mic_prod+m_kB*mat_J[i,j]*np.log(mat_dist[i]/mat_dist[j])/2
    return heat_dssp,entp_mic_prod,(heat_dssp+entp_mic_prod)

