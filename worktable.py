import numpy as np
import numpy.matlib
import numpy.linalg
import matplotlib.pyplot as plt
import ness_static as sts

# 28.03.2023 test
# m_x1 = 1
# m_x2 = 1
# m_h = 4.0
# m_g = 15
# filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
# with open(filename,'r') as f:
#     m_dist = np.loadtxt(f,delimiter='\n')
# m_energy = np.array([0,0,0,m_h,m_h,m_h,m_h,0])
# p_y0 = 0
# p_y1 = 0
# eave_y0 = 0
# eave_y1 = 0
# for i in range(4):
#     p_y0 = p_y0+m_dist[i]
#     p_y1 = p_y1+m_dist[i+4]
#     eave_y0 = eave_y0+m_dist[i]*m_energy[i]
#     eave_y1 = eave_y1+m_dist[i+4]*m_energy[i+4]
# eave_y0 = eave_y0/p_y0
# eave_y1 = eave_y1/p_y1
# ratio_eave = np.exp(eave_y1-eave_y0)
# ratio_ebar = p_y0/p_y1
# print("ave:",ratio_eave,'\nBar:',ratio_ebar)

# 02.05.2023 test
m_x1 = 1
m_x2 = 1
m_h = 7
m_g = 15

m_colorbar = ['k','b','c','g','y','r','m','violet']

plt.figure()
for j in np.arange(5):
    m_h = 2*j+1
    m_prec_ana = np.zeros(21)
    m_prec_ana_appr = np.zeros(21)
    m_prec_num = np.zeros(21)
    for i in range(21):
        m_g = i 
        A = np.exp(m_h/2)
        B = np.exp(m_g/2)
        m_sum = np.zeros(4) # compute analysitic precision
        m_s111 = np.zeros(4)
        m_s111[0] = 16*A*(3/B+B)
        m_sum[0] = 16*(4/B+4*B)*(A+2+1/A)
        m_s111[1] = 8*A*(5*B*B+10+5/(B*B))
        # m_sum[1] = 8*(1/A*(7*B*B+26+7/(B*B))+2*(10*B*B+20+10/(B*B))+A*(13*B*B+14+13/(B*B)))
        m_sum[1] = 16*((5*B*B+10+5/(B*B)))*(A+2+1/A)
        m_s111[2] = 8*A*(3*np.power(B,3)+7*B+5/B+1/np.power(B,3))
        m_sum[2] = 8*(1/A*(4*np.power(B,3)+12*B+12/B+4/np.power(B,3))+\
        (8*np.power(B,3)+24*B+24/B+8/np.power(B,3))+A*(4*np.power(B,3)+12*B+12/B+4/np.power(B,3)))
        m_s111[3] = 4*A*(np.power(B,4)+3*B*B+3+1/(B*B))
        m_sum[3] = (np.power(B,4)+4*B*B+6+4/(B*B)+1/np.power(B,4))*4*(A+2+1/A)
        m_s_s111 = np.sum(m_s111)
        m_s_sum = np.sum(m_sum)
        m_prec_ana[i] = m_s_s111/m_s_sum
        m_prec_ana_appr[i] = A*B/(A+2+1/A)/(B+1/B) # m_s111[3]/m_sum[3]
        m_mat_k = sts.cal_mat_k(m_x1,m_x2,m_g,m_h)
        m_ness = sts.cal_ness_distribution(m_mat_k)
        m_prec_num[i] = m_ness[7]
    plt.plot(range(21),m_prec_ana,marker='o',linestyle='dashed',color=m_colorbar[j])
    plt.plot(range(21),m_prec_ana_appr,marker='+',linewidth=0,color=m_colorbar[j])
    plt.plot(range(21),m_prec_num,color=m_colorbar[j],label='h_0=%d'%m_h)
plt.legend()
plt.xlabel('\u03B3')
plt.ylabel('precision')
plt.show()
