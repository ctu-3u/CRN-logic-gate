import numpy as np
import numpy.matlib
import numpy.linalg

#28.03.2023 test
m_x1 = 1
m_x2 = 1
m_h = 4.0
m_g = 15
filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
with open(filename,'r') as f:
    m_dist = np.loadtxt(f,delimiter='\n')
m_energy = np.array([0,0,0,m_h,m_h,m_h,m_h,0])
p_y0 = 0
p_y1 = 0
eave_y0 = 0
eave_y1 = 0
for i in range(4):
    p_y0 = p_y0+m_dist[i]
    p_y1 = p_y1+m_dist[i+4]
    eave_y0 = eave_y0+m_dist[i]*m_energy[i]
    eave_y1 = eave_y1+m_dist[i+4]*m_energy[i+4]
eave_y0 = eave_y0/p_y0
eave_y1 = eave_y1/p_y1
ratio_eave = np.exp(eave_y1-eave_y0)
ratio_ebar = p_y0/p_y1
print("Ave:",ratio_eave,'\nBar:',ratio_ebar)

#