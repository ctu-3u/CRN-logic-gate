import numpy as np
import numpy.matlib
import numpy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import ness_static as sts

## Analysis

# console
M_Calc_Prop = 0
M_Plot_H_Haxis = 0
M_Plot_G_Haxis = 1
M_Plot_Contour = 1

# basic parameters
m_x1 = 1
m_x2 = 1
m_index_correct_state = 3
m_index_correct_output = range(0,4)

m_num_g = 31
m_num_h = 8
m_d_g = -1
m_inc_h = 0

m_energy_nm = np.array([1,0,0,-1,1,0,0,-1])  # normailized state energies

m_list_g = np.arange(m_num_g)*m_d_g # for ploting axis 
m_list_h = np.arange(m_num_h)+m_inc_h
m_colorbar = ['k','b','c','g','y','r','m','violet']

# computing NESS distributions
for i in range(0,m_num_g):
    for j in range(0,m_num_h):
        m_g = i*m_d_g # \gamme
        m_h = j+m_inc_h # \h_0
        m_mat_k = sts.cal_mat_k(m_x1,m_x2,m_g,m_h)
        m_ness = sts.cal_ness_distribution(m_mat_k)
        # (m_mat_J,m_Jx,m_Jy) = cal_J_flux(m_ness,m_mat_k)
        filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
        with open(filename,'w') as f:
            np.savetxt(f,m_ness)

# computing properties
if M_Calc_Prop==1:
    for i in range(0,m_num_g):
        for j in range(0,m_num_h):
            m_g = i*m_d_g
            m_h = j+m_inc_h
            m_mat_k = sts.cal_mat_k(m_x1,m_x2,m_g,m_h)
            filename = f".\\resl_nessdist\\xo_{m_x1}xt_{m_x2}h_{m_h}g_{m_g}.dat"
            with open(filename,'r') as f:
                m_dist_rd = np.loadtxt(f,delimiter='\n') # distribution readout
            # correctness
            crrct_exct = m_dist_rd[m_index_correct_state]
            crrct_gnrl = np.sum(m_dist_rd[m_index_correct_output])
            record_row = np.array([m_g,m_h,crrct_exct,crrct_gnrl])
            f_record = f".\\resl_ness_ana\\correctness.dat"
            with open(f_record,'ab') as f:
                np.savetxt(f,[record_row],fmt="%d\t%d\t%.9e\t%.9e")
            # mutual information flow
            (m_idot_x,m_idot_y) = sts.cal_idot(m_dist_rd,m_mat_k)
            record_row = np.array([m_g,m_h,m_idot_x,m_idot_y])
            f_record = f".\\resl_ness_ana\\idot_flux.dat"
            with open(f_record,'ab') as f:
                np.savetxt(f,[record_row],fmt="%d\t%d\t%.18e\t%.18e")
            # mutual information
            m_I_mi = sts.cal_I_mutlinfo(m_dist_rd)
            record_row = np.array([m_g,m_h,m_I_mi])
            f_record = f".\\resl_ness_ana\\mutual_information.dat"
            with open(f_record,'ab') as f:
                np.savetxt(f,[record_row],fmt="%d\t%d\t%.18e")
            # effective dynamics
            m_energy = m_h*m_energy_nm/2
            (delta_effc,omega) = sts.cal_intrin_rt(m_dist_rd,m_mat_k,m_energy)
            record_row = np.array([m_g,m_h,delta_effc,omega])
            f_record = f".\\resl_ness_ana\\effc_rt.dat"
            with open(f_record,'ab') as f:
                np.savetxt(f,[record_row],fmt="%d\t%d\t%.7e\t%.7e")
            # entropy production
            (m_heat_d,m_entp_mic,m_entp_tot) = sts.cal_entropy_prod_total(m_dist_rd,m_mat_k)
            record_row = np.array([m_g,m_h,m_heat_d,m_entp_mic,m_entp_tot])
            f_record = f".\\resl_ness_ana\\entropy_prod.dat"
            with open(f_record,'ab') as f:
                np.savetxt(f,[record_row],fmt="%d\t%d\t%.7e\t%.7e\t%.7e")
            #

## Make plots

# reading results
filename_crrct = f".\\resl_ness_ana\\correctness.dat"
with open(filename_crrct,'r') as f:
    m_rd_crrct = np.loadtxt(f)
filename_idot = f".\\resl_ness_ana\\idot_flux.dat"
with open(filename_idot,'r') as f:
    m_rd_idot = np.loadtxt(f)
filename_effc = f".\\resl_ness_ana\\effc_rt.dat"
with open(filename_effc,'r') as f:
    m_rd_effc = np.loadtxt(f)
filename_mtif = f".\\resl_ness_ana\\mutual_information.dat"
with open(filename_mtif) as f:
    m_rd_mi = np.loadtxt(f)
filename_entr = f".\\resl_ness_ana\\entropy_prod.dat"
with open(filename_entr) as f:
    m_rd_et = np.loadtxt(f)

m_crrct_exct = np.reshape(m_rd_crrct[:,2],(m_num_g,m_num_h)) # exact correctness probability
m_crrct_gnrl = np.reshape(m_rd_crrct[:,3],(m_num_g,m_num_h)) # general correctness probability
m_idot_x = np.reshape(m_rd_idot[:,2],(m_num_g,m_num_h)) # mutual information flow in X domain
m_idot_y = np.reshape(m_rd_idot[:,3],(m_num_g,m_num_h)) # mutual information flow in Y domain
m_effc_d = np.reshape(m_rd_effc[:,2],(m_num_g,m_num_h)) # effective external energy driving
m_effc_o = np.reshape(m_rd_effc[:,3],(m_num_g,m_num_h)) # effective intrinsic jumping rate
m_I_mi = np.reshape(m_rd_mi[:,2],(m_num_g,m_num_h)) # effective mutual information I
m_heat_d = np.reshape(m_rd_et[:,2],(m_num_g,m_num_h)) # heat dissipation rate
m_entp_mic = np.reshape(m_rd_et[:,3],(m_num_g,m_num_h)) # micro entropy production rate
m_entp_tot = np.reshape(m_rd_et[:,4],(m_num_g,m_num_h)) # total entropy production rate

# ploting curves according to h_0
if M_Plot_H_Haxis==1:
    # ploting correctness probability
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_crrct_exct[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_crrct_gnrl[i,:],'--',color=m_colorbar[i])
    plt.xlabel("h_0 value")
    plt.ylabel("correctness probability")
    plt.savefig('.\\resl_ness_ana\\correctness_h.png')
    # ploting mutual information
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_I_mi[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Mutual information")
    plt.savefig('.\\resl_ness_ana\\I_mutl_info_h.png')
    # ploting mutual information in X domain
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_idot_x[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Mutual information flux in X1X2 domain")
    plt.savefig('.\\resl_ness_ana\\Idot_X_h.png')
    # ploting mutual information in Y domain
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_idot_y[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Mutual information flux in Y domain")
    plt.savefig('.\\resl_ness_ana\\Idot_Y_h.png')
    # ploting effective driving
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_effc_d[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Effective \u03B4")
    plt.title(" Effective external driving after coarse graining\n(positive direction y0 to y1) ")
    plt.savefig('.\\resl_ness_ana\\Effc_driving_h.png')
    # ploting effective intrinsic jumping rate
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_effc_o[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Effective \u03C9")
    plt.title(" Effective intrinsic jumping rate after coarse graining")
    plt.savefig('.\\resl_ness_ana\\Effc_intrin_rate_h.png')
    # ploting heat dissipation rate
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_heat_d[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Heat dissipation rate")
    plt.savefig('.\\resl_ness_ana\\Heat_dissipation_h.png')
    # ploting micro entropy production rate
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_entp_mic[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Microscopic entropy production rate")
    plt.savefig('.\\resl_ness_ana\\Micro_entropy_prod_h.png')
    # ploting total entropy production rate
    plt.figure()
    for i in range(m_num_g):
        plt.plot(m_list_h,m_entp_tot[i,:],label='\u03B3=%.1f'%m_list_g[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("h_0 value")
    plt.ylabel("Total entropy production rate")
    plt.savefig('.\\resl_ness_ana\\Total_entropy_prod_h.png')

# ploting curves according to \gamma
if M_Plot_G_Haxis==1:
    # ploting correctness probability
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_crrct_exct[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_crrct_gnrl[:,i],'--',color=m_colorbar[i])
    plt.xlabel("\u03B3 value")
    plt.ylabel("correctness probability")
    plt.savefig('.\\resl_ness_ana\\correctness_g.png')
    # ploting mutual information
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_I_mi[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Mutual information")
    plt.savefig('.\\resl_ness_ana\\I_mutl_info_g.png')
    # ploting mutual information in X domain
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_idot_x[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Mutual information flux in X1X2 domain")
    plt.savefig('.\\resl_ness_ana\\Idot_X_g.png')
    # ploting mutual information in Y domain
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_idot_y[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Mutual information flux in Y domain")
    plt.savefig('.\\resl_ness_ana\\Idot_Y_g.png')
    # ploting effective driving
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_effc_d[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Effective \u03B4")
    plt.title(" Effective external driving after coarse graining\n(positive direction y0 to y1) ")
    plt.savefig('.\\resl_ness_ana\\Effc_driving_g.png')
    # ploting effective intrinsic jumping rate
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_effc_o[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Effective \u03C9")
    plt.title(" Effective intrinsic jumping rate after coarse graining")
    plt.savefig('.\\resl_ness_ana\\Effc_intrin_rate_g.png')
    # ploting heat dissipation rate
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_heat_d[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Heat dissipation rate")
    plt.savefig('.\\resl_ness_ana\\Heat_dissipation.png')
    # ploting micro entropy production rate
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_entp_mic[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Microscopic entropy production rate")
    plt.savefig('.\\resl_ness_ana\\Micro_entropy_prod.png')
    # ploting total entropy production rate
    plt.figure()
    for i in range(m_num_h):
        plt.plot(m_list_g,m_entp_tot[:,i],label='h_0=%.1f'%m_list_h[i],color=m_colorbar[i])
        plt.legend()
    plt.xlabel("\u03B3 value")
    plt.ylabel("Total entropy production rate")
    plt.savefig('.\\resl_ness_ana\\Total_entropy_prod.png')

# ploting contours
if M_Plot_Contour==1:
    m_Ygrid = np.arange(m_num_g)*m_d_g # for ploting axis
    m_Xgrid = np.arange(m_num_h)+m_inc_h
    # ploting contour for correctness
    plt.figure()
    plt.contourf(m_Xgrid,m_Ygrid,m_crrct_exct,levels=np.arange(0,1,0.01))
    plt.colorbar()
    contour=plt.contour(m_Xgrid,m_Ygrid,m_crrct_exct,colors='white',levels=np.arange(0,1,0.5))
    plt.clabel(contour,inline=True,fontsize=8)
    plt.xlabel("h_0")
    plt.ylabel("\u03B3")
    plt.title("Contour of correctness under tuning factors")
    plt.savefig('.\\resl_ness_ana\\correct_contour.png')
    # ploting contour for heat dissipation
    plt.figure()
    plt.contourf(m_Xgrid,m_Ygrid,m_heat_d,levels=np.arange(0,16,0.01))
    plt.colorbar()
    contour=plt.contour(m_Xgrid,m_Ygrid,m_heat_d,colors='white',levels=np.arange(1,16,2))
    plt.clabel(contour,inline=True,fontsize=8)
    plt.xlabel("h_0")
    plt.ylabel("\u03B3")
    plt.title("Contour of heat dissipation")
    plt.savefig('.\\resl_ness_ana\\heat_dsspt_contour.png')
    # ploting contour for effective driving
    plt.figure()
    plt.contourf(m_Xgrid,m_Ygrid,m_effc_d,levels=np.arange(-0.5,1.5,0.01))
    plt.colorbar()
    contour=plt.contour(m_Xgrid,m_Ygrid,m_effc_d,colors='white',levels=np.arange(-0.5,1.5,0.5))
    plt.clabel(contour,inline=True,fontsize=8)
    plt.xlabel("h_0")
    plt.ylabel("\u03B3")
    plt.title("Contour of effective driving")
    plt.savefig('.\\resl_ness_ana\\effect_driving_contour.png')
    # ploting contour for intrinsic jumping rate
    plt.figure()
    plt.contourf(m_Xgrid,m_Ygrid,m_effc_o,levels=np.arange(0,1.5,0.01))
    plt.colorbar()
    plt.xlabel("h_0")
    plt.ylabel("\u03B3")
    plt.title("Contour of effective intrinsic jumping rate")
    plt.savefig('.\\resl_ness_ana\\effect_omega_contour.png')
    # ploting contour for mutual information
    plt.figure()
    plt.contourf(m_Xgrid,m_Ygrid,m_I_mi,levels=np.arange(0,0.7,0.01))
    plt.colorbar()
    contour=plt.contour(m_Xgrid,m_Ygrid,m_I_mi,colors='white',levels=np.arange(0,0.7,0.15))
    plt.clabel(contour,inline=True,fontsize=8)
    plt.xlabel("h_0")
    plt.ylabel("\u03B3")
    plt.title("Contour of mutual information")
    plt.savefig('.\\resl_ness_ana\\I_contour.png')