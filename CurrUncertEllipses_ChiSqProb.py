
from netCDF4 import Dataset,num2date



import pdb
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime#,timedelta

#import xarray

#lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
#lat = np.arange(40.066669,65+1/15.,1/15.)

#import os

from scipy.stats import chi2



from CurrUncertEllipses import *

def main():

    curr_uncert_prob_threshold_perc_data_in_xsd_table()
    fig = curr_uncert_prob_threshold_perc_data_in_xsd_figure()


def curr_uncert_prob_threshold_perc_data_in_xsd_figure():

    # Calculate and plot the percentage of data within an uncertainty ellipse
    # of a given size (in terms of standard deviations).
    #
    # As well deriving these values for the chi-squared distribution table, two
    # numerical methods are used (See Tinker et al. (2022) for details).
    #
    # Produces Appendix Figure 7 in Tinker et al. 2022


    #Array of Standard deviations
    n_std_mat = np.arange(0,3.2,0.1)

    # precentage of data within ellipse of a given standard deviation size, using the:
    #Statistical theoretical method (using chi squared probabilty tables)
    stat_sd_plev_mat = data_within_xsd_chi_sq(n_std_mat = n_std_mat)
    #Gaussian distribution method (Integrating a bivariate Gaussian distribution within the ellipse)
    gauss_sd_plev_mat = data_within_xsd_gauss_integ(n_std_mat = n_std_mat)
    #Random data method (asking the proprotion of a random bivariate gaussian data set is within an ellipse).
    rand_sd_plev_mat = data_within_xsd_random_cnt(n_std_mat = n_std_mat,npnts = 10000)# 100 = 1min, 1000 = 1 min # 10000 = 2 mins


    print('Start plotting',datetime.now())

    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(6.0)
    plt.subplots_adjust(top=0.95,bottom=0.15,left=0.15,right=0.95,hspace=0.2,wspace=0.2)
    plt.plot([0,nstd_cutoff(90),nstd_cutoff(90)],[90,90,0],'0.75')
    plt.plot([0,nstd_cutoff(95),nstd_cutoff(95)],[95,95,0],'0.75')
    plt.text(0.1,95,'95%', ha = 'left', va = 'center')
    plt.text(0.1,90,'90%', ha = 'left', va = 'center')
    plt.text(nstd_cutoff(90),5,'%.2f'%nstd_cutoff(90), ha = 'center', va = 'center')
    plt.text(nstd_cutoff(95),5,'%.2f'%nstd_cutoff(95), ha = 'center', va = 'center')
    plt.plot(n_std_mat, 100.*rand_sd_plev_mat.mean(axis = 1),'r', lw = 2, label = 'Random')
    plt.plot(n_std_mat, 100.*rand_sd_plev_mat.mean(axis = 1) + 2*rand_sd_plev_mat.std(axis = 1),'r-', lw = 1)
    plt.plot(n_std_mat, 100.*rand_sd_plev_mat.mean(axis = 1) - 2*rand_sd_plev_mat.std(axis = 1),'r-', lw = 1)
    plt.plot(n_std_mat, 100.*gauss_sd_plev_mat.mean(axis = 1),'b', lw = 2, label = 'Distr Integ')
    plt.plot(n_std_mat, 100.*gauss_sd_plev_mat.mean(axis = 1) + 2*gauss_sd_plev_mat.std(axis = 1),'b-', lw = 1)
    plt.plot(n_std_mat, 100.*gauss_sd_plev_mat.mean(axis = 1) - 2*gauss_sd_plev_mat.std(axis = 1),'b-', lw = 1)
    plt.plot(n_std_mat, 100.*stat_sd_plev_mat,'k--', lw = 2, label = 'Chi Sq')
    plt.xlabel('Size of uncertainty ellipse\n(number of standard deviation)')
    plt.ylabel('% Data within uncertainty ellipse')
    plt.ylim([0,100])
    plt.xlim([0,3])
    plt.legend()


    print('Return handle',datetime.now())
    return fig

def curr_uncert_prob_threshold_perc_data_in_xsd_table():
    # Produce a table of probabilty thresholds for ellipse size.
    # Produces Appendix Table 1 in Tinker et al. 2022

    perc_lev_mat = np.array([50, 75,90, 95, 97.5, 99,99.5 ])
    p_lev = 1-(perc_lev_mat/100.)
    chi_sq_table_vals_mat = nstd_cutoff(perc_lev_mat)**2
    nstd_thresh_size_mat = nstd_cutoff(perc_lev_mat)

    print('')
    print('------------------------------------------------------------------------------------')
    print('')
    print('Uncertainty Ellipse size (in standard deviations) and data coverage (%),,Chi Squared Distribution Table (with 2 degrees of freedom),,')
    print('Percentage of data within Uncertainty Ellipse,Size of uncertainty ellipse (# standard deviations),Critical value,Probability of exceeding the critical value')
    for ii,jj,kk,ll in zip(perc_lev_mat,nstd_thresh_size_mat,p_lev,chi_sq_table_vals_mat,):print('%.1f%%,%.4f,%.3f,%.3f'%(ii,jj,kk,ll))
    print('')
    print('------------------------------------------------------------------------------------')
    print('')


def nstd_cutoff(percent_val):
    #For a given percentage value, how big (in standard deviations)
    #must the ellipse be to capture that precengate of data
    #
    # Based on the Chi-squared inverse survival function

    nstd = np.sqrt(chi2.isf(1-percent_val/100, 2))

    return nstd

def data_within_xsd_chi_sq(n_std_mat = np.arange(0,3,0.1)):
    # following:
    #https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix
    # To calculate the amount of data within an ellipse of size x std devs,
    # we can use the chi squared probabilty table.

    #chi squared probability table:
    #https://people.richland.edu/james/lecture/m170/tbl-chi.html
    #c&p 2df row and headers:
    chi2_prob_2df = np.array([0.010,0.020,0.051,0.103,0.211,4.605,5.991,7.378,9.210,10.597])
    chi2_prob_plev = np.array([0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005])

    # this can be created with python scipy.stats chi2:
    # https://stackoverflow.com/questions/32301698/how-to-build-a-chi-square-distribution-table
    chi_sq_prob_2df_table = chi2.isf(chi2_prob_plev, 2)

    # plotting 1-chi2_prob_plev against np.sqrt(chi2_prob_plev) gives you the
    # required number of std devs (sqrt(chi2_prob_plev)) to encapsulate x % of
    # data (1-chi2_prob_plev).

    # for a given array of standard deviations, we can use this approach to
    # calculate the percentage data within the corresponding ellipse.
    # rather than using the inverse survival function, we now use the
    # survival function


    chi2_pval_mat = 1-chi2.sf(n_std_mat**2, 2)


    return chi2_pval_mat  #, chi2_prob_plev, chi_sq_prob_2df_table


def data_within_xsd_gauss_integ_val(U_mean = 0.,U_var = 1.,V_mean = 0.,V_var = 1.,UV_corr = 0.5, n_std = 1.96, plotting = False, verbose = True, npnt_counting = 151, n_std_limits = 2):

    # To calculate the amount of data within an ellipse of size x std devs,
    # we can integrate a bivariate gaussian distribution surface within the ellipse.
    # We do this numerically, so this is a semi-numerical semi-analytical method.
    #
    # We created a decretised bivariate gaussian distribution surface
    # (for a given means, variaences and covariance (actually correlation).
    # We find the (near constant) value of the surface around the ellipse, and
    # then (numerically) integrate the values of the surface that are greater
    # than this value.

    #Covariance from Pearsons Correlation.
    UV_cov = UV_corr*np.sqrt(U_var)*np.sqrt(V_var)

    #details of the ellipse
    X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_stats_parametric_equation(U_mean,V_mean,U_var, V_var, UV_cov)

    twoaltone = np.array(([-1,1]))
    ang = np.linspace(-np.pi,np.pi, 100)


    #limits of the Gaussian surface
    Xlim_val = n_std_limits*n_std*(X_elip_amp)
    Ylim_val = n_std_limits*n_std*(Y_elip_amp)
    if Xlim_val <(4*(X_elip_amp)):Xlim_val = (4*(X_elip_amp))
    if Ylim_val <(4*(Y_elip_amp)):Ylim_val = (4*(Y_elip_amp))
    Xlim = Xlim_val*twoaltone+U_mean
    Ylim = Ylim_val*twoaltone+V_mean

    # x and y mesh for the surface
    tmpx_test = np.linspace(np.min((Xlim)),np.max((Xlim)),npnt_counting)
    tmpy_test = np.linspace(np.min((Ylim)),np.max((Ylim)),npnt_counting)
    tmpx_test_mat,tmpy_test_mat = np.meshgrid(tmpx_test,tmpy_test)
    tmpdx = np.diff(tmpx_test).mean()
    tmpdy = np.diff(tmpy_test).mean()

    # the uncertainty ellipse
    Xo = n_std*(X_elip_amp*np.sin(ang + X_elip_phi))+U_mean
    Yo = n_std*(Y_elip_amp*np.sin(ang + Y_elip_phi))+V_mean

    #Calcuate the Gaussian Surface over the x and y mesh, and around the ellipse
    gauss = gauss_func_2d(tmpx_test_mat,tmpy_test_mat,U_mean,V_mean,U_var,V_var,UV_cov)[0]
    gauss_ell = gauss_func_2d(Xo,Yo,U_mean,V_mean,U_var,V_var,UV_cov)[0]

    # find the values that distribution values that are greater than the (mean)
    # ellipse distribution value
    ind_inside_ell = gauss>=gauss_ell.mean()

    # The infinite bivariate distrbution surface should integrate to 1.
    # By integrating the full decretised distribution, we get an idea of the
    # error term
    p_val_full_decrete_dist = gauss.sum()*tmpdx*tmpdy

    # Integrating the values greater than the ellipse values is equivalent to
    # integrating the values within the ellipse.

    p_val = gauss[ind_inside_ell].sum()*tmpdx*tmpdy

    if plotting:
        ax = []
        ax.append(plt.subplot(2,2,1))
        plt.pcolormesh(tmpx_test_mat,tmpy_test_mat,gauss)
        plt.contour(tmpx_test_mat,tmpy_test_mat,gauss, [gauss_ell.mean()], colors = 'y')
        plt.plot(Xo,Yo,'r--')
        ax.append(plt.subplot(2,2,2))
        plt.pcolormesh(tmpx_test_mat,tmpy_test_mat,ind_inside_ell)
        plt.contour(tmpx_test_mat,tmpy_test_mat,gauss, [gauss_ell.mean()], colors = 'y')
        plt.plot(Xo,Yo,'r--')
        ax.append(plt.subplot(2,2,3))
        plt.pcolormesh(tmpx_test_mat,tmpy_test_mat,)
        plt.contour(tmpx_test_mat,tmpy_test_mat,gauss, [gauss_ell.mean()], colors = 'y')
        plt.plot(Xo,Yo,'r--')


    if verbose: print(n_std, p_val)
    return p_val, p_val_full_decrete_dist
    #plt.show()



def data_within_xsd_random_cnt_val(U_mean = 0,U_var = 1,V_mean = 0,V_var = 1,UV_corr=0., npnts = 100000,n_std_mat = np.arange(0,3,0.01)):

    # To calculate the amount of data within an ellipse of size x std devs,
    # we can create a random data with a bivariate normal distribution for a
    # given set of means, variance and covariance (actually correlation).
    # We can then fit an ellipse to these data (for a given number of standard
    # deviations), and calucate the precentage of points within the ellipse.
    # We then cycle through a range of standard deviations (n_std_mat)


    #Covariance from Pearsons Correlation.
    UV_cov = UV_corr*np.sqrt(U_var)*np.sqrt(V_var)

    #Create a random data with a bivariate normal distribution
    U_mat,V_mat  = np.random.multivariate_normal([U_mean,V_mean], [[U_var,UV_cov],[UV_cov,V_var]], npnts).T



    #cycle through a range of elipses sizes of varying standard deviations
    n_perc_joint_mat = n_std_mat.copy()*0.
    for ni,n_std in enumerate(n_std_mat):

        #for a given standard deviation:
        #find the uncertainty ellipse, a details of it:

        X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_mat_parametric_equation(U_mat.reshape(-1,1,1),V_mat.reshape(-1,1,1), n_std = n_std)
        qmax,qmin, ecc, theta_max, zero_ang = ellipse_parameters_from_parametric_equation(X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,U_mean,V_mean)

        # find the ellipse foci (important for asking whether a point is within an ellipse or not)
        foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2 = find_parameteric_ellipse_foci(qmax, qmin,theta_max,U_mean,V_mean,n_std)


        # Ask which of our random data set are within the ellipse
        pnt_inside_ell_sig_1,foci_pnt_foci_dist_sig = point_inside_parameteric_ellipse(U_mat.reshape(-1,1,1),V_mat.reshape(-1,1,1),n_std,  foci_x_1,foci_y_1,foci_x_2,foci_y_2,qmax)

        # Record the percentage of data within our ellipse.
        n_perc_joint_mat[ni] = pnt_inside_ell_sig_1.sum()/pnt_inside_ell_sig_1.size

    # Repeat of a univariate normal discribution.
    # ask which points are within x standard deviation of the mean
    n_perc_single_mat = n_std_mat.copy()*0.
    U_std = U_mat.std()
    for ni,n_std in enumerate(n_std_mat):n_perc_single_mat[ni] = (np.abs((U_mat[:]-U_mean)/U_std)<=n_std).sum()/U_mat.size#((np.abs(U_mat)/U_std)<n_std).sum()/U_mat.size


    return n_perc_joint_mat, n_perc_single_mat


################################################################################


def data_within_xsd_gauss_integ(n_std_mat = np.arange(0,3,0.1), U_mean_mat = np.arange(-1.5,1.,0.5), V_mean_mat = np.arange(-1.5,2,0.5), U_var_mat = np.arange(0.25,1.5,0.25), V_var_mat = np.arange(0.25,1.5,0.25), UV_corr_mat = np.arange(-0.75,1.00,0.25)):


    # To calculate the amount of data within an ellipse of size x std devs,
    # we can integrate a bivariate gaussian distribution surface within the ellipse.
    #
    # Here we cycle through a range of values mean, variance and covarinace
    # (actually correlations) and apply data_within_xsd_gauss_integ_val to
    # create an ensemble of results, to show that there is very little dependence
    # on the shape and location of the ellipse.


    uv_ms_c_lst = [(U_mean,U_var,V_mean,V_var,UV_corr) for U_mean in U_mean_mat for V_mean in V_mean_mat for U_var in U_var_mat for V_var in V_var_mat for UV_corr in UV_corr_mat]
    uv_ms_c_mat =np.array(uv_ms_c_lst)


    print('Start Gaussian method',datetime.now()) # 2min run time
    gauss_sd_plev_lst = []
    for (U_mean,U_var,V_mean,V_var,UV_corr) in uv_ms_c_lst[:]:
        gauss_sd_plev_lst_curr_it = []
        for n_std in n_std_mat:
            gauss_sd_plev_lst_curr_it.append(data_within_xsd_gauss_integ_val(U_mean = U_mean,U_var = U_var,V_mean = V_mean,V_var = V_var,UV_corr=UV_corr,n_std = n_std, plotting = False, verbose = False)[0])
        gauss_sd_plev_lst.append(gauss_sd_plev_lst_curr_it)
    gauss_sd_plev_mat = np.array(gauss_sd_plev_lst)
    print('Stop Gaussian method',datetime.now())

    return gauss_sd_plev_mat.T


def data_within_xsd_random_cnt(n_std_mat = np.arange(0,3,0.1),npnts = 1000, U_mean_mat = np.arange(-1.5,1.,0.5), V_mean_mat = np.arange(-1.5,2,0.5), U_var_mat = np.arange(0.25,1.5,0.25), V_var_mat = np.arange(0.25,1.5,0.25), UV_corr_mat = np.arange(-0.75,1.00,0.25)): # 1e4 = 10 mins, 1e3 = 2 mins

    # To calculate the amount of data within an ellipse of size x std devs,
    # we can create a random data with a bivariate normal distribution for a
    # given set of means, variance and covariance (actually correlation).
    #
    # Here we cycle through a range of values mean, variance and covarinace
    # (actually correlations) and apply data_within_xsd_gauss_integ_val to
    # create an ensemble of results, to show that there is very little dependence
    # on the shape and location of the ellipse.


    uv_ms_c_lst = [(U_mean,U_var,V_mean,V_var,UV_corr) for U_mean in U_mean_mat for V_mean in V_mean_mat for U_var in U_var_mat for V_var in V_var_mat for UV_corr in UV_corr_mat]
    uv_ms_c_mat =np.array(uv_ms_c_lst)

    print('Start random method',datetime.now()) # 2min run time
    rand_sd_plev_lst = []
    for (U_mean,U_var,V_mean,V_var,UV_corr) in uv_ms_c_lst[:]:
        rand_sd_plev_lst.append(data_within_xsd_random_cnt_val(U_mean = U_mean,U_var = U_var,V_mean = V_mean,V_var = V_var,UV_corr=UV_corr,npnts = npnts,n_std_mat = n_std_mat)[0])
    rand_sd_plev_mat = np.array(rand_sd_plev_lst)
    print('Stop random method',datetime.now()) # 2min run time

    return rand_sd_plev_mat.T

    ###################################################################

if __name__ == "__main__":
    main()
