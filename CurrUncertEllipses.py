from netCDF4 import Dataset




import netCDF4
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sys

from datetime import datetime,timedelta

import xarray

import os





from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['contour.negative_linestyle'] = 'solid'
rcParams['pcolor.shading'] = 'auto'



def ellipse_init_proc(U_mat,V_mat):

    # initial processing of basic stats about the flow field.
    # Give two np arrays of U and V velocities.
    # U_mat, V_mat = (time, lat, lon)


    # Calculate the mean, magnitude, standard devaiation, variance co variance
    U_mean = U_mat.mean(axis = 0)
    V_mean = V_mat.mean(axis = 0)
    UV_mean = np.sqrt(U_mean**2 + V_mean**2)
    U_std = U_mat.std(axis = 0)
    V_std = V_mat.std(axis = 0)

    U_var = U_mat.var(axis = 0)
    V_var = V_mat.var(axis = 0)
    UV_cov = (( U_mat - U_mat.mean(axis = 0))*(V_mat - V_mat.mean(axis = 0))).mean(axis = 0)

    UV_mat = np.sqrt(U_mat**2 + V_mat**2)
    ang_xy = np.arctan2(V_mean, U_mean)

    return U_mean,V_mean,UV_mean,U_std,V_std,U_var,V_var,UV_cov,UV_mat,ang_xy

def ellipse_init_norm_test(U_mat,V_mat, min_time_samples = 8,alpha = 1e-3):


    # Normality Test

    nt = U_mat.shape[0]


    U_isnorm,V_isnorm = None,None
    if nt>min_time_samples:
        # test for normality
        ######################
        from scipy.stats import shapiro,normaltest
        U_k2, U_norm_p = normaltest(U_mat, axis = 0)
        V_k2, V_norm_p = normaltest(V_mat, axis = 0)

        U_isnorm = ((U_norm_p<alpha)==False)*1.
        V_isnorm = ((V_norm_p<alpha)==False)*1.
    else:
        print ('You need more than %i time samples test normality. Currently %i time samples'%(min_time_samples,nt))
        print ('Consider changing keyword min_time_samples')
        print ('Returning None,None')


    return U_isnorm,V_isnorm

def ellipse_params(U_mat,V_mat, n_std = 2.45,pnt_x = 0, pnt_y = 0):

    U_mean,V_mean,UV_mean,U_std,V_std,U_var,V_var,UV_cov,UV_mat,ang_xy = ellipse_init_proc(U_mat,V_mat)

    X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_mat_parametric_equation(U_mat,V_mat, n_std = n_std)
    qmax,qmin, ecc, theta_max, zero_ang = ellipse_parameters_from_parametric_equation(X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,U_mean,V_mean)
    XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir = find_num_std_to_point(U_mean,V_mean,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi)
    y_tang_1,y_tang_2,ang_wid = find_tangent_to_parametric_ellipse_at_a_point(U_mean,V_mean,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,pnt_x = pnt_x, pnt_y = pnt_y, n_std = n_std)
    foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2 = find_parameteric_ellipse_foci(qmax, qmin,theta_max,U_mean,V_mean, n_std = n_std)


    return U_mean,V_mean,UV_mean,U_std,V_std,U_var,V_var,UV_cov,UV_mat,ang_xy,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos,qmax,qmin, ecc, theta_max, zero_ang,XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir,y_tang_1,y_tang_2,ang_wid,foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2

def ellipse_params_add_to_dict(input_tuple):


    U_mean,V_mean,UV_mean,U_std,V_std,U_var,V_var,UV_cov,UV_mat,ang_xy,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos,qmax,qmin, ecc, theta_max, zero_ang,XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir,y_tang_1,y_tang_2,ang_wid,foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2 = input_tuple

    dict_out = {}

    dict_out['U_mean'] = U_mean
    dict_out['V_mean'] = V_mean
    dict_out['UV_mean'] = UV_mean
    dict_out['U_std'] = U_std
    dict_out['V_std'] = V_std
    dict_out['U_var'] = U_var
    dict_out['V_var'] = V_var
    dict_out['UV_cov'] = UV_cov
    dict_out['UV_mat'] = UV_mat
    dict_out['ang_xy'] = ang_xy
    dict_out['X_elip_amp'] = X_elip_amp
    dict_out['Y_elip_amp'] = Y_elip_amp
    dict_out['X_elip_phi'] = X_elip_phi
    dict_out['Y_elip_phi'] = Y_elip_phi
    dict_out['X_elip_phi_cos'] = X_elip_phi_cos
    dict_out['Y_elip_phi_cos'] = Y_elip_phi_cos
    dict_out['qmax'] = qmax
    dict_out['qmin'] = qmin
    dict_out['ecc'] =  ecc
    dict_out['theta_max'] =  theta_max
    dict_out['zero_ang'] =  zero_ang
    dict_out['XY_std_dir_corr'] = XY_std_dir_corr
    dict_out['XY_zero_num_std_from_mean'] = XY_zero_num_std_from_mean
    dict_out['pX_dir'] = pX_dir
    dict_out['pY_dir'] = pY_dir
    dict_out['y_tang_1'] = y_tang_1
    dict_out['y_tang_2'] = y_tang_2
    dict_out['ang_wid'] = ang_wid
    dict_out['foci_max'] = foci_max
    dict_out['foci_x_1'] = foci_x_1
    dict_out['foci_y_1'] = foci_y_1
    dict_out['foci_x_2'] = foci_x_2
    dict_out['foci_y_2'] = foci_y_2

    return dict_out



def confidence_ellipse_uv_stats_parametric_equation(mean_x,mean_y,var_x, var_y, cov_xy, n_std = 2.45,show_working = False,show_working_ii = 0,show_working_jj = 0):

    ang = np.pi*np.arange(-180.,181.)/180. # ang = np.linspace(-np.pi,np.pi,360*1000)
    rotang = -np.pi/4
    mag_xy = np.sqrt(mean_x**2 + mean_y**2 )
    ang_xy = np.arctan2(mean_y, mean_x )

    #are U and V correlated
    pearson_xy = cov_xy/np.sqrt(var_x * var_y)

    ell_radius_x = np.sqrt(1 + pearson_xy)
    ell_radius_y = np.sqrt(1 - pearson_xy)

    scale_x = np.sqrt(var_x) * n_std
    scale_y = np.sqrt(var_y) * n_std





    if show_working == True:
        ii,jj = 120,120
        ii,jj =  176,150
        ii,jj =  show_working_ii,show_working_jj

        ang_ind = np.abs(ang_xy[jj,ii] - ang).argmin()


        Xs = ell_radius_x[jj,ii] * np.cos(ang)
        Ys = ell_radius_y[jj,ii] * np.sin(ang)


        Xsr = Xs*np.cos(rotang) + Ys*np.sin(rotang)
        Ysr = -Xs*np.sin(rotang) + Ys*np.cos(rotang)

        Xsr2 = (np.sqrt(0.5)*(Xs-Ys))
        Ysr2 = (np.sqrt(0.5)*(Xs+Ys))

        Xsrs = Xsr*scale_x[jj,ii]
        Ysrs = Ysr*scale_y[jj,ii]

        Xsrs2 =  np.sqrt(var_x[jj,ii])*(np.sqrt(0.5)*(Xs-Ys))
        Ysrs2 =  np.sqrt(var_y[jj,ii])*(np.sqrt(0.5)*(Xs+Ys))
        Xs_minus_Ys = Xs-Ys
        Xs_plus_Ys = Xs+Ys
        Xsrs2 =  np.sqrt(var_x[jj,ii])*(np.sqrt(0.5)*(Xs_minus_Ys))
        Ysrs2 =  np.sqrt(var_y[jj,ii])*(np.sqrt(0.5)*(Xs_plus_Ys))

        # https://www.myphysicslab.com/springs/trig-identity-en.html



        X_amp = np.sqrt(  (  np.sqrt(0.5)*ell_radius_x  )**2 + (-np.sqrt(0.5)*ell_radius_y)**2)
        Y_amp = np.sqrt(  (  np.sqrt(0.5)*ell_radius_x)**2 + (np.sqrt(0.5)*ell_radius_y)**2)

        X_phi = np.arctan2((np.sqrt(0.5)*ell_radius_x),(-np.sqrt(0.5)*ell_radius_y))
        Y_phi = np.arctan2((np.sqrt(0.5)*ell_radius_x),(np.sqrt(0.5)*ell_radius_y))

        '''
        plt.plot(Ysr)
        plt.plot(Y_amp[jj,ii]*np.sin(ang +Y_phi[jj,ii]),':')
        #plt.plot(Y_amp[jj,ii]*np.sin(ang +np.arctan2(ell_radius_y,ell_radius_x)[jj,ii]))
        plt.show()

        plt.plot(Xsr)
        plt.plot(X_amp[jj,ii]*np.sin(ang +X_phi[jj,ii]),':')
        plt.show()
        '''

        Xsrs3 =  np.sqrt(var_x[jj,ii])*(X_amp[jj,ii]*np.sin(ang +X_phi[jj,ii]))
        Ysrs3 =  np.sqrt(var_y[jj,ii])*(Y_amp[jj,ii]*np.sin(ang +Y_phi[jj,ii]))


        Xsrst = Xsrs + mean_x[jj,ii]
        Ysrst = Ysrs + mean_y[jj,ii]



        Xsrst2 =  np.sqrt(var_x[jj,ii])*(X_amp[jj,ii]*np.sin(ang +X_phi[jj,ii])) + mean_x[jj,ii]
        Ysrst2 =  np.sqrt(var_y[jj,ii])*(Y_amp[jj,ii]*np.sin(ang +Y_phi[jj,ii])) + mean_y[jj,ii]



    # Ellipse is in the form of x_ellipse = a sin (x + phi), y_ellipse = a sin (y + phi)
    # What is the amplitude and phase of the waves of the ellipse:
    X_elip_amp = np.sqrt(var_x)*np.sqrt((np.sqrt(0.5)*ell_radius_x)**2 + (-np.sqrt(0.5)*ell_radius_y)**2)
    Y_elip_amp = np.sqrt(var_y)*np.sqrt((np.sqrt(0.5)*ell_radius_x)**2 + (np.sqrt(0.5)*ell_radius_y)**2)

    X_elip_phi = np.arctan2((np.sqrt(0.5)*ell_radius_x),(-np.sqrt(0.5)*ell_radius_y))
    Y_elip_phi = np.arctan2((np.sqrt(0.5)*ell_radius_x),(np.sqrt(0.5)*ell_radius_y))

    # what is the phase if the equation is = a cos (x + phi), y_ellipse = a cos (y + phi)
    X_elip_phi_cos = 2*np.pi - X_elip_phi   # X_elip_phi = 2*np.pi - X_elip_phi_cos
    Y_elip_phi_cos = 2*np.pi - Y_elip_phi   # Y_elip_phi = 2*np.pi - Y_elip_phi_cos


    return X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos


def confidence_ellipse_uv_mat_parametric_equation(U_mat,V_mat, n_std = 2.45,show_working = False,show_working_ii = 0,show_working_jj = 0):

    #Given an array of U and V velocities (dimensions: time, x, y), calculate the equation of the ellipse (the amplitude and phase of the  x and y sin wave)
    # such that

    #ang = np.pi*np.arange(-180.,181.)/180. # ang = np.linspace(-np.pi,np.pi,360*1000)
    #  x_ellipse = X_elip_amp sin (ang + X_elip_phi), y_ellipse = Y_elip_amp sin (ang + Y_elip_phi)
    #
    # also gives the phase if the equation given in terms of cosine
    #  x_ellipse = X_elip_amp cos (ang + X_elip_phi_cos), y_ellipse = Y_elip_amp cos (ang + Y_elip_phi_cos)


    #X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_mat_parametric_equation(U_mat,V_mat)

    ang = np.pi*np.arange(-180.,181.)/180. # ang = np.linspace(-np.pi,np.pi,360*1000)
    rotang = -np.pi/4




    #pdb.set_trace()
    mean_x = np.mean(U_mat,axis = 0)
    mean_y = np.mean(V_mat,axis = 0)
    var_x = np.var(U_mat,axis = 0)
    var_y = np.var(V_mat,axis = 0)
    cov_xy =  ((U_mat-mean_x)*(V_mat-mean_y)).mean(axis = 0)


    X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_stats_parametric_equation(mean_x,mean_y,var_x, var_y, cov_xy,show_working = show_working,show_working_ii = show_working_ii,show_working_jj = show_working_jj, n_std = n_std)


    return X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos

def ellipse_parameters_from_parametric_equation(X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,mean_x,mean_y):

    # qmax,qmin, ecc, theta_max, zero_ang = ellipse_parameters_from_parametric_equation(X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos,mean_x,mean_y)

    # what is are the semi major and semi minor axes, the Eccentricity

    # taken from Pugh A3:4a - A3:5n
    # alpha_2 = np.sqrt(X_elip_amp**4 + Y_elip_amp**4 + (2*X_elip_amp**2 * Y_elip_amp**2)*np.cos(2*(X_elip_phi_cos-Y_elip_phi_cos)))
    # we can use -X_elip_phi+Y_elip_phi instead of X_elip_phi_cos-Y_elip_phi_cos
    alpha_2 = np.sqrt(X_elip_amp**4 + Y_elip_amp**4 + (2*X_elip_amp**2 * Y_elip_amp**2)*np.cos(2*(-X_elip_phi+Y_elip_phi)))

    qmax = np.sqrt((X_elip_amp**2 + Y_elip_amp**2 + alpha_2)/2)
    qmin = np.sqrt((X_elip_amp**2 + Y_elip_amp**2 - alpha_2)/2)

    '''
    #proof of qmax and qmin

    plt.plot(X_elip_amp[jj,ii]*np.sin(ang +X_phi[jj,ii]),Y_elip_amp[jj,ii]*np.sin(ang +Y_phi[jj,ii]))
    plt.axvline(c='grey', lw=1)
    plt.axhline(c='grey', lw=1)
    plt.plot(qmax[jj,ii]*np.cos(ang),qmax[jj,ii]*np.sin(ang))
    plt.plot(qmin[jj,ii]*np.cos(ang),qmin[jj,ii]*np.sin(ang))
    plt.axis('equal')
    plt.show()
    '''


    ecc = (qmax - qmin)/(qmax + qmin)

    # what is are the angle of the semi major axes, and the angle to the origin

    '''
    # originally using X_elip_phi_cos-Y_elip_phi_cos, now replaced with -X_elip_phi+Y_elip_phi
    delt_num = (Y_elip_amp**2)*np.sin(2*(X_elip_phi_cos-Y_elip_phi_cos))
    delt_dem = (X_elip_amp**2) + (Y_elip_amp**2)*np.cos(2*(X_elip_phi_cos-Y_elip_phi_cos))
    delta = np.arctan2(delt_num,delt_dem)/2.

    theta_max_num = Y_elip_amp*np.cos((X_elip_phi_cos-Y_elip_phi_cos) - delta)
    theta_max_den = X_elip_amp*np.cos(delta)


    theta_max = np.arctan2(theta_max_num,theta_max_den)

    '''

    delt_num = (Y_elip_amp**2)*np.sin(2*(-X_elip_phi+Y_elip_phi))
    delt_dem = (X_elip_amp**2) + (Y_elip_amp**2)*np.cos(2*(-X_elip_phi+Y_elip_phi))
    delta = np.arctan2(delt_num,delt_dem)/2.

    theta_max_num = Y_elip_amp*np.cos((-X_elip_phi+Y_elip_phi) - delta)
    theta_max_den = X_elip_amp*np.cos(delta)


    #
    theta_max = np.arctan2(theta_max_num,theta_max_den)




    X_zero_ang =  X_elip_amp*np.sin(X_elip_phi) + mean_x
    Y_zero_ang =  Y_elip_amp*np.sin(Y_elip_phi) + mean_y
    zero_ang = np.arctan2((Y_zero_ang - mean_y),(X_zero_ang - mean_x))



    return qmax,qmin, ecc, theta_max, zero_ang

def find_num_std_to_point(mean_x,mean_y,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,x_pnt = None,y_pnt = None):
    # XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir = find_num_std_to_point(mean_x,mean_y,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi)

    nj, ni = mean_x.shape

    if x_pnt is None:  x_pnt = mean_x.copy()*0.
    if y_pnt is None:  y_pnt = mean_y.copy()*0.


    ang = np.pi*np.arange(-180.,181.)/180. # ang = np.linspace(-np.pi,np.pi,360*1000)
    rotang = -np.pi/4


    mag_xy = np.sqrt((mean_x - x_pnt)**2 + (mean_y - y_pnt)**2 )
    ang_xy = np.arctan2((mean_y - y_pnt), (mean_x - x_pnt))


    # How many standard deviation is the origin from the mean.

    #First, what is the ellipse "radius" in the direction of the origin.
    #   BUT need to convert the angles used in the ellipse equation to real angles and vice versa


        # reaarrange with wolfram alpha:
        # https://www.wolframalpha.com/widgets/view.jsp?id=b833b9ec052dba3d28359e9de6532441
        ##  Equation:  (( a*sin(x) + b*cos(x) )/(  c*sin(x) + d*cos(x)   ) )= y
        ##  Make this the subject: x
        #x~~2 tan^(-1)(sqrt(a^2 - 2 a c y + b^2 - 2 b d y + c^2 y^2 + d^2 y^2)/(b - d y) + a/(b - d y) - (c y)/(b - d y)) + 6.2832 n and b - d y!=0 and 2 a^4 d - 2 a^3 b c - 6 a^3 c d y - 2 a^2 b c (a^2



    (y) = np.tan(ang_xy)
    (a) = Y_elip_amp*np.cos(Y_elip_phi)
    (b) = Y_elip_amp*np.sin(Y_elip_phi)
    (c) = X_elip_amp*np.cos(X_elip_phi)
    (d) = X_elip_amp*np.sin(X_elip_phi)

    (n) = 0

    tmp_ang_around_ellipse_mat = 2*np.arctan(  ( np.sqrt((a**2) - (2*a*c*y) + (b**2) - (2*b*d*y) + ((c**2)*(y**2)) + ((d**2)*(y**2)))/(b - (d*y))) + (a/(b - (d*y)))      -     ((c*y)/(b - (d*y)))) + (2*np.pi * n)


    #correct for angle quadrant
    # tmp_ang_around_ellipse_mat = tmp_ang_around_ellipse_mat + (ang_xy/(np.pi/2)).astype('int')


    '''
    plt.subplot(2,2,1)
    plt.pcolormesh((tmp_ang_around_ellipse_mat - tmp_ang_around_ellipse)/np.pi,vmin = -2.05, vmax = 2.05)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.pcolormesh((ang_xy/(np.pi/2)).astype('int'),vmin = -2.05, vmax = 2.05)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.pcolormesh(np.floor(ang_xy/(np.pi/2)),vmin = -2.05, vmax = 2.05)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.pcolormesh(np.round(ang_xy/(np.pi/2)),vmin = -2.05, vmax = 2.05)
    plt.colorbar()
    plt.show()

    '''



    calculate_ellipse_angle_to_origin_pointbypoint = False

    if calculate_ellipse_angle_to_origin_pointbypoint:

        print(datetime.now(), 'start loops')
        tmp_ang_around_ellipse= theta_max.copy()*0.
        for ii in range(ni):
            for jj in range(nj):

                #http://www.petercollingridge.co.uk/tutorials/computational-geometry/finding-angle-around-ellipse/
                angle_ell = np.arctan2(   Y_elip_amp[jj,ii]*np.sin(  ang + Y_elip_phi[jj,ii]  )     ,   X_elip_amp[jj,ii]*np.sin(  ang + X_elip_phi[jj,ii]  )   )
                tmpminloc = ((angle_ell - ang_xy[jj,ii]-np.pi)%(2*np.pi)).argmin()
                tmp_ang_around_ellipse[jj,ii] = ang[tmpminloc]


                check_equations = False
                if check_equations:
                    # let sin (A + B) = sin(A)cos(B) + cos(A)sin(B).
                    #   therefore U sin(x+phi) = alpha [ sin(x) + cos(x) ] and alpha = U cos(phi)
                    #  so y =  tan(ang) =  Ysin(x+phiy)/Xsin(x+phix) = (asinx + bcosx)/(csinx+dcosx)
                    (y) = np.tan(ang_xy[jj,ii])
                    (a) = Y_elip_amp[jj,ii]*np.cos(Y_elip_phi[jj,ii])
                    (b) = Y_elip_amp[jj,ii]*np.sin(Y_elip_phi[jj,ii])
                    (c) = X_elip_amp[jj,ii]*np.cos(X_elip_phi[jj,ii])
                    (d) = X_elip_amp[jj,ii]*np.sin(X_elip_phi[jj,ii])


                    (n) = 0  #number of 2 x pi added


                    x = ang

                    plt.plot(((Y_elip_amp[jj,ii]*np.sin(  ang + Y_elip_phi[jj,ii]  ) )   / ( X_elip_amp[jj,ii]*np.sin(  ang + X_elip_phi[jj,ii]  ))))
                    plt.plot(((a*np.sin(x) + b*np.cos(x))/(c*np.sin(x) + d*np.cos(x))))
                    plt.show()

                    plt.plot(np.arctan((Y_elip_amp[jj,ii]*np.sin(  ang + Y_elip_phi[jj,ii]  ) )   / ( X_elip_amp[jj,ii]*np.sin(  ang + X_elip_phi[jj,ii]  ))))
                    plt.plot(np.arctan((a*np.sin(x) + b*np.cos(x))/(c*np.sin(x) + d*np.cos(x))))
                    plt.show()


                    plt.plot(np.arctan2((Y_elip_amp[jj,ii]*np.sin(  ang + Y_elip_phi[jj,ii]  ) )   , ( X_elip_amp[jj,ii]*np.sin(  ang + X_elip_phi[jj,ii]  ))))
                    plt.plot(np.arctan2((a*np.sin(x) + b*np.cos(x)),(c*np.sin(x) + d*np.cos(x))))
                    plt.plot(angle_ell)
                    plt.show()


                    # reaarrange with wolfram alpha:
                    # https://www.wolframalpha.com/widgets/view.jsp?id=b833b9ec052dba3d28359e9de6532441
                    ##  Equation:  (( a*sin(x) + b*cos(x) )/(  c*sin(x) + d*cos(x)   ) )= y
                    ##  Make this the subject: x
                    #x~~2 tan^(-1)(sqrt(a^2 - 2 a c y + b^2 - 2 b d y + c^2 y^2 + d^2 y^2)/(b - d y) + a/(b - d y) - (c y)/(b - d y)) + 6.2832 n and b - d y!=0 and 2 a^4 d - 2 a^3 b c - 6 a^3 c d y - 2 a^2 b c (a^2
                    x = 2*np.arctan(  ( np.sqrt((a**2) - (2*a*c*y) + (b**2) - (2*b*d*y) + ((c**2)*(y**2)) + ((d**2)*(y**2)))/(b - (d*y))) + (a/(b - (d*y)))      -     ((c*y)/(b - (d*y)))) + 6.2832 * n


        print(datetime.now(), 'stop loops')

        pdb.set_trace()

        tmp_ang_around_ellipse_mat = tmp_ang_around_ellipse



    # what is the (x,y) point on the ellipse that is in line with the origin.
    #take the angle to the origin, converted to ellipse angles, and put into the ellipse equation.

    pX_dir = X_elip_amp * np.sin(tmp_ang_around_ellipse_mat + X_elip_phi ) + mean_x
    pY_dir = Y_elip_amp * np.sin(tmp_ang_around_ellipse_mat + Y_elip_phi ) + mean_y
    XY_std_dir_corr = np.sqrt((pX_dir - mean_x)**2 + (pY_dir- mean_y)**2)

    XY_zero_num_std_from_mean = mag_xy/XY_std_dir_corr

    return XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir


    '''
    X_dir = ell_radius_x * np.cos(tmp_ang_around_ellipse_mat)
    Y_dir = ell_radius_y * np.sin(tmp_ang_around_ellipse_mat)


    X_dir_rot = X_dir*np.cos(rotang) + Y_dir*np.sin(rotang)
    Y_dir_rot = -X_dir*np.sin(rotang) + Y_dir*np.cos(rotang)


    X_dir_rot_scl = X_dir_rot * np.sqrt(var_x)*n_std
    Y_dir_rot_scl = Y_dir_rot * np.sqrt(var_y)*n_std

    # the x, y and distance (magnitude) of the 1 std value in the direction of the mean current.
    X_dir_rot_scl_tran = X_dir_rot_scl + mean_x
    Y_dir_rot_scl_tran = Y_dir_rot_scl + mean_y



    XY_std_dir_corr = np.sqrt((X_dir_rot_scl_tran - mean_x)**2 + (Y_dir_rot_scl_tran- mean_y)**2)

    XY_zero_num_std_from_mean = mag_xy/XY_std_dir_corr

    '''

def find_tangent_to_parametric_ellipse_at_a_point(mean_x,mean_y,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,pnt_x = 0, pnt_y = 0,calc_method ='analytical', n_std = 2.45):
        # y_tang_1,y_tang_2,ang_wid = find_tangent_to_parametric_ellipse_at_a_point(mean_x,mean_y,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,pnt_x = 0, pnt_y = 0)

        # Find the angular width of the ellipse as seen from the origin.
        # Two methods:
        #   Numerical: cycle around the ellipse (~ every degree), and find the angle to the origin. then find the min and max value, (correcting for crossing -np.pi/np.pi)
        #   analytical: the tangent of the ellipse that crosses the origin give the width... use wolfram alfa to solve

        nj, ni = mean_x.shape

        # http://wiki.dtonline.org/index.php/Tangents_and_Normals

        # https://www.analyzemath.com/CircleEq/circle_intersection.html

        if calc_method == 'numerical':


            ang_max_pi = np.ma.zeros((nj,ni))-1e6
            ang_min_pi = np.ma.zeros((nj,ni))+1e6
            ang_max_2pi = np.ma.zeros((nj,ni))-1e6
            ang_min_2pi = np.ma.zeros((nj,ni))+1e6

            print(datetime.now(),'start ellipse tangent search')
            for di in range(360):
                if (di % 10) == 0: print(di, datetime.now())
                ri = np.pi*di/180
                #angi_pi = np.arctan2(Y_elip_amp*np.sin(ri +Y_elip_phi)+ mean_y,X_elip_amp*np.sin(ri +X_elip_phi) + mean_x).reshape(nj,ni,1)
                angi_pi = np.arctan2(Y_elip_amp*np.sin(ri +Y_elip_phi)+ mean_y,X_elip_amp*np.sin(ri +X_elip_phi) + mean_x)#.reshape(nj,ni,1)
                angi_2pi = angi_pi.copy()
                angi_2pi[angi_2pi <0]+=2*np.pi

                #pdb.set_trace()
                #ang_max_pi =  np.ma.max(np.ma.append(ang_max_pi, angi_pi,  axis = 2),axis = 2).reshape(nj,ni,1)
                #ang_max_2pi = np.ma.max(np.ma.append(ang_max_2pi,angi_2pi, axis = 2),axis = 2).reshape(nj,ni,1)
                #ang_min_pi =  np.ma.min(np.ma.append(ang_min_pi, angi_pi,  axis = 2),axis = 2).reshape(nj,ni,1)
                #ang_min_2pi = np.ma.min(np.ma.append(ang_min_2pi,angi_2pi, axis = 2),axis = 2).reshape(nj,ni,1)
                ang_max_pi =  np.ma.maximum(ang_max_pi, angi_pi)
                ang_max_2pi = np.ma.maximum(ang_max_2pi,angi_2pi)
                ang_min_pi =  np.ma.minimum(ang_min_pi, angi_pi)
                ang_min_2pi = np.ma.minimum(ang_min_2pi,angi_2pi)

            print(datetime.now(),'Finish ellipse tangent search')
            pdb.set_trace()


        elif calc_method == 'analytical':

            #((a Cos[x] - b Sin[x]) (f + e Cos[x] + d Sin[x]) - (c + b Cos[x] + a Sin[x]) (d Cos[x] - e Sin[x]))/((f + e Cos[x] + d Sin[x])^2 (1 + (c + b Cos[x] + a Sin[x])^2/(f + e Cos[x] + d Sin[x])^2)) = 0

            '''

            #Wolfram alpha

            # find the angle to each point of an ellipse from the origin
            atan(( a*sin(x) + b*cos(x) +c )/(  d*sin(x) + e*cos(x)  +f )

            The min and max gradients are the tangents. Therefore find the derivative of this equation:
            ((a Cos[x] - b Sin[x]) (f + e Cos[x] + d Sin[x]) - (c + b Cos[x] + a Sin[x]) (d Cos[x] - e Sin[x]))/((f + e Cos[x] + d Sin[x])^2 (1 + (c + b Cos[x] + a Sin[x])^2/(f + e Cos[x] + d Sin[x])^2))

            # and then find where the derivative = 0
            ((a Cos[x] - b Sin[x]) (f + e Cos[x] + d Sin[x]) - (c + b Cos[x] + a Sin[x]) (d Cos[x] - e Sin[x]))/((f + e Cos[x] + d Sin[x])^2 (1 + (c + b Cos[x] + a Sin[x])^2/(f + e Cos[x] + d Sin[x])^2)) = 0

            #There are two roots, so two solutions

            x = tan^(-1)((-2 e f a^2 + 2 c d e a + 2 b d f a - 2 b c d^2 - sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)))/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)), (-(e f^2 a^3)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + (b d f^2 a^2)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + (2 c d e f a^2)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + e a - (f sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) a)/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) - (c^2 d^2 e a)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) - (2 b c d^2 f a)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) - b d + (c d sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)))/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) + (b c^2 d^3)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f))/(b f - c e)) + 2 pi c_1

            x = tan^(-1)((-2 e f a^2 + 2 c d e a + 2 b d f a - 2 b c d^2 + sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)))/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)), (-(e f^2 a^3)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + (b d f^2 a^2)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + (2 c d e f a^2)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + e a + (f sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) a)/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) - (c^2 d^2 e a)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) - (2 b c d^2 f a)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) - b d - (c d sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)))/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) + (b c^2 d^3)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f))/(b f - c e)) + 2 pi c_1

            x = np.pi* np.arange(361.)/180.

            ii,jj = 152,281

            ii,jj =  176,150


            (a) = Y_elip_amp[jj,ii]*np.cos(Y_elip_phi[jj,ii])
            (b) = Y_elip_amp[jj,ii]*np.sin(Y_elip_phi[jj,ii])
            (c) = mean_y[jj,ii]
            (d) = X_elip_amp[jj,ii]*np.cos(X_elip_phi[jj,ii])
            (e) = X_elip_amp[jj,ii]*np.sin(X_elip_phi[jj,ii])
            (f) = mean_x[jj,ii]

            #(c) = 0.005
            #(f) = 0.045


            ell_x = (  d*sin(x) + e*cos(x)  +f )
            ell_y = ( a*sin(x) + b*cos(x) + c )

            y = arctan2(ell_y,ell_x)
            dydx = ((a*cos(x)-b*sin(x))*(d*sin(x)+e*cos(x)+f)-(d*cos(x)-e*sin(x))*(a*sin(x)+b*cos(x)+c))/(((d*sin(x)+e*cos(x)+f)**2)*((a*sin(x)+b*cos(x)+c)**2/(d*sin(x)+e*cos(x)+f)**2+1))



            tmp_denom_1 = (-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2-sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))

            tmp_num_1 = (-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a-(f*sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d+(c*d*sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e)




            tmp_denom_2 =(-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2+sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))

            tmp_num_2 = (-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a+(f*sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d-(c*d*sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e)




            tang_grad_1 = np.arctan2(tmp_num_1,tmp_denom_1)
            tang_grad_2 = np.arctan2(tmp_num_2,tmp_denom_2)

            x = tang_grad_1
            dydx_tang_1  = ((a*cos(x)-b*sin(x))*(d*sin(x)+e*cos(x)+f)-(d*cos(x)-e*sin(x))*(a*sin(x)+b*cos(x)+c))/(((d*sin(x)+e*cos(x)+f)**2)*((a*sin(x)+b*cos(x)+c)**2/(d*sin(x)+e*cos(x)+f)**2+1))
            y_tang_1  = arctan2(( a*sin(x) + b*cos(x) +c ),(  d*sin(x) + e*cos(x)  +f ))
            x = tang_grad_2
            dydx_tang_2  = ((a*cos(x)-b*sin(x))*(d*sin(x)+e*cos(x)+f)-(d*cos(x)-e*sin(x))*(a*sin(x)+b*cos(x)+c))/(((d*sin(x)+e*cos(x)+f)**2)*((a*sin(x)+b*cos(x)+c)**2/(d*sin(x)+e*cos(x)+f)**2+1))
            y_tang_2  = arctan2(( a*sin(x) + b*cos(x) +c ),(  d*sin(x) + e*cos(x)  +f ))

            x = np.pi* np.arange(361.)/180.

            plt.subplot(2,1,1)
            plt.plot(x,y,x,dydx)
            plt.axhline()
            plt.axvline(tang_grad_1)
            plt.axvline(tang_grad_2)
            plt.axhline(y_tang_1)
            plt.axhline(y_tang_2)
            plt.subplot(2,1,2)
            plt.plot(ell_x,ell_y)
            plt.plot(f,c,'+')
            plt.axhline()
            plt.axvline()
            plt.axis('equal')
            tmpxlim = np.array(plt.gca().get_xlim())
            tmpylim = np.array(plt.gca().get_ylim())
            plt.plot(tmpxlim,tmpxlim*np.tan(y_tang_1))
            plt.plot(tmpxlim,tmpxlim*np.tan(y_tang_2))
            plt.xlim(tmpxlim)
            plt.ylim(tmpylim)
            plt.show()




            '''




            (a) = n_std*Y_elip_amp*np.cos(Y_elip_phi)
            (b) = n_std*Y_elip_amp*np.sin(Y_elip_phi)
            (c) = mean_y
            (d) = n_std*X_elip_amp*np.cos(X_elip_phi)
            (e) = n_std*X_elip_amp*np.sin(X_elip_phi)
            (f) = mean_x


            #from numpy import cos,sin,arctan2,sqrt
            # turn off warning for invalid values from negative sqrts
            invalerr = np.seterr()['invalid']
            np.seterr(invalid='ignore')
            #print (invalerr, np.seterr()['invalid'])

            tmp_denom_1 = (-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2-np.sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))

            tmp_num_1 = (-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a-(f*np.sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d+(c*d*np.sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e)




            tmp_denom_2 =(-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2+np.sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))

            tmp_num_2 = (-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a+(f*np.sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d-(c*d*np.sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e)


            # reset warning for invalid values from negative sqrts
            np.seterr(invalid=invalerr)
            #pdb.set_trace()

            '''

            #pdb.set_trace()

            set neg vals to zero before sqrt
            tmp_denom_1 = (-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2-np.sqrt(np.maximum((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f) ,0)) )/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))

            tmp_num_1 = (-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a-(f*np.sqrt(np.maximum((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f),0))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d+(c*d*np.sqrt(np.maximum((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f),0)) )/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e)




            tmp_denom_2 =(-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2+np.sqrt(np.maximum((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f),0)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))

            tmp_num_2 = (-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a+(f*np.sqrt(np.maximum((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f),0))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d-(c*d*np.sqrt(np.maximum((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f),0)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e)


            '''
            tang_grad_1 = np.arctan2(tmp_num_1,tmp_denom_1)
            tang_grad_2 = np.arctan2(tmp_num_2,tmp_denom_2)



            y_tang_1  = np.arctan2(( a*np.sin(tang_grad_1) + b*np.cos(tang_grad_1) +c ),(  d*np.sin(tang_grad_1) + e*np.cos(tang_grad_1)  + f ))
            y_tang_2  = np.arctan2(( a*np.sin(tang_grad_2) + b*np.cos(tang_grad_2) +c ),(  d*np.sin(tang_grad_2) + e*np.cos(tang_grad_2)  + f ))


            ang_wid_1 = (y_tang_1-y_tang_2)%(2*np.pi)
            ang_wid_2 = (y_tang_2-y_tang_1)%(2*np.pi)
            ang_wid = ang_wid_1.copy()
            tmpind = ang_wid>np.pi
            ang_wid[tmpind] = ang_wid_2[tmpind]

            '''
            plt.pcolormesh(180*ang_wid/np.pi)
            plt.colorbar()
            plt.show()
            '''

            #tang_grad_1,tang_grad_2,ang_wid

        '''
        from numpy import cos,sin,arctan2,sqrt

        #  so y =  tan(ang) =  Ysin(x+phiy)/Xsin(x+phix) = (asinx + bcosx)/(csinx+dcosx)
        #(y) = np.tan(ang_xy[jj,ii])
        (a) = Y_elip_amp[jj,ii]*np.cos(Y_elip_phi[jj,ii])
        (b) = Y_elip_amp[jj,ii]*np.sin(Y_elip_phi[jj,ii])
        (c) = mean_y[jj,ii]
        (d) = X_elip_amp[jj,ii]*np.cos(X_elip_phi[jj,ii])
        (e) = X_elip_amp[jj,ii]*np.sin(X_elip_phi[jj,ii])
        (f) = mean_x[jj,ii]


        #(n) = 0  #number of 2 x pi added

         x = np.pi* np.arange(361.)/180.
         y= arctan2(( a*sin(x) + b*cos(x) +c ),(  d*sin(x) + e*cos(x)  +f ))
         dydx=((a*cos(x)-b*sin(x))*(d*sin(x)+e*cos(x)+f)-(d*cos(x)-e*sin(x))*(a*sin(x)+b*cos(x)+c))/(((d*sin(x)+e*cos(x)+f)**2)*((a*sin(x)+b*cos(x)+c)**2/(d*sin(x)+e*cos(x)+f)**2+1))

         plt.plot(x,y,x,dydx)
        plt.axhline()
        plt.show()



        wolfram alpha
        ((a Cos[x] - b Sin[x]) (f + e Cos[x] + d Sin[x]) - (c + b Cos[x] + a Sin[x]) (d Cos[x] - e Sin[x]))/((f + e Cos[x] + d Sin[x])^2 (1 + (c + b Cos[x] + a Sin[x])^2/(f + e Cos[x] + d Sin[x])^2)) = 0





        x = tan^(-1)((-2 e f a^2 + 2 c d e a + 2 b d f a - 2 b c d^2 + sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)))/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)), (-(e f^2 a^3)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + (b d f^2 a^2)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + (2 c d e f a^2)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) + e a + (f sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) a)/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) - (c^2 d^2 e a)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) - (2 b c d^2 f a)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f) - b d - (c d sqrt((2 e f a^2 - 2 c d e a - 2 b d f a + 2 b c d^2)^2 - 4 (b^2 d^2 - 2 a b e d + a^2 e^2 - c^2 e^2 - b^2 f^2 + 2 b c e f) (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)))/(2 (c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f)) + (b c^2 d^3)/(c^2 d^2 - 2 a c f d + c^2 e^2 + a^2 f^2 + b^2 f^2 - 2 b c e f))/(b f - c e)) + 2 pi c_1




        arctan_num=((-2*e*f*a**2+2*c*d*e*a+2*b*d*f*a-2*b*c*d**2+sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))

        arctan_denom=((-(e*f**2*a**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(b*d*f**2*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+(2*c*d*e*f*a**2)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)+e*a+(f*sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))*a)/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))-(c**2*d**2*e*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-(2*b*c*d**2*f*a)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)-b*d-(c*d*sqrt((2*e*f*a**2-2*c*d*e*a-2*b*d*f*a+2*b*c*d**2)**2-4*(b**2*d**2-2*a*b*e*d+a**2*e**2-c**2*e**2-b**2*f**2+2*b*c*e*f)*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f)))/(2*(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))+(b*c**2*d**3)/(c**2*d**2-2*a*c*f*d+c**2*e**2+a**2*f**2+b**2*f**2-2*b*c*e*f))/(b*f-c*e))

        np.arctan2(arctan_denom,arctan_num)


        '''

        return y_tang_1,y_tang_2,ang_wid

def find_parameteric_ellipse_foci(qmax, qmin,theta_max,U_mean,V_mean,n_std = 2.45):

    # assuming the distance from foci to elipse curve to the other foci is the same as double the semi-major axes
    #   (hint: think of the point where the semi major axis meets the ellipse curve.)
    #  as the foci are along the semi major axis, and we know the semi minor axis,
    #       simple pythag gives us the distance from the center of the ellipse to the foci
    #       call this the foci_max
    #
    # then calc the foci by rotation and translation.

    foci_max = n_std*np.sqrt(qmax**2 - qmin**2) #distance from the centre of the ellipse to the foci
    foci_x_1 = foci_max * np.cos(theta_max)+ U_mean
    foci_y_1 = foci_max * np.sin(theta_max)+ V_mean
    foci_x_2 = foci_max * np.cos(theta_max+np.pi)+ U_mean
    foci_y_2 = foci_max * np.sin(theta_max+np.pi)+ V_mean

    return foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2

def point_inside_parameteric_ellipse(ii_pnt,jj_pnt, n_std,foci_x_1,foci_y_1,foci_x_2,foci_y_2, qmax):
    # pnt_inside_ell,foci_pnt_foci_dist = point_inside_parameteric_ellipse(ii_pnt,jj_pnt, n_std,foci_x_1,foci_y_1,foci_x_2,foci_y_2, qmax)

    # a point is inside the elipse if the sum of the distance to each of the foci is less than 2 times the semi major ellipse

    foci_pnt_foci_dist = np.sqrt((foci_x_1 - ii_pnt)**2 + (foci_y_1 - jj_pnt)**2) + np.sqrt((foci_x_2 - ii_pnt)**2 + (foci_y_2 - jj_pnt)**2)
    pnt_inside_ell = foci_pnt_foci_dist<(2*qmax*n_std)


    return pnt_inside_ell,foci_pnt_foci_dist

def find_ellipse_tangent_geometric_method_pnt(xpnt,ypnt,u_mean,v_mean,theta_max,qmax,qmin,Xelip,Yelip,show_example = False):

    #Find the tangent of an ellipse from a point using a geometric approach.
    #First transfrom the ellipse to a standard circle, of radius 1, and centre (0,0). Use the same transfromation on the point.
    #Find the distance and angle from the point to the centre of the circle.
    #Fit a right angle triangle from the point to the circle centre (the hypotenuse), and the opposite having a length of the radius (1)
    #   and the right angle at the edge of the circle. Pythag gives the legnht of the adjacent side (the distance form the point to the tangent point)
    #


    #        xpnt,ypnt = 0.,0.

    rotang = theta_max

    #move to the origin
    Xelip_trans = Xelip - u_mean
    Yelip_trans = Yelip - v_mean

    xpnt_trans = xpnt - u_mean
    ypnt_trans = ypnt - v_mean

    # rotate by the ellipse semi-major axis angle (theta max)

    Xelip_rot = Xelip_trans*np.cos(rotang) + Yelip_trans*np.sin(rotang)
    Yelip_rot = -Xelip_trans*np.sin(rotang) + Yelip_trans*np.cos(rotang)

    xpnt_rot = xpnt_trans*np.cos(rotang) + ypnt_trans*np.sin(rotang)
    ypnt_rot = -xpnt_trans*np.sin(rotang) + ypnt_trans*np.cos(rotang)


    #stretch by the semi-major and semi-minor axis
    Xelip_rot_str = Xelip_rot/qmax
    Yelip_rot_str = Yelip_rot/qmin

    xpnt_rot_str = xpnt_rot/qmax
    ypnt_rot_str = ypnt_rot/qmin

    # trig on right angle triangle.

    D_rot_str = np.sqrt((xpnt_rot_str-0.)**2 + (ypnt_rot_str-0.)**2)
    beta = np.arctan2(0-ypnt_rot_str,0-xpnt_rot_str)
    R = 1.
    Td = np.sqrt(D_rot_str**2 - R**2)
    alpha = np.arctan(R/Td)

    # distance from point to the tangent point

    dx_tang_1_rot_str = Td*np.cos(beta + alpha)
    dy_tang_1_rot_str = Td*np.sin(beta + alpha)
    dx_tang_2_rot_str = Td*np.cos(beta - alpha)
    dy_tang_2_rot_str = Td*np.sin(beta - alpha)

    # tangent points

    x_tang_1_rot_str = xpnt_rot_str + dx_tang_1_rot_str
    y_tang_1_rot_str = ypnt_rot_str + dy_tang_1_rot_str
    x_tang_2_rot_str = xpnt_rot_str + dx_tang_2_rot_str
    y_tang_2_rot_str = ypnt_rot_str + dy_tang_2_rot_str

    #transform to original coordinates

    x_tang_1_rot = x_tang_1_rot_str*qmax
    x_tang_2_rot = x_tang_2_rot_str*qmax
    y_tang_1_rot = y_tang_1_rot_str*qmin
    y_tang_2_rot = y_tang_2_rot_str*qmin

    x_tang_1_trans =  x_tang_1_rot*np.cos(-rotang) + y_tang_1_rot*np.sin(-rotang)
    y_tang_1_trans = -x_tang_1_rot*np.sin(-rotang) + y_tang_1_rot*np.cos(-rotang)
    x_tang_2_trans =  x_tang_2_rot*np.cos(-rotang) + y_tang_2_rot*np.sin(-rotang)
    y_tang_2_trans = -x_tang_2_rot*np.sin(-rotang) + y_tang_2_rot*np.cos(-rotang)

    x_tang_1 = x_tang_1_trans + u_mean
    y_tang_1 = y_tang_1_trans + v_mean
    x_tang_2 = x_tang_2_trans + u_mean
    y_tang_2 = y_tang_2_trans + v_mean

    ang_tang_1 = np.arctan2(y_tang_1 - ypnt, x_tang_1 - xpnt )
    ang_tang_2 = np.arctan2(y_tang_2 - ypnt, x_tang_2 - xpnt )


    if show_example:
        plt.plot(Xelip,Yelip,'r')
        plt.plot(xpnt,ypnt,'rx')
        plt.plot(Xelip_rot_str,Yelip_rot_str,'k')
        plt.plot(xpnt_rot_str,ypnt_rot_str,'kx')
        plt.axis('equal')
        plt.plot(xpnt_rot_str+dx_tang_1_rot_str,ypnt_rot_str+dy_tang_1_rot_str, 'g+')
        plt.plot(x_tang_2_rot_str,y_tang_2_rot_str, 'g+')
        plt.plot([xpnt_rot_str,x_tang_1_rot_str],[ypnt_rot_str,y_tang_1_rot_str], 'g')
        plt.plot([xpnt_rot_str,x_tang_2_rot_str],[ypnt_rot_str,y_tang_2_rot_str], 'g')
        plt.plot(x_tang_1,y_tang_1, 'y+')
        plt.plot(x_tang_2,y_tang_2, 'y+')
        plt.plot([xpnt,x_tang_1],[ypnt,y_tang_1], 'y')
        plt.plot([xpnt,x_tang_2],[ypnt,y_tang_2], 'y')
        plt.plot(xpnt + np.array([0,10*np.cos(ang_tang_1)]),ypnt + np.array([0,10*np.sin(ang_tang_1)]))
        plt.plot(xpnt + np.array([0,10*np.cos(ang_tang_2)]),ypnt + np.array([0,10*np.sin(ang_tang_2)]))
        plt.plot([xpnt,x_tang_1],[ypnt,y_tang_1], 'y')
        plt.plot([xpnt,x_tang_2],[ypnt,y_tang_2], 'y')
        plt.plot(u_mean,v_mean,'+')
        plt.axvline(0)
        plt.axhline(0)

        #plt.show()

    return ang_tang_1,ang_tang_2,x_tang_1,y_tang_1,x_tang_2,y_tang_2

    '''
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    # https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    # https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """



    if x.size != y.size:
        raise ValueError("x and y must be the same size")



    #  ax = plt.gca()
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)


    #transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)



    return ax.add_patch(ellipse)
    '''

def ellipse_area_single(tmpx_test,tmpy_test,X_elip_amp,X_elip_phi,U_mean,Y_elip_amp,Y_elip_phi,V_mean,foci_x_1,foci_y_1,foci_x_2,foci_y_2,qmax,npnt_counting = 100, n_std = 2.45):

    #npnt_counting = 10   # Area accuracy = 110.15 %
    #npnt_counting = 100  # Area accuracy = 100.12 %
    #npnt_counting = 1000 # Area accuracy = 100.005 %
    #npnt_counting = 10000 # crashed: too much memory


    tmpx_test_mat,tmpy_test_mat = np.meshgrid(tmpx_test,tmpy_test)




    ang = np.pi*np.arange(361)/180.

    Xo_196_1 = n_std*(X_elip_amp*np.sin(ang + X_elip_phi))+U_mean
    Yo_196_1 = n_std*(Y_elip_amp*np.sin(ang + Y_elip_phi))+V_mean

    pnt_inside_ell_sig_1,foci_pnt_foci_dist_sig = point_inside_parameteric_ellipse(tmpx_test_mat,tmpy_test_mat,n_std,   foci_x_1,foci_y_1,foci_x_2,foci_y_2,qmax )

    return pnt_inside_ell_sig_1

def overlapping_ellipse_area_from_dict(dict_1,dict_2,npnt_counting = 100, output_dict = True, n_std = 2.45):

    X_elip_amp_1 = dict_1['X_elip_amp']
    X_elip_phi_1 = dict_1['X_elip_phi']
    U_mean_1 = dict_1['U_mean']
    Y_elip_amp_1 = dict_1['Y_elip_amp']
    Y_elip_phi_1 = dict_1['Y_elip_phi']
    V_mean_1 = dict_1['V_mean']
    foci_x_1_1 = dict_1['foci_x_1']
    foci_y_1_1 = dict_1['foci_y_1']
    foci_x_2_1 = dict_1['foci_x_2']
    foci_y_2_1 = dict_1['foci_y_2']
    qmax_1 = dict_1['qmax']
    qmin_1 = dict_1['qmin']
    X_elip_amp_2 = dict_2['X_elip_amp']
    X_elip_phi_2 = dict_2['X_elip_phi']
    U_mean_2 = dict_2['U_mean']
    Y_elip_amp_2 = dict_2['Y_elip_amp']
    Y_elip_phi_2 = dict_2['Y_elip_phi']
    V_mean_2 = dict_2['V_mean']
    foci_x_1_2 = dict_2['foci_x_1']
    foci_y_1_2 = dict_2['foci_y_1']
    foci_x_2_2 = dict_2['foci_x_2']
    foci_y_2_2 = dict_2['foci_y_2']
    qmax_2 = dict_2['qmax']
    qmin_2 = dict_2['qmin']


    overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor, perc_overlap, perc_ratio_of_ellipse_area, perc_area_rat = overlapping_ellipse_area(    X_elip_amp_1,X_elip_phi_1,U_mean_1,Y_elip_amp_1,Y_elip_phi_1,V_mean_1,foci_x_1_1,foci_y_1_1,foci_x_2_1,foci_y_2_1,qmax_1,qmin_1,    X_elip_amp_2,X_elip_phi_2,U_mean_2,Y_elip_amp_2,Y_elip_phi_2,V_mean_2,foci_x_1_2,foci_y_1_2,foci_x_2_2,foci_y_2_2,qmax_2,qmin_2,    npnt_counting = 100, n_std = n_std)

    if output_dict:
        dict_out = {}
        dict_out['overlap_1'] = overlap_1
        dict_out['overlap_2'] = overlap_2
        dict_out['overlap_and'] = overlap_and
        dict_out['overlap_or'] = overlap_or
        dict_out['area_1'] = area_1
        dict_out['area_2'] = area_2
        dict_out['overlap_1not2'] = overlap_1not2
        dict_out['overlap_2not1'] = overlap_2not1
        dict_out['overlap_xor'] = overlap_xor
        dict_out['perc_overlap'] = perc_overlap
        dict_out['perc_ratio_of_ellipse_area'] = perc_ratio_of_ellipse_area
        dict_out['perc_area_rat'] = perc_area_rat

        return dict_out

    else:
        return overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor, perc_overlap, perc_ratio_of_ellipse_area, perc_area_rat

def overlapping_ellipse_area(X_elip_amp_1,X_elip_phi_1,U_mean_1,Y_elip_amp_1,Y_elip_phi_1,V_mean_1,foci_x_1_1,foci_y_1_1,foci_x_2_1,foci_y_2_1,qmax_1,qmin_1,X_elip_amp_2,X_elip_phi_2,U_mean_2,Y_elip_amp_2,Y_elip_phi_2,V_mean_2,foci_x_1_2,foci_y_1_2,foci_x_2_2,foci_y_2_2,qmax_2,qmin_2,npnt_counting = 100, verbose = True, n_std = 2.45):

    # Find the area of the ellipse overlap by comparing to masks
    # Create the masks by asking each point if it is inside the array or outside it.


    #~1:33

    #npnt_counting = 10
    #npnt_counting = 100  70 sec
    #npnt_counting = 1000

    nlat,nlon = X_elip_amp_1.shape

    #nlon = 297#lon.size
    #nlat = 375#lat.size


    #overlap_1 = np.ma.zeros((nlat, nlon))*np.ma.masked
    #overlap_2 = np.ma.zeros((nlat, nlon))*np.ma.masked
    #overlap_and = np.ma.zeros((nlat, nlon))*np.ma.masked
    #overlap_or = np.ma.zeros((nlat, nlon))*np.ma.masked


    overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor = [  np.ma.zeros((nlat, nlon))*np.ma.masked for nai in range(9)  ]

    twoaltone = np.array(([-1,1]))


    if verbose: print('Started ',datetime.now())

    for ii in range(nlon):
        if verbose &  ((ii%50) == 0): print('ii ',ii,datetime.now())
        for jj in range(nlat):
            if X_elip_amp_1.mask[jj,ii]: continue


            Xlim_1 = n_std*(X_elip_amp_1[jj,ii])*twoaltone+U_mean_1[jj,ii]
            Ylim_1 = n_std*(Y_elip_amp_1[jj,ii])*twoaltone+V_mean_1[jj,ii]
            Xlim_2 = n_std*(X_elip_amp_2[jj,ii])*twoaltone+U_mean_2[jj,ii]
            Ylim_2 = n_std*(Y_elip_amp_2[jj,ii])*twoaltone+V_mean_2[jj,ii]

            tmpx_test = np.linspace(np.min((Xlim_1,Xlim_2)),np.max((Xlim_1,Xlim_2)),npnt_counting)
            tmpy_test = np.linspace(np.min((Ylim_1,Ylim_2)),np.max((Ylim_1,Ylim_2)),npnt_counting)



            pnt_inside_ell_sig_1 = ellipse_area_single(tmpx_test,tmpy_test,X_elip_amp_1[jj,ii],X_elip_phi_1[jj,ii],U_mean_1[jj,ii],Y_elip_amp_1[jj,ii],Y_elip_phi_1[jj,ii],V_mean_1[jj,ii], foci_x_1_1[jj,ii],foci_y_1_1[jj,ii],foci_x_2_1[jj,ii],foci_y_2_1[jj,ii],qmax_1[jj,ii],npnt_counting=npnt_counting, n_std = n_std)

            pnt_inside_ell_sig_2 = ellipse_area_single(tmpx_test,tmpy_test,X_elip_amp_2[jj,ii],X_elip_phi_2[jj,ii],U_mean_2[jj,ii],Y_elip_amp_2[jj,ii],Y_elip_phi_2[jj,ii],V_mean_2[jj,ii], foci_x_1_2[jj,ii],foci_y_1_2[jj,ii],foci_x_2_2[jj,ii],foci_y_2_2[jj,ii],qmax_2[jj,ii],npnt_counting=npnt_counting, n_std = n_std)

            tmpdx = np.diff(tmpx_test).mean()
            tmpdy = np.diff(tmpy_test).mean()


            area_1[jj,ii] =  np.pi*n_std**2*(qmax_1[jj,ii])*(qmin_1[jj,ii])
            area_2[jj,ii] =  np.pi*n_std**2*(qmax_2[jj,ii])*(qmin_2[jj,ii])

            overlap_1[jj,ii] = (pnt_inside_ell_sig_1).sum()*tmpdx*tmpdy
            overlap_2[jj,ii] = (pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy
            overlap_1not2[jj,ii] = (pnt_inside_ell_sig_1 & (pnt_inside_ell_sig_2 == False)).sum()*tmpdx*tmpdy
            overlap_2not1[jj,ii] = (pnt_inside_ell_sig_2 & (pnt_inside_ell_sig_1 == False)).sum()*tmpdx*tmpdy
            overlap_and[jj,ii] = (pnt_inside_ell_sig_1&pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy
            overlap_or[jj,ii] = (pnt_inside_ell_sig_1|pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy
            overlap_xor[jj,ii] = (pnt_inside_ell_sig_1^pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy




    if verbose: print('Finished ',datetime.now())


    perc_overlap = 100.*overlap_and.astype('float')/overlap_or
    perc_ratio_of_ellipse_area = 100.*overlap_1.astype('float')/overlap_2
    perc_area_rat = 100.*area_1.astype('float')/area_2


    #return overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor


    return overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor, perc_overlap, perc_ratio_of_ellipse_area, perc_area_rat

def charact_val_comp_dist(dict_1, ii_pnt,jj_pnt,n_std = 2.45):
    # sig_change_mat,sig_change_mag_mat = charact_val_comp_dist(dict_1, ii_pnt,jj_pnt,n_std = 2.45)
    nj,ni = dict_1['U_mean'].shape

    iijj_mag = np.sqrt(ii_pnt**2 + jj_pnt**2 )
    iijj_ang = np.arctan2(jj_pnt, ii_pnt )



    U_mean = dict_1['U_mean']
    V_mean = dict_1['V_mean']
    #mag_xy = dict_1['mag_xy']
    ang_xy = dict_1['ang_xy']
    UV_mean = dict_1['UV_mean']
    X_dir_rot_scl_tran =  dict_1['pX_dir']
    Y_dir_rot_scl_tran =dict_1['pY_dir']

    ii_pnt_rot,jj_pnt_rot = iijj_mag*np.cos(ang_xy),iijj_mag*np.sin(ang_xy) # the point if rotated to match angle of mean
    ii_pnt_str,jj_pnt_str = UV_mean*np.cos(iijj_ang),UV_mean*np.sin(iijj_ang) # the point if streched to match length of mean


    XY_std_dir_corr_cur_val,XY_zero_num_std_from_mean_cur_val,pX_dir_cur_val,pY_dir_cur_val = find_num_std_to_point(dict_1['U_mean'],dict_1['V_mean'],dict_1['X_elip_amp'],dict_1['Y_elip_amp'],dict_1['X_elip_phi'],dict_1['Y_elip_phi'],x_pnt = ii_pnt,y_pnt = jj_pnt)

    XY_std_dir_corr_cur_val_rot,XY_zero_num_std_from_mean_cur_val_rot,pX_dir_cur_val,pY_dir_cur_val = find_num_std_to_point(dict_1['U_mean'],dict_1['V_mean'],dict_1['X_elip_amp'],dict_1['Y_elip_amp'],dict_1['X_elip_phi'],dict_1['Y_elip_phi'],x_pnt = ii_pnt_rot,y_pnt = jj_pnt_rot)

    XY_std_dir_corr_cur_val_str,XY_zero_num_std_from_mean_cur_val_str,pX_dir_cur_val,pY_dir_cur_val = find_num_std_to_point(dict_1['U_mean'],dict_1['V_mean'],dict_1['X_elip_amp'],dict_1['Y_elip_amp'],dict_1['X_elip_phi'],dict_1['Y_elip_phi'],x_pnt = ii_pnt_str,y_pnt = jj_pnt_str)




    sig_change_mat =  np.ma.zeros((nj,ni))
    sig_change_mat.mask = U_mean.mask
    sig_change_mat[(XY_zero_num_std_from_mean_cur_val<1)] = 1 # no sig diff at 1std
    sig_change_mat[(XY_zero_num_std_from_mean_cur_val>1) & (XY_zero_num_std_from_mean_cur_val_rot<1) & (XY_zero_num_std_from_mean_cur_val_str<1)] = 2 # Either rot or str will fix it
    sig_change_mat[(XY_zero_num_std_from_mean_cur_val>1) & (XY_zero_num_std_from_mean_cur_val_rot<1) & (XY_zero_num_std_from_mean_cur_val_str>1)] = 3 # Rot will fix it
    sig_change_mat[(XY_zero_num_std_from_mean_cur_val>1) & (XY_zero_num_std_from_mean_cur_val_rot>1) & (XY_zero_num_std_from_mean_cur_val_str<1) & (iijj_mag<UV_mean)] = 4 # Str bigger will fix it
    sig_change_mat[(XY_zero_num_std_from_mean_cur_val>1) & (XY_zero_num_std_from_mean_cur_val_rot>1) & (XY_zero_num_std_from_mean_cur_val_str<1) & (iijj_mag>=UV_mean)] = 5 # Str Smaller will fix it
    sig_change_mat[(XY_zero_num_std_from_mean_cur_val>1)& (XY_zero_num_std_from_mean_cur_val_rot>1) & (XY_zero_num_std_from_mean_cur_val_str>1)] = 6 # neither str or rot will fix it alone
    sig_change_mat.mask = U_mean.mask

    sig_change_mag_mat = XY_zero_num_std_from_mean_cur_val_rot.copy()
    sig_change_mag_mat[((XY_zero_num_std_from_mean_cur_val>1) & (XY_zero_num_std_from_mean_cur_val_rot>1) & (XY_zero_num_std_from_mean_cur_val_str<1)) == False] = np.ma.masked
    sig_change_mag_mat*= np.sign(iijj_mag-UV_mean)


    iijj_ang_diff = (iijj_ang-ang_xy)%(2*np.pi)
    iijj_ang_diff[iijj_ang_diff>np.pi] -= 2*np.pi
    iijj_ang_diff_deg=180*iijj_ang_diff/np.pi
    sig_change_ang_mat = np.abs(iijj_ang_diff_deg.copy())
    sig_change_ang_mat[((XY_zero_num_std_from_mean_cur_val>1) & (XY_zero_num_std_from_mean_cur_val_rot<1) & (XY_zero_num_std_from_mean_cur_val_str>1)) == False] = np.ma.masked



    return sig_change_mat,sig_change_mag_mat

def plot_charact_val_comp_dist(sig_change_mat,sig_change_mag_mat, lat, lon, land_sea_mask ):


    from matplotlib.colors import ListedColormap
    lev_6 = ListedColormap(['w','purple','limegreen','b','r','k'],name = 'lev_6')

    sig_cat = ['No Sig diff','Sig diff\n(rot|mag)','Sig diff\n(rot)','Sig diff\n(mag<clim)','Sig diff\n(mag>clim)','Sig diff\n(rot&mag)']

    fig = plt.figure()
    fig.set_figheight(4.75)
    fig.set_figwidth(7.5)
    fig.suptitle('Characterisation of significant anomalies', fontsize= 20)
    plt.subplots_adjust(top=0.9,bottom=0.05,left=0.075,right=0.95,hspace=0.2,wspace=0.2)

    plt.pcolormesh(lon,lat,sig_change_mat,cmap = lev_6,vmin = 0.5, vmax = 6.5)
    cbar = plt.colorbar(ticks = np.arange(1,6+1))
    cbar.ax.set_yticklabels(sig_cat,  ha = 'left', va = 'center')
    plt.pcolormesh(lon,lat,np.ma.masked_equal(land_sea_mask.astype('int'),0), cmap = 'gray',vmin = 0.5,vmax = 1.5)
    plt.pcolormesh(lon,lat,sig_change_mag_mat,vmin = -3,vmax = 3, cmap = 'bwr')
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel('Relative Magitude anomaly\n(how many std greater than climatology)',  ha = 'center', va = 'top')

    #
    # plt.axis('equal')
    #
    #
    # for tmpax in ax:tmpax.axis([-12.5,10.,47.5,61.])
    # for tmpax in ax:tmpax.axis('equal')
    # for tmpax in ax:tmpax.axis([-12.5,10.,47.5,61.])




    return

def ellipse_overlap_coefficient_pdf_from_dict(dict_1,dict_2,npnt_counting = 100, output_dict = True):

    X_elip_amp_1 = dict_1['X_elip_amp']
    X_elip_phi_1 = dict_1['X_elip_phi']
    U_mean_1 = dict_1['U_mean']
    U_var_1 = dict_1['U_var']
    UV_cov_1 = dict_1['UV_cov']
    Y_elip_amp_1 = dict_1['Y_elip_amp']
    Y_elip_phi_1 = dict_1['Y_elip_phi']
    V_mean_1 = dict_1['V_mean']
    V_var_1 = dict_1['V_var']

    X_elip_amp_2 = dict_2['X_elip_amp']
    X_elip_phi_2 = dict_2['X_elip_phi']
    U_mean_2 = dict_2['U_mean']
    U_var_2 = dict_2['U_var']
    UV_cov_2 = dict_2['UV_cov']
    Y_elip_amp_2 = dict_2['Y_elip_amp']
    Y_elip_phi_2 = dict_2['Y_elip_phi']
    V_mean_2 = dict_2['V_mean']
    V_var_2 = dict_2['V_var']


    gauss_1_2_overlapping_coef,gauss_1_int,gauss_2_int = ellipse_overlap_coefficient_pdf(
    X_elip_amp_1,X_elip_phi_1,U_mean_1,U_var_1,UV_cov_1,    Y_elip_amp_1,Y_elip_phi_1,V_mean_1,V_var_1,    X_elip_amp_2,X_elip_phi_2,U_mean_2,U_var_2,UV_cov_2,   Y_elip_amp_2,Y_elip_phi_2,V_mean_2,V_var_2,    npnt_counting = 100)

    if output_dict:


        dict_out = {}
        dict_out['gauss_1_2_overlapping_coef'] = gauss_1_2_overlapping_coef
        dict_out['OVL'] = gauss_1_2_overlapping_coef
        dict_out['gauss_1_int'] = gauss_1_int
        dict_out['gauss_2_int'] = gauss_2_int
        return dict_out
    else:
        return gauss_1_2_overlapping_coef,gauss_1_int,gauss_2_int


def ellipse_overlap_coefficient_pdf(
    X_elip_amp_1,X_elip_phi_1,U_mean_1,U_var_1,UV_cov_1,    Y_elip_amp_1,Y_elip_phi_1,V_mean_1,V_var_1,    X_elip_amp_2,X_elip_phi_2,U_mean_2,U_var_2,UV_cov_2,   Y_elip_amp_2,Y_elip_phi_2,V_mean_2,V_var_2,    npnt_counting = 100, nstd_lims = 4):


    nlat, nlon = U_var_1.shape
    nlat,nlon = X_elip_amp_1.shape

    U_std_1 = np.sqrt(U_var_1)
    V_std_1 = np.sqrt(V_var_1)
    U_std_2 = np.sqrt(U_var_2)
    V_std_2 = np.sqrt(V_var_2)

    gauss_1_int = np.ma.zeros((nlat, nlon))*np.ma.masked
    gauss_2_int = np.ma.zeros((nlat, nlon))*np.ma.masked
    #gauss_1_2_int = np.ma.zeros((nlat, nlon))*np.ma.masked
    gauss_1_2_overlapping_coef = np.ma.zeros((nlat, nlon))*np.ma.masked

    twoaltone = np.array(([-1,1]))
    ii,jj = 120,120
    for ii in range(nlon):
        if (ii%50) == 0: print('ii ',ii,datetime.now())
        for jj in range(nlat):
            if X_elip_amp_1.mask[jj,ii]: continue

            # npnt_counting = 100

            #nstd_lims = 2.45
            nstd_lims = 4

            Xlim_1 = nstd_lims*(X_elip_amp_1[jj,ii])*twoaltone+U_mean_1[jj,ii]
            Ylim_1 = nstd_lims*(Y_elip_amp_1[jj,ii])*twoaltone+V_mean_1[jj,ii]
            Xlim_2 = nstd_lims*(X_elip_amp_2[jj,ii])*twoaltone+U_mean_2[jj,ii]
            Ylim_2 = nstd_lims*(Y_elip_amp_2[jj,ii])*twoaltone+V_mean_2[jj,ii]

            tmpx = np.linspace(np.min((Xlim_1,Xlim_2)),np.max((Xlim_1,Xlim_2)),npnt_counting)
            tmpy = np.linspace(np.min((Ylim_1,Ylim_2)),np.max((Ylim_1,Ylim_2)),npnt_counting)


            dx = np.diff(tmpx).mean()
            dy = np.diff(tmpy).mean()


            xmat, ymat = np.meshgrid(tmpx,tmpy)

            #pdb.set_trace()

            gauss_func_2d_norm_out_1,A_1 = gauss_func_2d(xmat,ymat,U_mean_1[jj,ii],V_mean_1[jj,ii],U_var_1[jj,ii],V_var_1[jj,ii],UV_cov_1[jj,ii])
            gauss_func_2d_norm_out_2,A_2 = gauss_func_2d(xmat,ymat,U_mean_2[jj,ii],V_mean_2[jj,ii],U_var_2[jj,ii],V_var_2[jj,ii],UV_cov_2[jj,ii])


            # the integral under the bivariate gaussian distribution
            gauss_1_int[jj,ii] = (dx*dy*gauss_func_2d_norm_out_1).sum()
            gauss_2_int[jj,ii] = (dx*dy*gauss_func_2d_norm_out_2).sum()
            #gauss_1_2_int[jj,ii] = (dx*dy*np.sqrt(gauss_func_2d_norm_out_1*gauss_func_2d_norm_out_2)).sum()
            gauss_1_2_overlapping_coef[jj,ii] = (np.minimum(gauss_func_2d_norm_out_1,gauss_func_2d_norm_out_2)*dx*dy).sum()

    return gauss_1_2_overlapping_coef,gauss_1_int,gauss_2_int#, gauss_1_2_int

def gauss_func_2d(xmat,ymat,xmean,ymean,xvar,yvar,uv_cov):


    # jj,ii = 120,120
    # xmean,ymean,xvar,yvar,uv_cov = U_mean_1[jj,ii],V_mean_1[jj,ii],U_var_1[jj,ii],V_var_1[jj,ii],UV_cov_1[jj,ii]
    # npnt_counting = 200
    # Xlim_1 = 2.45*(X_elip_amp_1[jj,ii])*twoaltone+U_mean_1[jj,ii]
    # Ylim_1 = 2.45*(Y_elip_amp_1[jj,ii])*twoaltone+V_mean_1[jj,ii]
    # Xlim_2 = Xlim_1
    # Ylim_2 = Ylim_1
    # Xlim_2 = [-0.25,0.25]
    # Ylim_2 = [-0.25,0.25]
    # npnt_counting = 2000
    # tmpx = np.linspace(np.min((Xlim_1,Xlim_2)),np.max((Xlim_1,Xlim_2)),npnt_counting)
    # tmpy = np.linspace(np.min((Ylim_1,Ylim_2)),np.max((Ylim_1,Ylim_2)),npnt_counting)
    #
    # dx = np.diff(tmpx).mean()
    # dy= np.diff(tmpy).mean()
    #
    # xmat, ymat = np.meshgrid(tmpx,tmpy)
    #
    #

    xstd = np.sqrt(xvar)
    ystd = np.sqrt(yvar)


    #
    #
    # Wikipedia description.
    #
    # A = 1./(2*np.pi*xstd*ystd)
    #
    # gauss_func_2d_norm_out = A*np.exp(-(((xmat - xmean)**2/(2.*xstd**2)) + ((ymat - ymean)**2/(2.*ystd**2))))
    #
    #
    #
    # gauss_func_2d_norm_out_rot = A*np.exp( - (((xmat - xmean)**2)/(2*xvar) - (((xmat - xmean)*(ymat - ymean))/(4*uv_cov)) + (((ymat - ymean)**2)/(2*yvar))) )


    # https://mathworld.wolfram.com/BivariateNormalDistribution.html
    #rho = np.corrcoef(U_mat[:,jj,ii],V_mat[:,jj,ii])[1,0]
    rho = uv_cov/(xstd*ystd)

    A = 1./(2*np.pi*xstd*ystd*np.sqrt(1-rho**2))

    Z = ((xmat - xmean)**2)/(xvar) - 2*rho*((xmat - xmean)*(ymat - ymean))/(xstd*ystd) +  ((ymat - ymean)**2)/(yvar)

    gauss_func_2d_norm_out = A*np.exp(-Z/(2*(1-rho**2)))



    #print (dx*dy*gauss_func_2d_norm_out).sum()

    # Xo_196_1 = 2.45*(X_elip_amp_1[jj,ii]*np.sin(ang + X_elip_phi_1[jj,ii]))+U_mean_1[jj,ii]
    # Yo_196_1 = 2.45*(Y_elip_amp_1[jj,ii]*np.sin(ang + Y_elip_phi_1[jj,ii]))+V_mean_1[jj,ii]
    #
    # gauss_func_2d_norm_out_rot_t1 = A*np.exp( - (((xmat - xmean)**2)/(2*xvar) - (((xmat - xmean)*(ymat - ymean))/(4*uv_cov)) + (((ymat - ymean)**2)/(2*yvar))) )
    # gauss_func_2d_norm_out_rot_t2  = A*np.exp( - (((xmat - xmean)**2)/(xvar) - (((xmat - xmean)*(ymat - ymean))/(2*uv_cov)) + (((ymat - ymean)**2)/(yvar))) )
    # gauss_func_2d_norm_out_rot_t1 = A*np.exp( - (  (((xmat - xmean)**2)/(2*xvar) - (((xmat - xmean)*(ymat - ymean))/(4*uv_cov)) + (((ymat - ymean)**2)/(2*yvar))) ))
    # gauss_func_2d_norm_out_rot_t1 = A*np.exp( - (  (((xmat - xmean)**2)/(2*xvar) - (((xmat - xmean)*(ymat - ymean))/(4*uv_cov)) + (((ymat - ymean)**2)/(2*yvar))) ))
    #
    #
    # plt.pcolormesh(xmat, ymat,gauss_func_2d_norm_out_rot2)
    # plt.contour(xmat, ymat,gauss_func_2d_norm_out_rot2,colors = 'r',linewidths = 2)
    # #plt.contour(xmat, ymat,gauss_func_2d_norm_out_rot,colors = 'y')
    # plt.contour(xmat, ymat,gauss_func_2d_norm_out_rot_t1,colors = 'g')
    # plt.contour(xmat, ymat,gauss_func_2d_norm_out_rot_t2,colors = 'b')
    # plt.plot(Xo_196_1,Yo_196_1,'w',lw = 2)
    # plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'x')
    # plt.axis('equal')
    # plt.show()
    #
    # plt.pcolormesh(xmat, ymat,gauss_func_2d_norm_out_rot2)
    # #plt.contour(xmat, ymat,gauss_func_2d_norm_out,colors = '0.5')
    # #plt.contour(xmat, ymat,gauss_func_2d_norm_out_rot,colors = '0.75')
    # plt.contour(xmat, ymat,gauss_func_2d_norm_out_rot2,colors = '0.25')
    # plt.plot(Xo_196_1,Yo_196_1,'w')
    # plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'x')
    # plt.axis('equal')
    # plt.show()

    return gauss_func_2d_norm_out,A

def overlapping_ellipse_area_pdf(
    X_elip_amp_1,X_elip_phi_1,U_mean_1,Y_elip_amp_1,Y_elip_phi_1,V_mean_1,U_var_1,V_var_1,UV_cov_1,qmax_1,qmin_1,
    X_elip_amp_2,X_elip_phi_2,U_mean_2,Y_elip_amp_2,Y_elip_phi_2,V_mean_2,U_var_2,V_var_2,UV_cov_2,qmax_2,qmin_2,
    npnt_counting = 100, n_std = 2.45):

    # Find the area of the ellipse overlap by comparing to masks
    # Create the masks using the 2d guassian equation

    #~1:45 for amm7
    nlat,nlon = X_elip_amp_1.shape



    overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor = [  np.ma.zeros((nlat, nlon))*np.ma.masked for nai in range(9)  ]

    twoaltone = np.array(([-1,1]))
    ang = np.linspace(-np.pi,np.pi, 100)



    print('Started ',datetime.now())

    for ii in range(nlon):
        if (ii%50) == 0: print('ii ',ii,datetime.now())
        for jj in range(nlat):
            if X_elip_amp_1.mask[jj,ii]: continue


            Xlim_1 = n_std*(X_elip_amp_1[jj,ii])*twoaltone+U_mean_1[jj,ii]
            Ylim_1 = n_std*(Y_elip_amp_1[jj,ii])*twoaltone+V_mean_1[jj,ii]
            Xlim_2 = n_std*(X_elip_amp_2[jj,ii])*twoaltone+U_mean_2[jj,ii]
            Ylim_2 = n_std*(Y_elip_amp_2[jj,ii])*twoaltone+V_mean_2[jj,ii]

            tmpx_test = np.linspace(np.min((Xlim_1,Xlim_2)),np.max((Xlim_1,Xlim_2)),npnt_counting)
            tmpy_test = np.linspace(np.min((Ylim_1,Ylim_2)),np.max((Ylim_1,Ylim_2)),npnt_counting)
            tmpx_test_mat,tmpy_test_mat = np.meshgrid(tmpx_test,tmpy_test)
            tmpdx = np.diff(tmpx_test).mean()
            tmpdy = np.diff(tmpy_test).mean()


            Xo_196_1 = n_std*(X_elip_amp_1[jj,ii]*np.sin(ang + X_elip_phi_1[jj,ii]))+U_mean_1[jj,ii]
            Yo_196_1 = n_std*(Y_elip_amp_1[jj,ii]*np.sin(ang + Y_elip_phi_1[jj,ii]))+V_mean_1[jj,ii]
            Xo_196_2 = n_std*(X_elip_amp_2[jj,ii]*np.sin(ang + X_elip_phi_2[jj,ii]))+U_mean_2[jj,ii]
            Yo_196_2 = n_std*(Y_elip_amp_2[jj,ii]*np.sin(ang + Y_elip_phi_2[jj,ii]))+V_mean_2[jj,ii]



            gauss_1 = gauss_func_2d(tmpx_test_mat,tmpy_test_mat,U_mean_1[jj,ii],V_mean_1[jj,ii],U_var_1[jj,ii],V_var_1[jj,ii],UV_cov_1[jj,ii])[0]
            gauss_ell_1 = gauss_func_2d(Xo_196_1,Yo_196_1,U_mean_1[jj,ii],V_mean_1[jj,ii],U_var_1[jj,ii],V_var_1[jj,ii],UV_cov_1[jj,ii])[0]

            gauss_2 = gauss_func_2d(tmpx_test_mat,tmpy_test_mat,U_mean_2[jj,ii],V_mean_2[jj,ii],U_var_2[jj,ii],V_var_2[jj,ii],UV_cov_2[jj,ii])[0]
            gauss_ell_2 = gauss_func_2d(Xo_196_2,Yo_196_2,U_mean_2[jj,ii],V_mean_2[jj,ii],U_var_2[jj,ii],V_var_2[jj,ii],UV_cov_2[jj,ii])[0]



            gauss_mask_1 = gauss_1>gauss_ell_1.mean()
            gauss_mask_2 = gauss_2>gauss_ell_2.mean()
            pnt_inside_ell_sig_1 = gauss_mask_1
            pnt_inside_ell_sig_2 = gauss_mask_2


            area_1[jj,ii] =  np.pi*n_std**2*(qmax_1[jj,ii])*(qmin_1[jj,ii])
            area_2[jj,ii] =  np.pi*n_std**2*(qmax_2[jj,ii])*(qmin_2[jj,ii])

            overlap_1[jj,ii] = (pnt_inside_ell_sig_1).sum()*tmpdx*tmpdy
            overlap_2[jj,ii] = (pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy
            overlap_1not2[jj,ii] = (pnt_inside_ell_sig_1 & (pnt_inside_ell_sig_2 == False)).sum()*tmpdx*tmpdy
            overlap_2not1[jj,ii] = (pnt_inside_ell_sig_2 & (pnt_inside_ell_sig_1 == False)).sum()*tmpdx*tmpdy
            overlap_and[jj,ii] = (pnt_inside_ell_sig_1&pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy
            overlap_or[jj,ii] = (pnt_inside_ell_sig_1|pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy
            overlap_xor[jj,ii] = (pnt_inside_ell_sig_1^pnt_inside_ell_sig_2).sum()*tmpdx*tmpdy




    print('Finished ',datetime.now())

    return overlap_1,overlap_2,overlap_and,overlap_or,area_1,area_2,overlap_1not2,overlap_2not1,overlap_xor





def ellipse_params_uv_stats(U_mean,V_mean,UV_mean,U_std,V_std,U_var,V_var,UV_cov,ang_xy, n_std = 2.45,pnt_x = 0, pnt_y = 0):

    #U_mat,V_mat = U_mean*np.ma.masked,V_mean*np.ma.masked
    UV_mat = np.sqrt(V_mean**2+U_mean**2)*np.ma.masked


    X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_stats_parametric_equation(U_mean,V_mean,U_var,V_var,UV_cov, n_std = n_std)
    qmax,qmin, ecc, theta_max, zero_ang = ellipse_parameters_from_parametric_equation(X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,U_mean,V_mean)
    XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir = find_num_std_to_point(U_mean,V_mean,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi)
    y_tang_1,y_tang_2,ang_wid = find_tangent_to_parametric_ellipse_at_a_point(U_mean,V_mean,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,pnt_x = pnt_x, pnt_y = pnt_y, n_std = n_std)
    foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2 = find_parameteric_ellipse_foci(qmax, qmin,theta_max,U_mean,V_mean, n_std = n_std)


    return U_mean,V_mean,UV_mean,U_std,V_std,U_var,V_var,UV_cov,UV_mat,ang_xy,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos,qmax,qmin, ecc, theta_max, zero_ang,XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir,y_tang_1,y_tang_2,ang_wid,foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2





def ens_ellipse_overlap_coefficient_pdf_dict(ens_dict,    npnt_counting = 100, nstd_lims = 4):
    '''
    ens_dict = {}
    for ens in tmp_ens_mat: ens_dict[ens] = ellipse_params_add_to_dict(ellipse_params_uv_stats(...))
    '''

    tmp_ens_mat = [ss for ss in ens_dict.keys()]
    tmp_n_ens = len(tmp_ens_mat)
    eg_ens = tmp_ens_mat[0]
    nlat, nlon = ens_dict[eg_ens]['X_elip_amp'].shape


    #gauss_1_2_overlapping_coef = np.ma.zeros((nlat, nlon))*np.ma.masked
    ens_min_gauss_mat = np.ma.zeros((nlat, nlon))*np.ma.masked

    twoaltone = np.array(([-1,1]))
    #ii,jj = 120,120
    for ii in range(nlon):
        if (ii%50) == 0: print('ii ',ii,datetime.now())
        for jj in range(nlat):
            if ens_dict[eg_ens]['X_elip_amp'].mask[jj,ii]: continue

            nstd_lims = 4


            tmp_Xlim_lst = []
            tmp_Ylim_lst = []

            for ei,tmp_ens in enumerate(tmp_ens_mat):
                tmp_Xlim_lst.append(nstd_lims*(ens_dict[tmp_ens]['X_elip_amp'][jj,ii])*twoaltone+ens_dict[tmp_ens]['U_mean'][jj,ii])
                tmp_Ylim_lst.append(nstd_lims*(ens_dict[tmp_ens]['Y_elip_amp'][jj,ii])*twoaltone+ens_dict[tmp_ens]['V_mean'][jj,ii])

            tmp_Xlim_mat = np.array(tmp_Xlim_lst)
            tmp_Ylim_mat = np.array(tmp_Ylim_lst)

            tmpx = np.linspace(tmp_Xlim_mat.min(),tmp_Xlim_mat.max(),npnt_counting)
            tmpy = np.linspace(tmp_Ylim_mat.min(),tmp_Ylim_mat.max(),npnt_counting)


            xmat, ymat = np.meshgrid(tmpx,tmpy)

            dx = np.diff(tmpx).mean()
            dy = np.diff(tmpy).mean()

            #pdb.set_trace()

            tmp_gauss_mat =  np.zeros((tmp_n_ens,npnt_counting, npnt_counting))


            for ei,tmp_ens in enumerate(tmp_ens_mat):                tmp_gauss_mat[ei,:,:] =  gauss_func_2d(xmat,ymat, ens_dict[tmp_ens]['U_mean'][jj,ii],ens_dict[tmp_ens]['V_mean'][jj,ii],ens_dict[tmp_ens]['U_var'][jj,ii],ens_dict[tmp_ens]['V_var'][jj,ii],ens_dict[tmp_ens]['UV_cov'][jj,ii])[0]
            #pdb.set_trace()
            ens_min_gauss_mat[jj,ii] = dx*dy*tmp_gauss_mat.min(axis = 0).sum()

    return ens_min_gauss_mat

#def main():
#
#if __name__ == "__main__":
#    main()
