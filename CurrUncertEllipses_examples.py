
from netCDF4 import Dataset,num2date



import pdb
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from CurrUncertEllipses import *
from CurrUncertEllipses_ChiSqProb import *

def main():

    example_for_uncertainty_ellipses()

def example_for_uncertainty_ellipses(datadir='./'):

    #Load datasets, with time in the dimension [0].

    lon,lat,baroc_tide_Umat,baroc_tide_Vmat,baroc_notide_Umat,baroc_notide_Vmat = load_example_dataset(datadir=datadir)

    # number of years
    nyr = baroc_tide_Umat.shape[0]//12

    #Extract winter (DJF) months, (skipping the JF from first year, and D from last, to give contigious winter months)
    baroc_tide_Umat = baroc_tide_Umat.reshape(nyr,12,375, 297)[:,[0,1,11],:,:].reshape(nyr*3,375,297)[2:-1,:,:]
    baroc_tide_Vmat = baroc_tide_Vmat.reshape(nyr,12,375, 297)[:,[0,1,11],:,:].reshape(nyr*3,375,297)[2:-1,:,:]
    baroc_notide_Umat = baroc_notide_Umat.reshape(nyr,12,375, 297)[:,[0,1,11],:,:].reshape(nyr*3,375,297)[2:-1,:,:]
    baroc_notide_Vmat = baroc_notide_Vmat.reshape(nyr,12,375, 297)[:,[0,1,11],:,:].reshape(nyr*3,375,297)[2:-1,:,:]

    U_mat,V_mat = baroc_tide_Umat,baroc_tide_Vmat




    #Calculate Ellipse parameters from two data sets.
    #This can be done with ellipse_params, but ellipse_params_add_to_dict adds
    #them to a dictionary, rather than lots of variables, which can make it
    #easier too compare two datasets

    n_std = 2.45
    ens_ellipse = {}
    ens_ellipse['baroc_tide'] = ellipse_params_add_to_dict(ellipse_params(baroc_tide_Umat,baroc_tide_Vmat, n_std=n_std))
    ens_ellipse['baroc_notide'] = ellipse_params_add_to_dict(ellipse_params(baroc_notide_Umat,baroc_notide_Vmat, n_std=n_std))

    #Compare two the ellipses from two datasets
    overlap_dict = overlapping_ellipse_area_from_dict( ens_ellipse['baroc_tide'], ens_ellipse['baroc_notide'])

    #Compare two the distributions from two datasets
    OVL_dict = ellipse_overlap_coefficient_pdf_from_dict(ens_ellipse['baroc_tide'], ens_ellipse['baroc_notide'])

    dict_1,dict_2 = ens_ellipse['baroc_tide'], ens_ellipse['baroc_notide']


    U_mat_1 = baroc_tide_Umat
    V_mat_1 = baroc_tide_Vmat
    U_mat_2 = baroc_notide_Umat
    V_mat_2 = baroc_notide_Vmat

    ii_pnt = U_mat_2[6,:,:]
    jj_pnt = V_mat_2[6,:,:]
    land_sea_mask = ii_pnt.mask



    #Examples of analysis and visualation from one distribution
    example_analysis_one_dist(U_mat_1,V_mat_1,lat,lon,n_std=2.45)
    plt.show()

    #Examples of analysis and visualation when comparing two distributions
    # ~2 mins to run
    example_analysis_two_dist(U_mat_1,V_mat_1, U_mat_2,V_mat_2,lat,lon,n_std=2.45)
    plt.show()

    #Examples of analysis and visualation when comparing an exemplar current field to a distributions
    example_analysis_one_dist_one_value(U_mat_1,V_mat_1, ii_pnt,jj_pnt,lat, lon, land_sea_mask, n_std=2.45)
    plt.show()


    #Example ellipse from a given point.

    #Define location
    ii,jj = 200,200
    dict_in = dict_1
    U_mat,V_mat = U_mat_1,V_mat_1

    #Example of an ellipse from a point in one distribution
    example_ellipse_one_dist(dict_in,U_mat,V_mat,ii,jj,n_std=2.45)
    plt.show()

    #Example of two ellipses from two distribution at one point
    example_ellipse_two_dist(dict_1,dict_2,U_mat_1,V_mat_1, U_mat_2,V_mat_2,ii,jj,n_std=2.45)
    plt.show()

    #Example of an ellipse from a point in one distribution, and a comparison to an exemplar
    example_ellipse_one_dist_one_point(dict_in,U_mat,V_mat,ii_pnt, jj_pnt, ii,jj,n_std=2.45)
    plt.show()

    #Appendix Figures 1 and 4 from Tinker et al. 2022
    #Example figures to help explain the methodology.

    fig1,fig2 = example_figures_from_paper()
    plt.show()



    fig = curr_uncert_prob_threshold_perc_data_in_xsd_figure()
    plt.show()
    curr_uncert_prob_threshold_perc_data_in_xsd_table()

    pdb.set_trace()


def load_example_dataset(datadir='./'):


    lon = np.arange(-19.888889,12.99967+1/9.,1/9.)
    lat = np.arange(40.066669,65+1/15.,1/15.)

    rootgrp = Dataset(datadir + 'baroc_tide_U.nc', 'r', format='NETCDF4')
    baroc_tide_Umat = rootgrp.variables['vobtcrtx_dood_dm'][:]
    rootgrp.close()
    rootgrp = Dataset(datadir + 'baroc_tide_V.nc', 'r', format='NETCDF4')
    baroc_tide_Vmat = rootgrp.variables['vobtcrty_dood_dm'][:]
    rootgrp.close()
    rootgrp = Dataset(datadir + 'baroc_notide_U.nc', 'r', format='NETCDF4')
    baroc_notide_Umat = rootgrp.variables['vobtcrtx_dood_dm'][:]
    rootgrp.close()
    rootgrp = Dataset(datadir + 'baroc_notide_V.nc', 'r', format='NETCDF4')
    baroc_notide_Vmat = rootgrp.variables['vobtcrty_dood_dm'][:]
    rootgrp.close()



    #pdb.set_trace()
    return lon,lat,baroc_tide_Umat,baroc_tide_Vmat,baroc_notide_Umat,baroc_notide_Vmat


def example_figures_from_paper(n_std = 2.45):

    # Create Appendix figureF 2 and 5 from Tinker et al 2022.

    ang = np.pi*np.arange(-180.,181.)/180. # ang = np.linspace(-np.pi,np.pi,360*1000)


    #Create some random data, and uncertainty ellipses for them

    mu1,sig1 = 5.,1.6
    mu2,sig2 = -3.5,1.05
    rho = -0.73

    norm1 = np.random.normal(0,1, 1000000)
    norm2 = np.random.normal(0,1, 1000000)

    norm3 = (rho)*norm1 +np.sqrt(1-rho**2)*norm2
    norm_rp10 = (1.)*norm1 +np.sqrt(1-1.**2)*norm2
    norm_rp05 = (0.5)*norm1 +np.sqrt(1-0.5**2)*norm2
    norm_rp00 = (0.)*norm1 +np.sqrt(1-0.**2)*norm2
    norm_rm10 = (-1.)*norm1 +np.sqrt(1-(-1.)**2)*norm2

    U_mat = mu1 + sig1*norm1
    V_mat = mu2 + sig2*norm3


    V_mat_rp10 = mu2 + sig2*norm_rp10
    V_mat_rp05 = mu2 + sig2*norm_rp05
    V_mat_rp00 = mu2 + sig2*norm_rp00
    V_mat_rm10 = mu2 + sig2*norm_rm10


    U_mat  = U_mat.reshape(-1,1,1)
    V_mat  = V_mat.reshape(-1,1,1)

    U_mat = np.tile(U_mat,(1,1,2))
    V_mat = np.tile(V_mat,(1,1,2))
    U_mat[:,:,1]*=0.5
    U_mat[:,:,1]+=.1
    V_mat[:,:,1]*=0.4
    V_mat[:,:,1]+=.2



    U_mean,V_mean = U_mat.mean(axis = 0),V_mat.mean(axis = 0)
    U_var,V_var = U_mat.var(axis = 0),V_mat.var(axis = 0)

    UV_cov =  ((U_mat - U_mean)*(V_mat - V_mean)).mean(axis = 0)

    jj,ii = 0,0

    X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,X_elip_phi_cos,Y_elip_phi_cos = confidence_ellipse_uv_mat_parametric_equation(U_mat,V_mat, n_std = n_std)

    qmax,qmin, ecc, theta_max, zero_ang = ellipse_parameters_from_parametric_equation(X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,U_mean,V_mean)
    XY_std_dir_corr,XY_zero_num_std_from_mean,pX_dir,pY_dir = find_num_std_to_point(U_mean,V_mean,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi)
    y_tang_1,y_tang_2,ang_wid = find_tangent_to_parametric_ellipse_at_a_point(U_mean,V_mean,X_elip_amp,Y_elip_amp,X_elip_phi,Y_elip_phi,pnt_x = 0, pnt_y = 0, n_std = n_std)
    foci_max,foci_x_1,foci_y_1,foci_x_2,foci_y_2 = find_parameteric_ellipse_foci(qmax, qmin,theta_max,U_mean,V_mean,n_std)


    Xelip = n_std*X_elip_amp[jj,ii] * np.sin(ang +X_elip_phi[jj,ii] ) + U_mean[jj,ii]
    Yelip = n_std*Y_elip_amp[jj,ii] * np.sin(ang +Y_elip_phi[jj,ii] ) + V_mean[jj,ii]

    Xelip_0 = XY_zero_num_std_from_mean[jj,ii]*X_elip_amp[jj,ii] * np.sin(ang +X_elip_phi[jj,ii] )  + U_mean[jj,ii]
    Yelip_0 = XY_zero_num_std_from_mean[jj,ii]*Y_elip_amp[jj,ii] * np.sin(ang +Y_elip_phi[jj,ii] ) + V_mean[jj,ii]


    Xelip_1 = 1.*X_elip_amp[jj,ii] * np.sin(ang +X_elip_phi[jj,ii] ) + U_mean[jj,ii]
    Yelip_1 = 1.*Y_elip_amp[jj,ii] * np.sin(ang +Y_elip_phi[jj,ii] ) + V_mean[jj,ii]



    #Rasterize the ellipse
    xlim = np.array([-5,15])
    ylim = np.array([-12.5,4])
    npnt_counting = 500
    tmpx_test = np.linspace(xlim[0],xlim[1],npnt_counting)
    tmpy_test = np.linspace(ylim[0],ylim[1],npnt_counting)
    dx = np.diff(tmpx_test).mean()
    dy = np.diff(tmpy_test).mean()

    tmpx_test_mat,tmpy_test_mat = np.meshgrid(tmpx_test,tmpy_test)



    out1 = ellipse_area_single(tmpx_test,tmpy_test,X_elip_amp[jj,ii],X_elip_phi[jj,ii],U_mean[jj,ii],Y_elip_amp[jj,ii],Y_elip_phi[jj,ii],V_mean[jj,ii],foci_x_1[jj,ii],foci_y_1[jj,ii],foci_x_2[jj,ii],foci_y_2[jj,ii],qmax[jj,ii], n_std = n_std)

    out2 = ellipse_area_single(tmpx_test,tmpy_test,X_elip_amp[jj,1],X_elip_phi[jj,1],U_mean[jj,1],Y_elip_amp[jj,1],Y_elip_phi[jj,1],V_mean[jj,1],foci_x_1[jj,1],foci_y_1[jj,1],foci_x_2[jj,1],foci_y_2[jj,1],qmax[jj,1], n_std = n_std)


    print(datetime.now())



    Xelip2 = n_std*X_elip_amp[jj,1] * np.sin(ang +X_elip_phi[jj,1] ) + U_mean[jj,1]
    Yelip2 = n_std*Y_elip_amp[jj,1] * np.sin(ang +Y_elip_phi[jj,1] ) + V_mean[jj,1]

    xpnt,ypnt = 0.,0.,
    #fig0 = plt.figure()
    ang_tang_1,ang_tang_2,x_tang_1_pnt_meth,y_tang_1_pnt_meth,x_tang_2_pnt_meth,y_tang_2_pnt_meth= find_ellipse_tangent_geometric_method_pnt(xpnt,ypnt,U_mean[jj,ii],V_mean[jj,ii],theta_max[jj,ii],n_std*qmax[jj,ii],n_std*qmin[jj,ii],Xelip,Yelip,show_example = False)



    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    from scipy.stats import norm

    xvals= np.linspace(xlim[0],xlim[1],1000)
    yvals= np.linspace(ylim[0],ylim[1],1000)

    xnorm = norm.pdf(xvals,U_mat[:,0,0].mean(),U_mat[:,0,0].std())
    ynorm = norm.pdf(yvals,V_mat_rp00.mean(),V_mat_rp00.std())
    xnorm2 = norm.pdf(xvals,U_mat[:,jj,1].mean(),U_mat[:,jj,1].std())



    letter_mat = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


    #plot Appendix figure 2, to show how the method is built up.

    fig1 = plt.figure()
    fig1.set_figheight(5.0)
    fig1.set_figwidth(15.0)
    fig1.set_figheight(9)
    fig1.set_figwidth(9.0)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
    ax = []
    ax.append(plt.subplot(3,3,1))
    plt.plot(U_mat[:50000,0,0],V_mat_rp00[:50000],'.',ms  = 0.2, alpha = 1, color  = 'darkgreen')
    plt.plot(U_mat[:50000,0,0],V_mat_rp05[:50000],'.',ms  = 0.2, alpha = 1, color  = 'r')
    plt.plot(U_mat[:50000,0,0],V_mat_rp10[:50000],'.',ms  = 0.2, alpha = 1, color  = 'k')
    plt.plot(U_mat[:50000,0,0],V_mat_rm10[:50000],'.',ms  = 0.2, alpha = 1, color  = 'gold')
    plt.plot(xvals,10*xnorm+ylim[0], '0.5')
    plt.plot(10*ynorm+xlim[0],yvals,'0.5')
    plt.axhline(V_mat_rp05.mean() + n_std*V_mat_rp05.std(), color = '0.95', ls = '--')
    plt.axhline(V_mat_rp05.mean() - n_std*V_mat_rp05.std(), color = '0.95', ls = '--')
    plt.axvline(U_mat[:,0,0].mean() + n_std*U_mat[:,0,0].std(), color = '0.95', ls = '--')
    plt.axvline(U_mat[:,0,0].mean() - n_std*U_mat[:,0,0].std(), color = '0.95', ls = '--')
    ax.append(plt.subplot(3,3,2))
    plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'.',ms  = 0.1, alpha = 0.05, color  = 'b')
    plt.plot(U_mat[:20,jj,ii],V_mat[:20,jj,ii],'o',color = 'gold',ms  = 3)
    plt.plot(U_mean[jj,ii],V_mean[jj,ii],'k+')
    ax.append(plt.subplot(3,3,3))
    plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'.',ms  = 0.1, alpha = 0.1, color  = 'b')
    plt.plot(U_mat[:20,jj,ii],V_mat[:20,jj,ii],'o',color = 'gold',ms  = 3)
    plt.plot(U_mean[jj,ii],V_mean[jj,ii],'k+')
    plt.plot(Xelip,Yelip,'r')
    ax.append(plt.subplot(3,3,4))
    plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'.',ms  = 0.1, alpha = 0.1, color  = 'b')
    plt.plot(U_mat[:20,jj,ii],V_mat[:20,jj,ii],'o',color = 'gold',ms  = 3)
    plt.plot(U_mean[jj,ii],V_mean[jj,ii],'k+')
    plt.plot(Xelip,Yelip,'r')
    plt.plot(Xelip_0,Yelip_0,'g')
    plt.text(1,-9,'origin at %.2f std'%XY_zero_num_std_from_mean[0,0])
    ax.append(plt.subplot(3,3,5))
    plt.plot(U_mat[:,jj,ii]-2,V_mat[:,jj,ii]+2,'.',ms  = 0.1, alpha = 0.1, color  = 'b')
    plt.plot(U_mat[:20,jj,ii]-2,V_mat[:20,jj,ii]+2,'o',color = 'gold',ms  = 3)
    plt.plot(U_mean[jj,ii]-2,V_mean[jj,ii]+2,'k+')
    plt.plot(Xelip-2,Yelip+2,'r')
    plt.plot(Xelip_1-2,Yelip_1+2,'y')
    ax.append(plt.subplot(3,3,6))
    plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'.',ms  = 0.1, alpha = 0.1, color  = 'b')
    plt.plot(U_mat[:20,jj,ii],V_mat[:20,jj,ii],'o',color = 'gold',ms  = 3)
    plt.plot(U_mean[jj,ii],V_mean[jj,ii],'k+')
    plt.plot(Xelip,Yelip,'r')
    plt.plot(Xelip_0,Yelip_0,'g')
    plt.plot(xlim*np.cos(y_tang_1[jj,ii]),xlim*np.sin(y_tang_1[jj,ii]),'0.5')
    plt.plot(xlim*np.cos(y_tang_2[jj,ii]),xlim*np.sin(y_tang_2[jj,ii]),'0.5')
    plt.text(0.95,0.05,'Tangent at %.1f and %.1f'%(180*y_tang_1[0,0]/np.pi,180*y_tang_2[0,0]/np.pi),ha = 'right',va = 'bottom',transform=ax[-1].transAxes,bbox=dict(facecolor='white', alpha=0.75, pad=4, edgecolor='none'))
    ax.append(plt.subplot(3,3,7))
    plt.pcolormesh(tmpx_test,tmpy_test,out1.astype('int'))
    plt.plot(Xelip,Yelip,'r')
    ax.append(plt.subplot(3,3,8))
    plt.pcolormesh(tmpx_test,tmpy_test,out1.astype('int') + out2.astype('int'))
    plt.plot(Xelip,Yelip,'r')
    plt.plot(Xelip2,Yelip2,'k')
    ax.append(plt.subplot(3,3,9))
    plt.fill(np.minimum(xnorm, xnorm2))
    plt.plot(xnorm2,'r', lw = 2)
    plt.plot(xnorm,'r', lw = 2)
    plt.axhline(0, color = 'k', lw = 2)
    for ai,tmpax in enumerate(ax): plt.text(0.05,0.95,letter_mat[ai]+')',ha = 'left', va = 'top',fontsize = 16,transform=tmpax.transAxes,bbox=dict(facecolor='white', alpha=0.75, pad=4, edgecolor='none'))
    for ai,tmpax in enumerate(ax[:-1]): tmpax.axhline(0, color = '0.5')
    for ai,tmpax in enumerate(ax[:-1]): tmpax.axvline(0, color = '0.5')
    #for ai,tmpax in enumerate(ax): tmpax.axis('equal')
    for ai,tmpax in enumerate(ax[:-1]): tmpax.set_xlim(xlim)
    for ai,tmpax in enumerate(ax[:-1]): tmpax.set_ylim(ylim)
    for tmpax in ax[:6] + ax[-1:]: tmpax.set_xticks([])
    for tmpax in ax[1:3] + ax[4:6] + ax[7:9]: tmpax.set_yticks([])


    # Create Appendix Figure 5, to show how to compare a single residual current
    # to a climatology using uncertainty ellipses.

    mean_ang = np.arctan2(V_mean[jj,ii],U_mean[jj,ii])

    # First, find circles arond the origin pass through the centre and edges of
    # the ellipse
    dist_from_origin = np.sqrt(Xelip**2 + Yelip**2)
    dist_from_origin_min = dist_from_origin.min()
    dist_from_origin_max = dist_from_origin.max()

    dist_from_origin_to_mean = np.sqrt(V_mean[jj,ii]**2+U_mean[jj,ii]**2)



    Xcirc_mean = dist_from_origin_to_mean * np.cos(ang )
    Ycirc_mean = dist_from_origin_to_mean * np.sin(ang )

    Xcirc_min = dist_from_origin_min * np.cos(ang )
    Ycirc_min = dist_from_origin_min * np.sin(ang )
    Xcirc_max = dist_from_origin_max * np.cos(ang )
    Ycirc_max = dist_from_origin_max * np.sin(ang )

    np.linspace(y_tang_1[jj,ii],y_tang_2[jj,ii],1000)


    # Using these circles, and the tangents, find the patches for the different
    # regions described in the appendix of Tinker et al 2022.

    arc_ang_inner = np.append(y_tang_1[jj,ii],(np.append(np.linspace(y_tang_1[jj,ii],y_tang_2[jj,ii],1000),y_tang_2[jj,ii])))
    arc_dist_inner = [0] + [dist_from_origin_min]*1000 + [0]


    arc_ang_outer = np.append(np.linspace(y_tang_1[jj,ii],y_tang_2[jj,ii],1000),np.linspace(y_tang_2[jj,ii],y_tang_1[jj,ii],1000))
    arc_dist_outer =  [dist_from_origin_max]*1000 + [dist_from_origin_max+100]*1000

    arc_ang_sq = np.append(np.linspace(y_tang_1[jj,ii],y_tang_2[jj,ii],1000),np.linspace(y_tang_2[jj,ii],y_tang_1[jj,ii],1000))
    arc_dist_sq =  [dist_from_origin_max]*1000 + [dist_from_origin_min]*1000

    arc_ang_gr = np.append(ang, ang[::-1])
    arc_dist_gr =  [dist_from_origin_max]*361 + [dist_from_origin_min]*361


    arc_ang_gr_excl_once = np.append(np.linspace(y_tang_2[ii,jj],0,100), np.linspace(0,2*np.pi+y_tang_1[ii,jj],100))
    arc_ang_gr_excl = np.append(arc_ang_gr_excl_once, arc_ang_gr_excl_once[::-1])
    arc_dist_gr_excl =  [dist_from_origin_max]*200 + [dist_from_origin_min]*200

    linelim = np.array([0,100])

    # Plot Appendix figure 5 of Tinker et al. 2022.

    fig2 = plt.figure()
    fig2.set_figheight(5.0)
    fig2.set_figwidth(15.0)
    fig2.set_figheight(8)
    fig2.set_figwidth(8.0)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
    ax = []
    ax.append(plt.subplot(1,1,1))
    ax[0].set_facecolor((0,0,0,0.25))
    plt.fill(arc_dist_outer*np.cos(arc_ang_outer),arc_dist_outer*np.sin(arc_ang_outer),'r', alpha = 0.3)
    plt.fill(arc_dist_inner*np.cos(arc_ang_inner),arc_dist_inner*np.sin(arc_ang_inner),'b', alpha = 0.3, zorder = 1)
    plt.fill(arc_dist_sq*np.cos(arc_ang_sq),arc_dist_sq*np.sin(arc_ang_sq),color = 'purple', alpha = 0.3)
    plt.fill(arc_dist_gr_excl*np.cos(arc_ang_gr_excl),arc_dist_gr_excl*np.sin(arc_ang_gr_excl),color = 'g', alpha = 0.3)
    plt.fill(Xelip,Yelip,color = 'w')

    plt.plot(U_mean[jj,ii],V_mean[jj,ii],'k+')
    plt.plot(Xelip,Yelip,'k', lw = 2)
    plt.axis('equal')
    for ai,tmpax in enumerate(ax): tmpax.axhline(0, color = '0.5')
    for ai,tmpax in enumerate(ax): tmpax.axvline(0, color = '0.5')
    for ai,tmpax in enumerate(ax): tmpax.set_xlim(xlim)
    for ai,tmpax in enumerate(ax): tmpax.set_ylim(ylim)


    return fig1, fig2

def example_ellipse_one_dist(dict_in,U_mat,V_mat,ii,jj,n_std=2.45):


    ang = np.pi*np.arange(-180.,181.)/180.

    Xelip = n_std*dict_in['X_elip_amp'][jj,ii] * np.sin(ang +dict_in['X_elip_phi'][jj,ii] ) + dict_in['U_mean'][jj,ii]
    Yelip = n_std*dict_in['Y_elip_amp'][jj,ii] * np.sin(ang +dict_in['Y_elip_phi'][jj,ii] ) + dict_in['V_mean'][jj,ii]

    inside_ellipse,tmp = point_inside_parameteric_ellipse(U_mat[:,jj,ii],V_mat[:,jj,ii], n_std,dict_in['foci_x_1'][jj,ii],dict_in['foci_y_1'][jj,ii],dict_in['foci_x_2'][jj,ii],dict_in['foci_y_2'][jj,ii], dict_in['qmax'][jj,ii])


    fig = plt.figure()
    fig.set_figheight(10.0)
    fig.set_figwidth(18.0)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)

    plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'b.')
    plt.axvline(dict_in['U_mean'][jj,ii], color = '0.75')
    plt.axhline(dict_in['V_mean'][jj,ii], color = '0.75')
    plt.axvline(0, color = 'k')
    plt.axhline(0, color = 'k')
    plt.plot(dict_in['U_mean'][jj,ii],dict_in['V_mean'][jj,ii],'r+')
    plt.plot(Xelip, Yelip)
    plt.plot(U_mat[:,jj,ii][inside_ellipse],V_mat[:,jj,ii][inside_ellipse],'bo')

    return fig

def example_ellipse_two_dist(dict_1,dict_2,U_mat_1,V_mat_1, U_mat_2,V_mat_2,ii,jj,n_std=2.45):

    ang = np.pi*np.arange(-180.,181.)/180.

    Xelip_1 = n_std*dict_1['X_elip_amp'][jj,ii] * np.sin(ang +dict_1['X_elip_phi'][jj,ii] ) + dict_1['U_mean'][jj,ii]
    Yelip_1 = n_std*dict_1['Y_elip_amp'][jj,ii] * np.sin(ang +dict_1['Y_elip_phi'][jj,ii] ) + dict_1['V_mean'][jj,ii]
    Xelip_2 = n_std*dict_2['X_elip_amp'][jj,ii] * np.sin(ang +dict_2['X_elip_phi'][jj,ii] ) + dict_2['U_mean'][jj,ii]
    Yelip_2 = n_std*dict_2['Y_elip_amp'][jj,ii] * np.sin(ang +dict_2['Y_elip_phi'][jj,ii] ) + dict_2['V_mean'][jj,ii]

    inside_ellipse_1,tmp = point_inside_parameteric_ellipse(U_mat_1[:,jj,ii],V_mat_1[:,jj,ii], n_std,dict_1['foci_x_1'][jj,ii],dict_1['foci_y_1'][jj,ii],dict_1['foci_x_2'][jj,ii],dict_1['foci_y_2'][jj,ii], dict_1['qmax'][jj,ii])
    inside_ellipse_2,tmp = point_inside_parameteric_ellipse(U_mat_2[:,jj,ii],V_mat_2[:,jj,ii], n_std,dict_2['foci_x_1'][jj,ii],dict_2['foci_y_1'][jj,ii],dict_2['foci_x_2'][jj,ii],dict_2['foci_y_2'][jj,ii], dict_2['qmax'][jj,ii])

    inside_ellipse_1_2,tmp = point_inside_parameteric_ellipse(U_mat_1[:,jj,ii],V_mat_1[:,jj,ii], n_std,dict_2['foci_x_1'][jj,ii],dict_2['foci_y_1'][jj,ii],dict_2['foci_x_2'][jj,ii],dict_2['foci_y_2'][jj,ii], dict_2['qmax'][jj,ii])

    inside_ellipse_2_1,tmp = point_inside_parameteric_ellipse(U_mat_2[:,jj,ii],V_mat_2[:,jj,ii], n_std,dict_1['foci_x_1'][jj,ii],dict_1['foci_y_1'][jj,ii],dict_1['foci_x_2'][jj,ii],dict_1['foci_y_2'][jj,ii], dict_1['qmax'][jj,ii])


    fig = plt.figure()
    fig.set_figheight(10.0)
    fig.set_figwidth(18.0)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)


    plt.axvline(0, color = 'k')
    plt.axhline(0, color = 'k')

    plt.plot(U_mat_1[:,jj,ii],V_mat_1[:,jj,ii],'b.')
    plt.axvline(dict_1['U_mean'][jj,ii], color = 'b', alpha = 0.25)
    plt.axhline(dict_1['V_mean'][jj,ii], color = 'b', alpha = 0.25)
    plt.plot(U_mat_1[:,jj,ii][inside_ellipse_1],V_mat_1[:,jj,ii][inside_ellipse_1],'bo')
    plt.plot(U_mat_1[:,jj,ii][inside_ellipse_1&inside_ellipse_1_2],V_mat_1[:,jj,ii][inside_ellipse_1&inside_ellipse_1_2],'r+')
    plt.plot(Xelip_1, Yelip_1,'b')

    plt.plot(U_mat_2[:,jj,ii],V_mat_2[:,jj,ii],'g.')
    plt.axvline(dict_2['U_mean'][jj,ii], color = 'g', alpha = 0.25)
    plt.axhline(dict_2['V_mean'][jj,ii], color = 'g', alpha = 0.25)
    plt.plot(U_mat_2[:,jj,ii][inside_ellipse_2],V_mat_2[:,jj,ii][inside_ellipse_2],'go')
    plt.plot(U_mat_2[:,jj,ii][inside_ellipse_2&inside_ellipse_2_1],V_mat_2[:,jj,ii][inside_ellipse_2&inside_ellipse_2_1],'r+')
    plt.plot(Xelip_2, Yelip_2,'g')

    return fig

def example_ellipse_one_dist_one_point(dict_in,U_mat,V_mat,ii_pnt, jj_pnt, ii,jj,n_std=2.45):


    ang = np.pi*np.arange(-180.,181.)/180.

    Xelip = n_std*dict_in['X_elip_amp'][jj,ii] * np.sin(ang +dict_in['X_elip_phi'][jj,ii] ) + dict_in['U_mean'][jj,ii]
    Yelip = n_std*dict_in['Y_elip_amp'][jj,ii] * np.sin(ang +dict_in['Y_elip_phi'][jj,ii] ) + dict_in['V_mean'][jj,ii]

    inside_ellipse,tmp = point_inside_parameteric_ellipse(U_mat[:,jj,ii],V_mat[:,jj,ii], n_std,dict_in['foci_x_1'][jj,ii],dict_in['foci_y_1'][jj,ii],dict_in['foci_x_2'][jj,ii],dict_in['foci_y_2'][jj,ii], dict_in['qmax'][jj,ii])


    fig = plt.figure()
    fig.set_figheight(10.0)
    fig.set_figwidth(18.0)
    plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)

    plt.plot(U_mat[:,jj,ii],V_mat[:,jj,ii],'b,')
    plt.axvline(dict_in['U_mean'][jj,ii], color = '0.75')
    plt.axhline(dict_in['V_mean'][jj,ii], color = '0.75')
    plt.axvline(0, color = 'k')
    plt.axhline(0, color = 'k')
    plt.plot(dict_in['U_mean'][jj,ii],dict_in['V_mean'][jj,ii],'r+')
    plt.plot(Xelip, Yelip)
    plt.plot(U_mat[:,jj,ii][inside_ellipse],V_mat[:,jj,ii][inside_ellipse],'b.')

    plt.plot(ii_pnt[jj,ii], jj_pnt[jj,ii],'ko')


    return fig

def example_analysis_one_dist(U_mat_1,V_mat_1,lat,lon,n_std=2.45, doplot=True):

    dict_1 = ellipse_params_add_to_dict(ellipse_params(U_mat_1,V_mat_1, n_std=n_std))

    if doplot:
        fig = plt.figure()
        fig.set_figheight(10.0)
        fig.set_figwidth(18.0)
        plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
        ax = []
        ax.append(plt.subplot(2,3,1))
        plt.pcolormesh(lon,lat,dict_1['qmax'])
        plt.title('qmax')
        plt.colorbar()
        ax.append(plt.subplot(2,3,2))
        plt.pcolormesh(lon,lat,dict_1['qmin'])
        plt.title('qmin')
        plt.colorbar()
        ax.append(plt.subplot(2,3,3))
        plt.pcolormesh(lon,lat,dict_1['ecc'])
        plt.title('Eccentricity')
        plt.colorbar()
        ax.append(plt.subplot(2,3,4))
        plt.pcolormesh(lon,lat,dict_1['theta_max'])
        plt.title('theta_max')
        plt.colorbar()
        ax.append(plt.subplot(2,3,5))
        plt.pcolormesh(lon,lat,dict_1['XY_zero_num_std_from_mean'], vmin = 0,vmax = 3)
        plt.title('No of Std Devs between mean and origin\n(XY_zero_num_std_from_mean)')
        plt.colorbar()
        ax.append(plt.subplot(2,3,6))
        plt.pcolormesh(lon,lat,dict_1['ang_wid'])
        plt.title('angular width of ellipse (ang_wid)')
        plt.colorbar()

    return dict_1

def example_analysis_two_dist(U_mat_1,V_mat_1, U_mat_2,V_mat_2,lat,lon,n_std=2.45):

    dict_1 = example_analysis_one_dist(U_mat_1,V_mat_1,lat,lon,n_std=2.45,doplot=False)
    dict_2 = ellipse_params_add_to_dict(ellipse_params(U_mat_2,V_mat_2, n_std=n_std))
    overlap_dict = overlapping_ellipse_area_from_dict(dict_1, dict_2)
    OVL_dict = ellipse_overlap_coefficient_pdf_from_dict(dict_1, dict_2)


    fig = plt.figure()
    fig.set_figheight(10.0)
    fig.set_figwidth(10.0)
    plt.subplots_adjust(top=0.90,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
    ax = []
    ax.append(plt.subplot(2,2,1))
    plt.pcolormesh(lon,lat,overlap_dict['perc_overlap'])
    plt.title('Overlap area as a percentage of total area\n(perc_overlap)')
    plt.colorbar()
    ax.append(plt.subplot(2,2,2))
    plt.pcolormesh(lon,lat,overlap_dict['perc_ratio_of_ellipse_area'], vmin = 50, vmax = 200)
    plt.title('Ratio of ellipse areas\n(perc_ratio_of_ellipse_area (equiv to perc_area_rat))')
    plt.colorbar()
    ax.append(plt.subplot(2,2,3))
    plt.pcolormesh(lon,lat,OVL_dict['OVL'], vmin = 0.2,vmax =1)
    plt.title('Overlap Coefficient (OVL)')
    plt.colorbar()
    ax.append(plt.subplot(2,2,4))
    plt.plot(OVL_dict['OVL'].ravel(),overlap_dict['perc_overlap'].ravel()/100,',')
    plt.axis([0,1,0,1])
    plt.title('Overlap Coefficient (OVL) vs Percentage Overlap')
    plt.xlabel('Overlap Coefficient (OVL)')
    plt.ylabel('Percentage Overlap')




def example_analysis_one_dist_one_value(U_mat_1,V_mat_1, ii_pnt,jj_pnt,lat, lon, land_sea_mask, n_std=2.45):
    dict_1 = example_analysis_one_dist(U_mat_1,V_mat_1,lat,lon,n_std=2.45,doplot=False)
    sig_change_mat,sig_change_mag_mat = charact_val_comp_dist(dict_1, ii_pnt,jj_pnt,n_std=2.45)
    plot_charact_val_comp_dist(sig_change_mat,sig_change_mag_mat, lat, lon, land_sea_mask )
    plt.show()









if __name__ == "__main__":
    main()
