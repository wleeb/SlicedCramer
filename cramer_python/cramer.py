import finufft
import wprin as wp
import numpy as np
import numpy.matlib as ml
import numpy.random as rnd
import scipy.linalg as la
from scipy.stats import norm
import time as tm
import matplotlib.pyplot as plt
import numpy.fft as dft
import ctypes as ct
import os

#
#
#

def main():
    wp.prini(13,1)


###    n=50
    n=228
    nangls=int((n+6)/2)
    nangls=102
    m=n+2

    wp.prinf('n=',n,1)
    wp.prinf('nangls=',nangls,1)
    wp.prinf('m=',m,1)

    test_fast_rotate(n,nangls,m)

###    exit()

    p=3.4
    n=50
    m=n
    nangls=2*n

    test_compare_slow(n,nangls,p,m)

    exit()


#
#
#

def test_fast_rotate(n,nangls,m):

    gs_rot = np.empty([4000,4000],dtype='float64')
    gs = np.empty([4000,4000],dtype='float64')
    fs = np.empty([4000,4000],dtype='float64')
    cen = np.empty(10,dtype='float64')

    wids = np.empty(100,dtype='float64')
    heis = np.empty(100,dtype='float64')
    cens = np.empty([100,2],dtype='float64')
#
    wids2 = np.empty(100,dtype='float64')
    heis2 = np.empty(100,dtype='float64')
    cens2 = np.empty([100,2],dtype='float64')

    cens2_rot = np.empty([100,2],dtype='float64')

    ts = np.empty(8000,dtype='float64')

    kfs = np.empty(8000,dtype='int32')
    kfs2 = np.empty(8000,dtype='int32')

    fzs = np.empty(80000,dtype='complex128')
    fhats = np.empty(80000,dtype='complex128')
    freqs = np.empty(80000,dtype='float64')
    freqs2 = np.empty(80000,dtype='float64')

    vnorms = np.empty(10000,dtype='float64')
    vnorms2 = np.empty(10000,dtype='float64')
    wdists = np.empty(10000,dtype='float64')
    wdists2 = np.empty(10000,dtype='float64')

    dists = np.empty(40000,dtype='float64')
    dists2 = np.empty(40000,dtype='float64')


    rng = rnd.default_rng(61)



#
#    interval radius and endpoints
#
    r = 8.40301
    r = 9.40301
    r = 10.40301
    r = 11.40301
    r = 12.40301


    wp.prin2('r=',r,1)


    wp.prinf('n=',n,1)


#
#    gaussian mixture parameters
#
    ngaus = 8
    cens[0:ngaus,0:2] = rng.normal(0,0.312,[ngaus,2])
    cens2[0:ngaus,0:2] = rng.normal(0,0.3102,[ngaus,2])


    krot = np.int(nangls/3)
    krot = 2
###    krot = 0
    theta = 2*np.pi * krot / nangls

    theta = np.pi * 1.404391

###    theta=np.pi/2

    wp.prin2('theta=',theta,1)


    rotcols(theta,cens2,ngaus,cens2_rot)

    wp.prinr2('cens2_rot=',cens2_rot,ngaus,2)
    wp.prinr2('cens2=',cens2,ngaus,2)
    

###    exit()


    wp.prinr2('cens=',cens,ngaus,2)
    wp.prinr2('cens2=',cens2,ngaus,2)
###    exit()

    wids[0:ngaus] = rng.uniform(.1,.3,ngaus)
    wids2[0:ngaus] = rng.uniform(.1,.3,ngaus)

    wp.prin2('wids=',wids,ngaus)


    heis[0:ngaus] = rng.uniform(1,3,ngaus)
    heis2[0:ngaus] = rng.uniform(1,3,ngaus)
    wp.prin2('heis=',heis,ngaus)

    hsum = 0.0
    for i in range(0,ngaus):
        hsum = hsum + heis[i]

    for i in range(0,ngaus):
        heis[i] = heis[i] / hsum


    hsum = 0.0
    for i in range(0,ngaus):
        hsum = hsum + heis[i]

    wp.prin2('hsum=',hsum,1)
    chk0 = 1-hsum
    wp.prin2('chk0=',chk0,1)


    hsum2 = 0.0
    for i in range(0,ngaus):
        hsum2 = hsum2 + heis2[i]

    for i in range(0,ngaus):
        heis2[i] = heis2[i] / hsum2


    hsum2 = 0.0
    for i in range(0,ngaus):
        hsum2 = hsum2 + heis2[i]

    wp.prin2('hsum2=',hsum2,1)
    chk0 = 1-hsum2
    wp.prin2('chk0=',chk0,1)




    dt = 2*r/n

#
#    grid on interval and gaussian mixture
#
    ts[0:n+1] = np.linspace(-r,r,n+1)


    evalmix2d_fast(ts,n,cens,wids,heis,ngaus,fs)
    evalmix2d_fast(ts,n,cens2,wids2,heis2,ngaus,gs)


    evalmix2d_fast(ts,n,cens2_rot,wids2,heis2,ngaus,gs_rot)


###    wp.prin2('fs=',fs,n)
###    wp.prin2('gs=',gs,n)


#
#    check 2D integrals are 1
#
###    wp.prin2('fs=',fs[:,0:n],12)
    fint = np.sum(fs[0:n,0:n]) * dt**2
    gint = np.sum(gs[0:n,0:n]) * dt**2

    wp.prin2('fint=',fint,1)
    wp.prin2('gint=',gint,1)

    chk0 = 1-fint
    wp.prin2('chk0, fint=',chk0,1)

    chk0 = 1-gint
    wp.prin2('chk0, gint=',chk0,1)



###    dist2 = slicedcramer_nufft(fs,gs,ts,n,r,p,nangls,vnorms,kfs,freqs)


    nrots=np.int(nangls + 1)

    nrots = 2*nangls
    nrots=np.int(nangls**2 / 2)
    wp.prinf('nrots=',nrots,1)

###    eucl_polar(gs,gs_rot,ts,n,r,nangls,cens,wids,heis,ngaus)

    dist = slicedeucl_rots(gs_rot,gs,ts,n,r,nangls,nrots,dists,dists2)

    dist_dumb = np.min(dists[0:nangls])

    wp.prin2('dist=',dist,1)
    wp.prin2('dist_dumb=',dist_dumb,1)


    wp.prin2('dists=',dists,nangls)
    wp.prin2('dists2=',dists2,nrots)

    cdist = slicedcramer_rots(gs_rot,gs,ts,n,r,nangls,nrots,dists,dists2)

    wp.prin2('dists=',dists,nangls)
    wp.prin2('dists2=',dists2,nrots)


    cdist_dumb = np.min(dists[0:nangls])

    wp.prin2('cdist=',cdist,1)
    wp.prin2('cdist_dumb=',cdist_dumb,1)


###    exit()


    wdist = slicedwasser_rots(gs_rot,gs,ts,n,r,nangls,m,nrots,dists,dists2)


    p=2
    wdist2 = slicedwasser_nufft(gs_rot,gs,ts,n,r,p,nangls,m,wdists2)

    wp.prin2('dists=',dists,nangls)
    wp.prin2('dists2=',dists2,nrots)


    wp.prin2('dist=',dists,1)
    wp.prin2('wdist2=',wdist2,1)

    chk0 = dists[0] - wdist2
    wp.prin2('chk0=',chk0,1)


    wp.prin2('wdists2=',wdists2,nangls)

    wp.prin2('wdist=',wdist,1)
    wp.prin2('cdist=',cdist,1)


    return()

#
#
#

def rotcols(angl,cols,m,cols2):

    cc = np.cos(angl)
    ss = np.sin(angl)

    for i in range(0,m):
        cols2[i,0] = cc*cols[i,0] - ss*cols[i,1]
        cols2[i,1] = ss*cols[i,0] + cc*cols[i,1]

    return()

#
#
#

def test_compare_slow(n,nangls,p,m):

    gs = np.empty([4000,4000],dtype='float64')
    fs = np.empty([4000,4000],dtype='float64')
    cen = np.empty(10,dtype='float64')

    wids = np.empty(100,dtype='float64')
    heis = np.empty(100,dtype='float64')
    cens = np.empty([100,10],dtype='float64')
#
    wids2 = np.empty(100,dtype='float64')
    heis2 = np.empty(100,dtype='float64')
    cens2 = np.empty([100,10],dtype='float64')

    ts = np.empty(8000,dtype='float64')

    kfs = np.empty(8000,dtype='int32')
    kfs2 = np.empty(8000,dtype='int32')

    fzs = np.empty(80000,dtype='complex128')
    fhats = np.empty(80000,dtype='complex128')
    freqs = np.empty(80000,dtype='float64')
    freqs2 = np.empty(80000,dtype='float64')

    vnorms = np.empty(10000,dtype='float64')
    vnorms2 = np.empty(10000,dtype='float64')
    wdists = np.empty(10000,dtype='float64')
    wdists2 = np.empty(10000,dtype='float64')


    rng = rnd.default_rng(61)



#
#    interval radius and endpoints
#
    r = 8.40301


    wp.prin2('r=',r,1)


    wp.prinf('n=',n,1)


#
#    gaussian mixture parameters
#
    ngaus = 4
    cens[0:ngaus,0:2] = rng.normal(0,1.512,[ngaus,2])
    cens2[0:ngaus,0:2] = rng.normal(0,0.3102,[ngaus,2])




    wp.prinr2('cens=',cens,ngaus,2)
###    exit()

    wids[0:ngaus] = rng.uniform(.1,.4,ngaus)
    wids2[0:ngaus] = rng.uniform(.1,.4,ngaus)

    wp.prin2('wids=',wids,ngaus)


    heis[0:ngaus] = rng.uniform(1,3,ngaus)
    heis2[0:ngaus] = rng.uniform(1,3,ngaus)
    wp.prin2('heis=',heis,ngaus)

    hsum = 0.0
    for i in range(0,ngaus):
        hsum = hsum + heis[i]

    for i in range(0,ngaus):
        heis[i] = heis[i] / hsum


    hsum = 0.0
    for i in range(0,ngaus):
        hsum = hsum + heis[i]

    wp.prin2('hsum=',hsum,1)
    chk0 = 1-hsum
    wp.prin2('chk0=',chk0,1)



    hsum2 = 0.0
    for i in range(0,ngaus):
        hsum2 = hsum2 + heis2[i]

    for i in range(0,ngaus):
        heis2[i] = heis2[i] / hsum2


    hsum2 = 0.0
    for i in range(0,ngaus):
        hsum2 = hsum2 + heis2[i]

    wp.prin2('hsum2=',hsum2,1)
    chk0 = 1-hsum2
    wp.prin2('chk0=',chk0,1)




    cen[0] = 0.40301
    cen[1] = -1.9891
    wid = 1.10301
    dt = 2*r/n

#
#    grid on interval and gaussian mixture
#
    ts[0:n+1] = np.linspace(-r,r,n+1)


    evalmix2d_fast(ts,n,cens,wids,heis,ngaus,fs)
    evalmix2d_fast(ts,n,cens2,wids2,heis2,ngaus,gs)



#
#    check 2D integrals are 1
#
###    wp.prin2('fs=',fs[:,0:n],12)
    fint = np.sum(fs[0:n,0:n]) * dt**2
    gint = np.sum(gs[0:n,0:n]) * dt**2

    wp.prin2('fint=',fint,1)
    wp.prin2('gint=',gint,1)

    chk0 = 1-fint
    wp.prin2('chk0, hsum=',chk0,1)

    chk0 = 1-gint
    wp.prin2('chk0, hsum=',chk0,1)


###    exit()

###    gs[0:n,0:n] = 0


#
#    evaluate sliced Cramer fast and slow ways and compare
#
    wp.prini(0,0)
    slnorm = slicedcramer_nufft(fs,gs,ts,n,r,p,nangls,vnorms)
    slnorm2 = slicedcramer_dumb(fs,gs,ts,n,r,p,nangls,vnorms2)


    wp.prini(13,0)


    wp.prin2('slnorm=',slnorm,1)
    wp.prin2('slnorm2=',slnorm2,1)

    chk0_cramer = slnorm-slnorm2
    wp.prin2('chk0=',chk0_cramer,1)


###    exit()

    chk0 = lpdist_fast(vnorms,vnorms2,nangls,2,1)
    wp.prin2('chk0=',chk0,1)


#
#    evaluate sliced Wasserstein fast and slow ways and compare
#
    wp.prini(0,0)


    wdist2 = slicedwasser_nufft(fs,gs,ts,n,r,p,nangls,m,wdists2)
    wdist = slicedwasser_dumb(fs,gs,ts,n,r,p,nangls,m,wdists)

    wp.prini(13,0)


    wp.prin2('wdists=',wdists,nangls)
    wp.prin2('wdists2=',wdists2,nangls)

    wp.prin2('wdist=',wdist,1)
    wp.prin2('wdist2=',wdist2,1)


    chk0_wass = wdist - wdist2
    wp.prin2('chk0, Wasserstein=',chk0_wass,1)

    exit()

    err_norms = 0
    err_freqs = 0
    for i in range(0,nangls):
        err_norms = err_norms + np.abs(wdists[i] - wdists2[i]) / nangls
        err_freqs = err_freqs + np.abs(freqs[i] - freqs2[i]) / nangls


    wp.prin2('chk0, freqs=',err_freqs,1)


    wp.prin2('chk0, Wasserstein projections=',err_norms,1)



    wp.prin2('slnorm=',slnorm,1)
    wp.prin2('slnorm2=',slnorm2,1)

    chk0_cramer = slnorm - slnorm2

    wp.prin2('chk0, Cramer=',chk0_cramer,1)
    wp.prin2('chk0, Wasserstein=',chk0_wass,1)


    return()

def wasser_fast(fvals,gvals,tgrid,m,n,a,b,p,fints,gints,finvs,ginvs,difs):

    ivals_f = np.empty(m+10,dtype='int32')
    ivals_g = np.empty(m+10,dtype='int32')
    yedges=np.empty(m+10,dtype='float64')
    ys=np.empty(m+10,dtype='float64')


    wp.prinf('n=',n,1)
    wp.prinf('m=',m,1)


    wp.prin2('tgrid=',tgrid,n+1)

    wp.prin2('a=',a,1)
    wp.prin2('b=',b,1)

    hh=(b-a)/n

    wp.prin2('hh=',hh,1)


#
#    partial integrals on grid
#
    fl = np.cumsum(fvals[0:n+1])*hh
    fr = np.cumsum(fvals[1:n+1])*hh

    fints[0] = 0
    fints[1:n] = (fl[0:n-1] + fr[0:n-1]) / 2
    fints[n]=1



    gl = np.cumsum(gvals[0:n+1])*hh
    gr = np.cumsum(gvals[1:n+1])*hh

    gints[0] = 0
    gints[1:n] = (gl[0:n-1] + gr[0:n-1]) / 2
    gints[n]=1

    wp.prin2('fints=',fints,n+1)
    wp.prin2('gints=',gints,n+1)


    ymin = 0.0
    ymax = 1.0

    wp.prin2('ymin=',ymin,1)
    wp.prin2('ymax=',ymax,1)

#
#    edges between 0 and 1 -- includes endpoints
#
    yedges=np.linspace(ymin,ymax,m+1)

    dy = yedges[1] - yedges[0]

    wp.prin2('yedges=',yedges,m+1)
    wp.prin2('dy=',dy,1)

#
#    midpoint grid between 0 and 1
#
    ys[0:m] = (yedges[0:m] + yedges[1:m+1])/2

    wp.prin2('ys=',ys,m)

    dy2 = ys[1] - ys[0]
    wp.prin2('dy2=',dy2,1)
    chk0 = dy2 - dy
    wp.prin2('chk0=',chk0,1)


#
#    inverse CDFs over [0,1]
#

    cdfinvs_c(ys,m,n,finvs,fints,tgrid,ivals_f)
###    cdfinvs_march(ys,m,a,b,n,finvs,fints,tgrid,ivals_f)



    cdfinvs_c(ys,m,n,ginvs,gints,tgrid,ivals_g)
###    cdfinvs_march(ys,m,a,b,n,ginvs,gints,tgrid,ivals_g)



#
#    distance between inverse CDFs, by midpoint rule
#

    difs[0:m] = finvs[0:m] - ginvs[0:m]


    if (p<=0 or p==np.inf):
        wdist = np.max(np.abs(difs[0:m]))
        return(wdist)

    wdist = la.norm(difs[0:m],ord=p)
    wdist = wdist * dy**(1/p)


###    wp.prini(13,0)
    wp.prin2('difs=',difs,m)


    wp.prin2('wdist, inside=',wdist,1)

###    exit()



    return(wdist)

#
#
#
####################################################################################
#
#            This is the end of the test code and the beginning of the
#            sliced Cramer and sliced Wasserstein code proper.
#
####################################################################################
#
#
#            This file contains the following user-callable functions:
#
#  slicedcramer_nufft - evaluates the 2D sliced p-Cramer distance, using the
#        FINUFFT to evaluate the Radon transform and vectorization throughout.
#  slicedwasser_nufft - evaluates the 2D sliced p-Wasserstein distance, using the
#        FINUFFT to evaluate the Radon transform and vectorization throughout.
#  slicedlebesg_nufft - evaluates the 2D sliced p-Lebesgue distance, using the
#        FINUFFT to evaluate the Radon transform and vectorization throughout.
#  slicedcramer_rots - evaluates the 2D sliced 2-Cramer distances between all
#        rotations of the input functions.
#  slicedwasser_rots - evaluates the 2D sliced 2-Wasserstein distances between
#        all rotations of the input functions.
#  slicedeucl_rots - evaluates the 2D sliced Euclidean distances between all
#        rotations of the input functions.
#
#
#            These additional functions, among others, may also be useful:
#
#  slicedlebesg_dumb - evaluates the 2D sliced p-Lebesgue distance, without the
#        FINUFFT or vectorization. It is a slower but more transparent version of
#        slicedlebesg_nufft.
#  slicedwasser_dumb - evaluates the 2D sliced Wasserstein distance, without the
#        FINUFFT or vectorization. It is a slower but more transparent version of
#        slicedwasser_nufft.
#  slicedcramer_dumb - evaluates the 2D sliced Cramer distance, without the
#        FINUFFT or vectorization. It is a slower but more transparent version of
#        slicedcramer_nufft.
#  slicedcramer_ps - evaluates the 2D sliced p-Cramer distances for multiple
#        values of p.
#  slicedwasser_ps - evaluates the 2D sliced p-Wasserstein distances for multiple
#        values of p.
#  vnorm_fourier_fast - evaluates the 1D Volterra norm from samples of the
#        Fourier transform; vectorized code that uses the Numpy FFT.
#  cramer1d - evaluates the 1D Cramer distance from equispaced samples.
#  vnorm_fast - evaluates the 1D Volterra norm from samples of a function (not its
#        Fourier transform); vectorized code that uses the Numpy FFT.
#  evalmix2d_fast - evaluates 2D Gaussian mixture (or the tomographic projection
#        of a 3D mixture) on a 2D grid. Fully vectorized.
#
#
####################################################################################
#
#
#

def slicedcramer_rots(gs,hs,ts,n,r,nangls,nrots,dists,dists2):
#
#                            description:
#
#    This code evaluates the sliced 2-Cramer distances between a function
#    h and all rotations of a function g. Both h and g are evaluated on on a
#    2D grid of equispaced points. The distances are evaluated both ``on-grid''
#    and ``off-grid'', the sizes of which are determined by the user.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  nangls - the number of projection angles used to evaluate the sliced distances;
#    suggest setting it to O(n)
#  nrots - the number of rotations for which distances are computed; must be
#    greater than or equal to nangls, and can be as large as O(n**2)
#
#
#                        output parameters:
#
#  dists - nangls-length array containing the distances between h and
#    R_phi(g), for angles phi on the same grid used to evaluate the distances
#  dists2 - nrots-length array containing the distances between h and
#    R_phi(g), for angles phi on the extended grid
#
#
#                    return parameters:
#
#  dist_min - the minimum sliced 2-Cramer distance on the extended grid, i.e.
#    the minimum of dists2
#
#
#

    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')

    prods = np.empty(nangls,dtype='float64')
    prods2 = np.empty(nrots,dtype='float64')
###    dists = np.empty(nangls,dtype='float64')
###    dists2 = np.empty(nangls,dtype='float64')

    angls = np.empty(nangls+1,dtype='float64')

    difs_arr = np.empty([nangls,n],dtype='float64')
#
    gprojs_arr = np.empty([nangls,n],dtype='float64')
    ghats_arr = np.empty([nangls,n],dtype='complex128')
    gphats_arr = np.empty([nangls,n],dtype='complex128')
    gps_arr = np.empty([nangls,n],dtype='float64')
#
    hprojs_arr = np.empty([nangls,n],dtype='float64')
    hhats_arr = np.empty([nangls,n],dtype='complex128')
    hphats_arr = np.empty([nangls,n],dtype='complex128')
    hps_arr = np.empty([nangls,n],dtype='float64')

    hps_rot = np.empty([nangls,n],dtype='float64')

    hhats_rot = np.empty([nangls,n],dtype='complex128')

###    fprojs_arr[0:nangls,0:n] = 10000

    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    

    gs_comp = np.empty([n,n],dtype='complex128')
    hs_comp = np.empty([n,n],dtype='complex128')

    gczs = np.empty(n*nangls,dtype='complex128')
    hczs = np.empty(n*nangls,dtype='complex128')



    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)


    rq = n/(4*r)
###    wp.prin2('rq=',rq,1)

#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,2*rpi,nangls+1)


    wp.prin2('angls=',angls,nangls+1)

###    exit()


    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])
#
    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)

#
#    compute the projections with the FINUFFT
#
    gs_comp[0:n,0:n] = gs[0:n,0:n]
    finufft.nufft2d2(xs, ys, gs_comp, gczs, eps=1e-15, isign=-1)
    gczs = gczs / (2*rq)**2
    ghats_arr[0:nangls,0:n] = np.reshape(gczs[0:nangls*n],[nangls,n])

    hs_comp[0:n,0:n] = hs[0:n,0:n]
    finufft.nufft2d2(xs, ys, hs_comp, hczs, eps=1e-15, isign=-1)
    hczs = hczs / (2*rq)**2
    hhats_arr[0:nangls,0:n] = np.reshape(hczs[0:nangls*n],[nangls,n])



#
#    evaluate the volterra transforms
#
    voltr_fourier_fast(ghats_arr,ts,nangls,n,-r,r,\
        gps_arr,gphats_arr,kfs,freqs)

    voltr_fourier_fast(hhats_arr,ts,nangls,n,-r,r,\
        hps_arr,hphats_arr,kfs,freqs)


    nh2 = np.int(n/2)

    wp.prinr2('gps_arr=',gps_arr[:,nh2:],10,6)
    wp.prinr2('hps_arr=',hps_arr[:,nh2:],10,6)

    distalign_core(gps_arr,hps_arr,nangls,n,nrots,r,dists,prods2,dists2)


    wp.prin2('dists=',dists,nangls)
    wp.prin2('dists2=',dists2,nrots)


###    exit()

    dist_min = np.min(dists2[0:nrots])


    return(dist_min)

#
#
#

def slicedeucl_rots(gs,hs,ts,n,r,nangls,nrots,dists,dists2):
#
#                            description:
#
#    This code evaluates the sliced Euclidean distances between a function
#    h and all rotations of a function g. Both h and g are evaluated on on a
#    2D grid of equispaced points. The distances are evaluated both ``on-grid''
#    and ``off-grid'', the sizes of which are determined by the user.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  nangls - the number of projection angles used to evaluate the sliced distances;
#    suggest setting it to O(n)
#  nrots - the number of rotations for which distances are computed; must be
#    greater than or equal to nangls, and can be as large as O(n**2)
#
#
#                        output parameters:
#
#  dists - nangls-length array containing the distances between h and
#    R_phi(g), for angles phi on the same grid used to evaluate the distances
#  dists2 - nrots-length array containing the distances between h and
#    R_phi(g), for angles phi on the extended grid
#
#
#                    return parameters:
#
#  dist_min - the minimum sliced Euclidean distance on the extended grid, i.e.
#    the minimum of dists2
#
#
#

    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')

    prods = np.empty(nangls,dtype='float64')
    prods2 = np.empty(nrots,dtype='float64')

    angls = np.empty(nangls+1,dtype='float64')

    difs_arr = np.empty([nangls,n],dtype='float64')
#
    gprojs_arr = np.empty([nangls,n],dtype='float64')
    ghats_arr = np.empty([nangls,n],dtype='complex128')
    gphats_arr = np.empty([nangls,n],dtype='complex128')
    gps_arr = np.empty([nangls,n],dtype='float64')
#
    hprojs_arr = np.empty([nangls,n],dtype='float64')
    hhats_arr = np.empty([nangls,n],dtype='complex128')
    hphats_arr = np.empty([nangls,n],dtype='complex128')
    hps_arr = np.empty([nangls,n],dtype='float64')

###    fprojs_arr[0:nangls,0:n] = 10000

    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    

    gs_comp = np.empty([n,n],dtype='complex128')
    hs_comp = np.empty([n,n],dtype='complex128')

    gczs = np.empty(n*nangls,dtype='complex128')
    hczs = np.empty(n*nangls,dtype='complex128')



    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)


    rq = n/(4*r)
###    wp.prin2('rq=',rq,1)

#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,2*rpi,nangls+1)


    wp.prin2('angls=',angls,nangls+1)

###    exit()


    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])
#
    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)

#
#    compute the projections with the FINUFFT
#
    gs_comp[0:n,0:n] = gs[0:n,0:n]
    finufft.nufft2d2(xs, ys, gs_comp, gczs, eps=1e-15, isign=-1)
    gczs = gczs / (2*rq)**2
    ghats_arr[0:nangls,0:n] = np.reshape(gczs[0:nangls*n],[nangls,n])

    hs_comp[0:n,0:n] = hs[0:n,0:n]
    finufft.nufft2d2(xs, ys, hs_comp, hczs, eps=1e-15, isign=-1)
    hczs = hczs / (2*rq)**2
    hhats_arr[0:nangls,0:n] = np.reshape(hczs[0:nangls*n],[nangls,n])




#
#    projections in real space
#
    iftrs_fast(ghats_arr,freqs,ts,gprojs_arr,nangls,n)
    iftrs_fast(hhats_arr,freqs,ts,hprojs_arr,nangls,n)


#
#    all distances
#
    distalign_core(gprojs_arr,hprojs_arr,nangls,n,nrots,r,dists,prods2,dists2)


    wp.prin2('dists=',dists,nangls)
    wp.prin2('dists2=',dists2,nrots)


    dist_min = np.min(dists2[0:nrots])


    return(dist_min)

#
#
#

def slicedwasser_rots(gs,hs,ts,n,r,nangls,m,nrots,dists,dists2):
#
#                            description:
#
#    This code evaluates the sliced 2-Wasserstein distances between a function
#    h and all rotations of a function g. Both h and g are evaluated on on a
#    2D grid of equispaced points. The distances are evaluated both ``on-grid''
#    and ``off-grid'', the sizes of which are determined by the user.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  nangls - the number of projection angles used to evaluate the sliced distances;
#    suggest setting it to O(n)
#  m - the number of points used to discretized the inverse CDFs
#  nrots - the number of rotations for which distances are computed; must be
#    greater than or equal to nangls, and can be as large as O(n**2)
#
#
#                        output parameters:
#
#  dists - nangls-length array containing the distances between h and
#    R_phi(g), for angles phi on the same grid used to evaluate the distances
#  dists2 - nrots-length array containing the distances between h and
#    R_phi(g), for angles phi on the extended grid
#
#
#                    return parameters:
#
#  dist_min - the minimum sliced 2-Wasserstein distance on the extended grid,
#    i.e. the minimum of dists2
#
#

    prods = np.empty(nangls,dtype='float64')
    prods2 = np.empty(nrots,dtype='float64')

    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')
#
    angls = np.empty(nangls+1,dtype='float64')
#
    hprojs_arr = np.empty([nangls,n],dtype='float64')
    hhats_arr = np.empty([nangls,n],dtype='complex128')
    hphats_arr = np.empty([nangls,n],dtype='complex128')
    hps_arr = np.empty([nangls,n+1],dtype='float64')
    hinvs_arr = np.empty([nangls,m+1],dtype='float64')
#
    gprojs_arr = np.empty([nangls,n],dtype='float64')
    ghats_arr = np.empty([nangls,n],dtype='complex128')
    gphats_arr = np.empty([nangls,n],dtype='complex128')
    gps_arr = np.empty([nangls,n+1],dtype='float64')
    ginvs_arr = np.empty([nangls,m+1],dtype='float64')
#

    hczs = np.empty(n*nangls,dtype='complex128')
    gczs = np.empty(n*nangls,dtype='complex128')

    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    
    hs_comp = np.empty([n,n],dtype='complex128')
    gs_comp = np.empty([n,n],dtype='complex128')

    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)

###    wp.prinr2('fs=',fs,n,6)


    p=2

    rq = n/(4*r)


#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

###    wp.prinf('kfs=',kfs,n)
###    wp.prin2('freqs=',freqs,n+1)
    wp.prin2('rq=',rq,1)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,2*rpi,nangls+1)

###    dangl = 2*rpi / nangls


###    wp.prin2('angls=',angls,nangls+1)
###    wp.prin2('dangl=',dangl,1)



    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])


    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)



#
#    projections of h
#
    hs_comp[0:n,0:n] = hs[0:n,0:n]
    finufft.nufft2d2(xs, ys, hs_comp, hczs, eps=1e-15, isign=-1)
    hczs = hczs / (2*rq)**2
    hhats_arr[0:nangls,0:n] = np.reshape(hczs[0:nangls*n],[nangls,n])
    iftrs_fast(hhats_arr,freqs,ts,hprojs_arr,nangls,n)


#
#    projections of g
#
    gs_comp[0:n,0:n] = gs[0:n,0:n]
    finufft.nufft2d2(xs, ys, gs_comp, gczs, eps=1e-15, isign=-1)
    gczs = gczs / (2*rq)**2
    ghats_arr[0:nangls,0:n] = np.reshape(gczs[0:nangls*n],[nangls,n])
    iftrs_fast(ghats_arr,freqs,ts,gprojs_arr,nangls,n)


    wp.prinz('hhats_arr=',hhats_arr,n)
    wp.prinz('ghats_arr=',ghats_arr,n)


    wp.prinf('nangls=',nangls,1)
    wp.prinf('n=',n,1)

###    exit()

    wp.prin2('hprojs_arr=',hprojs_arr,n)
    wp.prin2('gprojs_arr=',gprojs_arr,30)


#
#    compute all wasserstein distances
#

    invcdfs_fast(hprojs_arr,gprojs_arr,ts,m,n,-r,r, \
        hps_arr,gps_arr,hinvs_arr,ginvs_arr,nangls)

    dhalf=1/2
    distalign_core(ginvs_arr,hinvs_arr,nangls,m,nrots,dhalf,dists,prods2,dists2)


###    dist_min = np.min(dists[0:nangls])
    dist_min = np.min(dists2[0:nrots])


    return(dist_min)

#
#
#

def voltr_fourier_fast(fzs,ts,m,n,a,b,fps,fpzs,kfs,freqs):
#
#    vectorized code for computing the Volterra transforms of m functions,
#    taking as inputs n values of their Fourier transforms on an
#    equispaced frequency grid.
#

    zzzs = np.empty(n,dtype='complex128')

    nh2 = np.int(n/2)

#
#    phase factors
#
    zzzs[0:n] = np.exp(2*np.pi*1j*freqs[0:n]*a)


#
#    fourier coefficients of partial integral
#
    fpzs[0:m,1:n] = np.divide(fzs[0:m,1:n],freqs[1:n])
    fpzs[0:m,1:n] = fpzs[0:m,1:n] / (2*np.pi*1j)
    fpzs[0:m,nh2] = 0
#
    fpzs[0:m,0] = -np.sum(np.multiply(fpzs[0:m,1:nh2],zzzs[1:nh2]),1)
    fpzs[0:m,0] = fpzs[0:m,0]-np.sum(np.multiply(fpzs[0:m,nh2+1:n],zzzs[nh2+1:n]),1)


###    fpzs[0:m,0] = 0


#
#    evaluate partial integrals and find norms
#
    iftrs_fast(fpzs,freqs,ts,fps,m,n)


    return()

#
#
#

def invcdfs_fast(hs,gs,ts,m,n,a,b,hps,gps,hinvs,ginvs,k):

    ivals_h = np.empty([k,m],dtype='int32')
    ivals_g = np.empty([k,m],dtype='int32')
    yedges=np.empty(m+1,dtype='float64')
    ys=np.empty(m+1,dtype='float64')

    hh=(b-a)/n

    hl = np.cumsum(hs[0:k,0:n+1],axis=1)*hh
    hr = np.cumsum(hs[0:k,1:n+1],axis=1)*hh

    hps[0:k,0] = 0
    hps[0:k,1:n] = (hl[0:k,0:n-1] + hr[0:k,0:n-1]) / 2
    hps[0:k,n]=1



    gl = np.cumsum(gs[0:k,0:n+1],axis=1)*hh
    gr = np.cumsum(gs[0:k,1:n+1],axis=1)*hh

    gps[0:k,0] = 0
    gps[0:k,1:n] = (gl[0:k,0:n-1] + gr[0:k,0:n-1]) / 2
    gps[0:k,n]=1


    ymin = 0.0
    ymax = 1.0

    yedges=np.linspace(ymin,ymax,m+1)

    dy = yedges[1] - yedges[0]
    ys[0:m] = (yedges[0:m] + yedges[1:m+1])/2


#
#    distance between inverse CDFs, by midpoint rule
#
    cdfinvs_all_c(ts,m,n,hps,hinvs,ivals_h,ys,k)
    cdfinvs_all_c(ts,m,n,gps,ginvs,ivals_g,ys,k)


    return()

#
#
#

def distalign_core(gps_arr,hps_arr,nangls,n,nrots,r,dists,prods2,dists2):
#
#    evaluates all inner products and distances between rotated functions,
#    with values on a polar grid; inner products/distances are evaluated on
#    a finer grid of angles
#
    gf_rings = np.empty([nangls,n],dtype='complex128')
    hf_rings = np.empty([nangls,n],dtype='complex128')
    pf_rings = np.empty([nangls,n],dtype='complex128')

    p_rings = np.empty([nangls,n],dtype='float64')

    prods = np.empty(nangls,dtype='float64')

    p2_rings = np.empty([nrots,n],dtype='float64')
    pf2_rings = np.empty([nrots,n],dtype='complex128')



    dt = 2*r / n
    da = 1/nangls

    gps_normsq = np.sum(gps_arr[0:nangls,0:n]**2) * da * dt
    hps_normsq = np.sum(hps_arr[0:nangls,0:n]**2) * da * dt


    gf_rings[0:nangls,0:n] = dft.fft(gps_arr[0:nangls,0:n],axis=0) * da
    hf_rings[0:nangls,0:n] = dft.fft(hps_arr[0:nangls,0:n],axis=0) * da


    pf_rings[0:nangls,0:n] = gf_rings[0:nangls,0:n] * np.conj(hf_rings[0:nangls,0:n])
###    p_rings[0:nangls,jj] = np.real(dft.ifft(pf_rings[0:nangls,jj])) / da
    p_rings[0:nangls,0:n] = np.real(dft.ifft(pf_rings[0:nangls,0:n],axis=0)) / da
    prods[0:nangls] = np.sum(p_rings[0:nangls,0:n],axis=1) * dt


    dists[0:nangls] = gps_normsq + hps_normsq - 2*prods[0:nangls]
    dists[0:nangls] = np.sqrt(np.abs(dists[0:nangls]))



    nh2 = np.int(n/2)

#
#    zero-pad fourier coefficients of product
#
    wp.prinf('n=',n,1)
    wp.prinf('nh2=',nh2,1)


###    pf2_rings[0:nrots,0:n] = 88888888


    nah2 = np.int(nangls/2)


    wp.prinz('pf_rings, halfway - 1=',pf_rings[nah2-1,nh2],1)
    wp.prinz('pf_rings, halfway + 1=',pf_rings[nah2,nh2],1)


    pf2_rings[0:nah2,0:n] = pf_rings[0:nah2,0:n]
    pf2_rings[nrots-nah2:nrots,0:n] = pf_rings[nah2:nangls,0:n]
    pf2_rings[nah2:nrots-nah2,0:n] = np.zeros([nrots-nangls,n])



    wp.prinz('pf_rings=',pf_rings[:,nh2],nangls)
    wp.prinz('pf2_rings=',pf2_rings[:,nh2],nrots)


#
#    invert and find distances
#
    da2  = 1/nrots
    p2_rings[0:nrots,0:n] = np.real(dft.ifft(pf2_rings[0:nrots,0:n],axis=0)) / da2
    prods2 = np.sum(p2_rings[0:nrots,0:n],axis=1) * dt

    dists2[0:nrots] = gps_normsq + hps_normsq - 2*prods2[0:nrots]
    dists2[0:nrots] = np.sqrt(np.abs(dists2[0:nrots]))




    return()

#
#
#

def slicedcramer_nufft(gs,hs,ts,n,r,p,nangls,vnorms):
#
#                            description:
#
#    This code evaluates the sliced Cramer distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    vectorized and uses the FINUFFT to evaluate the projections; a slower,
#    dumber, but more readable code, whose output should be identical
#    up to round-off, is slicedcramer_dumb.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#
#
#                        output parameters:
#
#  vnorms - nangls-length array containing the distances between each pair
#    of projections
#
#
#                    return parameters:
#
#  dist - the sliced Cramer distance
#
#

    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')
    angls = np.empty(nangls+1,dtype='float64')

    fprojs_arr = np.empty([nangls,n],dtype='float64')
    fhats_arr = np.empty([nangls,n],dtype='complex128')
    fphats_arr = np.empty([nangls,n],dtype='complex128')
    fps_arr = np.empty([nangls,n],dtype='float64')

###    fprojs_arr[0:nangls,0:n] = 10000


    fs = np.empty([n,n],dtype='float64')


    czs = np.empty(n*nangls,dtype='complex128')
    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    
    fs_comp = np.empty([n,n],dtype='complex128')


    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)



    fs[0:n,0:n] = hs[0:n,0:n] - gs[0:n,0:n]

    rq = n/(4*r)
###    wp.prin2('rq=',rq,1)

#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,rpi,nangls+1)


    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])
#
    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)

#
#    compute the projections with the FINUFFT
#
    fs_comp[0:n,0:n] = fs[0:n,0:n]
    finufft.nufft2d2(xs, ys, fs_comp, czs, eps=1e-15, isign=-1)
    czs = czs / (2*rq)**2

###    wp.prinz('czs=',czs,n*nangls)

    fhats_arr[0:nangls,0:n] = np.reshape(czs[0:nangls*n],[nangls,n])


#
#    evaluate the sliced norm
#
    vnorms_fourier_fast(fhats_arr,ts,nangls,n,-r,r,p,\
        fps_arr,fphats_arr,kfs,freqs,vnorms)

    dww = 1/nangls
    dist = lpnorm_fast(vnorms,nangls,p,dww)


    return(dist)

#
#
#

def vnorms_fourier_fast(fzs,ts,m,n,a,b,p,fps,fpzs,kfs,freqs,vnorms):
#
#    vectorized code for computing the Volterra p-norms of m functions,
#    taking as inputs their Fourier transforms on a frequency grid.
#

    zzzs = np.empty(n,dtype='complex128')

    nh2 = np.int(n/2)

#
#    phase factors
#
    zzzs[0:n] = np.exp(2*np.pi*1j*freqs[0:n]*a)


#
#    fourier coefficients of partial integral
#
    fpzs[0:m,1:n] = np.divide(fzs[0:m,1:n],freqs[1:n])
    fpzs[0:m,1:n] = fpzs[0:m,1:n] / (2*np.pi*1j)
    fpzs[0:m,nh2] = 0
#
    fpzs[0:m,0] = -np.sum(np.multiply(fpzs[0:m,1:nh2],zzzs[1:nh2]),1)
    fpzs[0:m,0] = fpzs[0:m,0]-np.sum(np.multiply(fpzs[0:m,nh2+1:n],zzzs[nh2+1:n]),1)


#
#    evaluate partial integrals and find norms
#
    iftrs_fast(fpzs,freqs,ts,fps,m,n)

    fpmaxs = np.max(np.abs(fps[0:m,0:n]),axis=1)

    if (p <= 0 or p == np.inf):
        vnorms[0:m] = fpmaxs[0:m]
        return()

    dt = (b-a) / n
    vnorms[0:m] = la.norm(fps[0:m,0:n],ord=p,axis=1)* dt**(1/p)



    return()

#
#
#

def iftrs_fast(fzs,freqs,ts,fs,m,n):
#
#    evaluates m functions on a grid of length n,
#    given samples of their Fourier transforms
#
    dpi=np.pi
    zi = 1j

    df = freqs[1] - freqs[0]

#
#    phase correction
#
    zzzs = np.exp(2*dpi*zi*freqs[0:n]*ts[0])


#
#    inverse DFT, rescaled
#
    gggs = np.multiply(fzs[0:m,0:n],zzzs[0:n])
    fs[0:m,0:n] = np.real(dft.ifft(gggs[0:m,0:n]))  * df * n



    return()


###    zzzs[0:n] = 1

    for i in range(0,m):
        gg = np.multiply(fzs[i,0:n],zzzs[0:n])
        fs[i,0:n] = np.real(dft.ifft(gg))  * df * n


    return()

#
#
#

def vnorm_fourier_fast(fzs,ts,n,a,b,p,fps,fpzs,kfs,freqs):
#
#                            description:
#
#    This code evaluates the mean-centered Volterra p-norm using the
#    DFT and spectral integration.
#
#
#
#                        input parameters:
#
#  fzs - n-length complex array containing values of the Fourier transform
#    of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#  n - the number of samples; assumed to be an even integer (for now)
#  a,b - the endpoints of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#
#
#                        output parameters:
#
#  fps - n-length real array containing samples of the Volterra transform,
#    evaluated by spectral integration
#  fpzs - n-length complex array containing the Fourier coefficients
#    of f's Volterra transform
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#
#
#                    return parameters:
#
#  vnorm - the Volterra p-norm
#
#

    zzzs = np.empty(n,dtype='complex128')

    nh2 = np.int(n/2)



#
#    spectral integration
#

    fpzs[1:n] = np.divide(fzs[1:n],freqs[1:n])
    fpzs[1:n] = fpzs[1:n] / (2*np.pi*1j)

    fpzs[nh2] = 0

    wp.prinz('fzs=',fzs,6)


#
#    zero-th frequency, excluding frequencies n/2 and -n/2
#
    zzzs[0:n] = np.exp(2*np.pi*1j*freqs[0:n]*a)

    fpzs[0] = -np.sum(np.multiply(fpzs[1:nh2],zzzs[1:nh2]))
    fpzs[0] = fpzs[0]-np.sum(np.multiply(fpzs[nh2+1:n],zzzs[nh2+1:n]))

    wp.prinz('fpzs=',fpzs,2)




#
#    evaluate Vf and find p-norm
#
    cz0=iftr_fast(fpzs,freqs,ts,fps,n)
###    wp.prin2('cz0=',cz0,1)

    fpmax=np.max(np.abs(fps[0:n]))
###    wp.prin2('fpmax=',fpmax,1)
###    exit()


#
#    check if constantly 0, or if p is infinite
#
    if (p <= 0 or p == np.inf):
        vnorm = fpmax
        return(vnorm)
#
    if (fpmax == 0):
        vnorm=0.0
        return(vnorm)


###    wp.prin2('fpmax=',fpmax,1)


#
#    otherwise, find p-norm
#
    dt = (b-a) / n

    vnorm=np.sum(np.abs(fps[0:n] / fpmax)**p) * dt
    vnorm = vnorm**(1/p)
    vnorm = vnorm * fpmax

###    wp.prin2('vnorm, inside=',vnorm,1)


###    exit()

    return(vnorm)

#
#
#

def slicedcramer_dumb(gs,hs,ts,n,r,p,nangls,vnorms):
#
#                            description:
#
#    This code evaluates the sliced Cramer distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    slow and dumb, without vectorization or use of the NUFFT; a fast version,
#    whose output should be identical up to round-off, is slicedcramer_nufft.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#
#
#                        output parameters:
#
#  vnorms - nangls-length array containing the distances between each pair
#    of projections
#
#
#                    return parameters:
#
#  dist - the sliced Cramer distance
#
#

    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')
#
    angls = np.empty(nangls+1,dtype='float64')
    fprojs_arr = np.empty([nangls,n],dtype='float64')

    fhats_arr = np.empty([nangls,n],dtype='complex128')
    fphats_arr = np.empty([nangls,n],dtype='complex128')

    fps_arr = np.empty([nangls,n],dtype='float64')

    fs = np.empty([n,n],dtype='float64')



    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)

###    wp.prinr2('fs=',fs,n,6)


    for i in range(0,n):
        for j in range(0,n):
            fs[i,j] = hs[i,j] - gs[i,j]

    rq = n/(4*r)



#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    for k in range(0,nh2):
        kfs[k] = k

    for k in range(nh2,n):
        kfs[k] = k-n

    for k in range(0,n):
        freqs[k] = kfs[k] / (2*r)


    wp.prinf('kfs=',kfs,n)
    wp.prin2('freqs=',freqs,n+1)
    wp.prin2('rq=',rq,1)

#
#    angles grid from 0 to pi
#
    rpi = np.pi

    for i in range(0,nangls+1):
        angls[i] = i*rpi/nangls

    wp.prin2('angls=',angls,nangls+1)
###    wp.prin2('dangl=',dangl,1)


###    exit()



#
#    compute the Volterra norm of each projection
#
    for iang in range(0,nangls):
        angl=angls[iang]

        for i in range(0,n):
            sq = np.cos(angl)*freqs[i]
            tq = np.sin(angl)*freqs[i]
            fhats_arr[iang,i] = ftr_2d_fast(sq,tq,fs,ts,n,r)

        cz0 = iftr_fast(fhats_arr[iang,:],freqs,ts,fprojs_arr[iang,:],n)

        vnorms[iang] = vnorm_fourier_dumb(fhats_arr[iang,:],ts,n,\
            -r,r,p,fps_arr[iang,:],fphats_arr[iang,:],kfs,freqs)


###        vnorms[iang] = vnorm_fourier_fast(fhats_arr[iang,:],ts,n,\
###            -r,r,p,fps_arr[iang,:],fphats_arr[iang,:],kfs,freqs)




###    wp.prin2('angls=',angls,nangls)
###    wp.prin2('vnorms=',vnorms,nangls)

#
#    evaluate sliced norm
#
    vnorm_max = 0.0

    for iang in range(0,nangls):
        if (vnorms[iang] > vnorm_max):
            vnorm_max = vnorms[iang]
    wp.prin2('vnorm_max=',vnorm_max,1)

    if (vnorm_max == 0):
        dist = 0
        return(dist)

    if (p<=0 or p==np.inf):
        dist = vnorm_max
        return(dist)


    dist = 0.0
    for iang in range(0,nangls):
        dist = dist + (vnorms[iang]/vnorm_max)**p / nangls
    dist = dist**(1/p)
    dist = dist*vnorm_max


###    wp.prinr2('fprojs_arr=',fprojs_arr,nangls,6)
    wp.prin2('slnorm, inside=',dist,1)
###    wp.prin2('vnorms=',vnorms,nangls)


    return(dist)

#
#
#

def slicedwasser_nufft(gs,hs,ts,n,r,p,nangls,m,wdists):
#
#                            description:
#
#    This code evaluates the sliced Wasserstein distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    vectorized and uses the FINUFFT to evaluate the projections; a slower,
#    dumber, but more readable code, whose output should be identical
#    up to round-off, is slicedwasser_dumb.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#  m - the number of points used to discretized the inverse CDFs
#
#
#                        output parameters:
#
#  wdists - nangls-length array containing the distances between each pair
#    of projections
#
#
#                    return parameters:
#
#  dist - the sliced Wasserstein distance
#
#

    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')
#
    angls = np.empty(nangls+1,dtype='float64')
#
    hprojs_arr = np.empty([nangls,n],dtype='float64')
    hhats_arr = np.empty([nangls,n],dtype='complex128')
    hphats_arr = np.empty([nangls,n],dtype='complex128')
    hps_arr = np.empty([nangls,n+1],dtype='float64')
    hinvs_arr = np.empty([nangls,m+1],dtype='float64')
#
    gprojs_arr = np.empty([nangls,n],dtype='float64')
    ghats_arr = np.empty([nangls,n],dtype='complex128')
    gphats_arr = np.empty([nangls,n],dtype='complex128')
    gps_arr = np.empty([nangls,n+1],dtype='float64')
    ginvs_arr = np.empty([nangls,m+1],dtype='float64')
#
    difs_arr = np.empty([nangls,m+1],dtype='float64')


    hczs = np.empty(n*nangls,dtype='complex128')
    gczs = np.empty(n*nangls,dtype='complex128')

    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    
    hs_comp = np.empty([n,n],dtype='complex128')
    gs_comp = np.empty([n,n],dtype='complex128')



    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)

###    wp.prinr2('fs=',fs,n,6)


    rq = n/(4*r)


#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

###    wp.prinf('kfs=',kfs,n)
###    wp.prin2('freqs=',freqs,n+1)
    wp.prin2('rq=',rq,1)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,rpi,nangls+1)

    dangl = rpi / nangls


###    wp.prin2('angls=',angls,nangls+1)
###    wp.prin2('dangl=',dangl,1)



    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])


    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)



#
#    projections of h
#
    hs_comp[0:n,0:n] = hs[0:n,0:n]
    finufft.nufft2d2(xs, ys, hs_comp, hczs, eps=1e-15, isign=-1)
    hczs = hczs / (2*rq)**2
    hhats_arr[0:nangls,0:n] = np.reshape(hczs[0:nangls*n],[nangls,n])
    iftrs_fast(hhats_arr,freqs,ts,hprojs_arr,nangls,n)


#
#    projections of g
#
    gs_comp[0:n,0:n] = gs[0:n,0:n]
    finufft.nufft2d2(xs, ys, gs_comp, gczs, eps=1e-15, isign=-1)
    gczs = gczs / (2*rq)**2
    ghats_arr[0:nangls,0:n] = np.reshape(gczs[0:nangls*n],[nangls,n])
    iftrs_fast(ghats_arr,freqs,ts,gprojs_arr,nangls,n)


    wp.prinz('hhats_arr=',hhats_arr,n)
    wp.prinz('ghats_arr=',ghats_arr,n)


    wp.prinf('nangls=',nangls,1)
    wp.prinf('n=',n,1)

###    exit()

    wp.prin2('hprojs_arr=',hprojs_arr,n)
    wp.prin2('gprojs_arr=',gprojs_arr,30)


#
#    compute all wasserstein distances
#

    wassers_fast(hprojs_arr,gprojs_arr,ts,m,n,-r,r,p, \
        hps_arr,gps_arr,hinvs_arr,ginvs_arr,difs_arr,nangls,wdists)

###    wp.prin2('wdists=',wdists,nangls)

#
#    evaluate sliced distance
#
    dww = 1/nangls
    wdist = lpnorm_fast(wdists,nangls,p,dww)


    return(wdist)

#
#
#

def wassers_fast(hs,gs,ts,m,n,a,b,p,hps,gps,hinvs,ginvs,difs,k,wdists):

    ivals_h = np.empty([k,m],dtype='int32')
    ivals_g = np.empty([k,m],dtype='int32')
    yedges=np.empty(m+1,dtype='float64')
    ys=np.empty(m+1,dtype='float64')

    hh=(b-a)/n

    hl = np.cumsum(hs[0:k,0:n+1],axis=1)*hh
    hr = np.cumsum(hs[0:k,1:n+1],axis=1)*hh

    hps[0:k,0] = 0
    hps[0:k,1:n] = (hl[0:k,0:n-1] + hr[0:k,0:n-1]) / 2
    hps[0:k,n]=1


    gl = np.cumsum(gs[0:k,0:n+1],axis=1)*hh
    gr = np.cumsum(gs[0:k,1:n+1],axis=1)*hh

    gps[0:k,0] = 0
    gps[0:k,1:n] = (gl[0:k,0:n-1] + gr[0:k,0:n-1]) / 2
    gps[0:k,n]=1


    ymin = 0.0
    ymax = 1.0

    yedges=np.linspace(ymin,ymax,m+1)

    dy = yedges[1] - yedges[0]
    ys[0:m] = (yedges[0:m] + yedges[1:m+1])/2


#
#    distance between inverse CDFs, by midpoint rule
#
    cdfinvs_all_c(ts,m,n,hps,hinvs,ivals_h,ys,k)
    cdfinvs_all_c(ts,m,n,gps,ginvs,ivals_g,ys,k)
    difs[0:k,0:m] = hinvs[0:k,0:m] - ginvs[0:k,0:m]


    if (p<=0 or p==np.inf):
        wdists[0:k] = np.max(np.abs(difs[0:k,0:m]),axis=1)
        return()

    wdists[0:k] = la.norm(difs[0:k,0:m],ord=p,axis=1)
    wdists[0:k] = wdists[0:k] * dy**(1/p)


###    wp.prin2('wdists, here=',wdists,k)

    return()

#
#
#

def cdfinvs_all_c(ts,m,n,hps,hinvs,ivals,ys,k):

    path = os.getcwd()
    clib = ct.CDLL(os.path.join(path, 'cdfinvs.so'))

    clib.cdfinvs_all.argtypes= \
        [np.ctypeslib.ndpointer(dtype=np.float64), \
        np.ctypeslib.ndpointer(dtype=np.float64), \
        ct.c_int, \
        ct.c_int, \
        ct.c_int, \
        np.ctypeslib.ndpointer(dtype=np.int32)]

    clib.cdfinvs_all(ys,hps,n,m,k,ivals)


###    wp.prinf('ivals=',ivals[k-1,:],m)



    hinvs[0:k,0:m] = (ts[ivals[0:k,0:m]]+ts[ivals[0:k,0:m]+1])/2

###    for ijk in range(0,k):
###        hinvs[ijk,0:m] = (ts[ivals[ijk,0:m]]+ts[ivals[ijk,0:m]+1])/2

    return()


    for ijk in range(0,k):
        cdfinvs_c(ys,m,n,hinvs[ijk,:],hps[ijk,:],ts,ivals[ijk,:])


    return()

#
#
#

def cdfinvs_c(ys,m,n,finvs,fints,tgrid,ivals):

    path = os.getcwd()
    clib = ct.CDLL(os.path.join(path, 'cdfinvs.so'))

    clib.cdfinvs_march.argtypes= \
        [np.ctypeslib.ndpointer(dtype=np.float64), \
        np.ctypeslib.ndpointer(dtype=np.float64), \
        ct.c_int, \
        ct.c_int, \
        np.ctypeslib.ndpointer(dtype=np.int32)]


    clib.cdfinvs_march(ys,fints,n,m,ivals)

    finvs[0:m] = (tgrid[ivals[0:m]]+tgrid[ivals[0:m]+1])/2


    return()

#
#
#


def slicedlebesg_nufft(gs,hs,ts,n,r,p,nangls,vnorms,kfs,freqs):
#
#                            description:
#
#    This code evaluates the sliced Lebesgue distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    vectorized and uses the FINUFFT to evaluate the projections; a slower,
#    dumber, but more readable code, whose output should be identical
#    up to round-off, is slicedcramer_dumb.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#
#
#                        output parameters:
#
#  vnorms - nangls-length array containing the distances between each pair
#    of projections
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#
#
#                    return parameters:
#
#  dist - the sliced Cramer distance
#
#

    angls = np.empty(nangls+1,dtype='float64')

    fprojs_arr = np.empty([nangls,n],dtype='float64')
    fhats_arr = np.empty([nangls,n],dtype='complex128')
    fphats_arr = np.empty([nangls,n],dtype='complex128')
    fps_arr = np.empty([nangls,n],dtype='float64')

###    fprojs_arr[0:nangls,0:n] = 10000


    fs = np.empty([n,n],dtype='float64')


    czs = np.empty(n*nangls,dtype='complex128')
    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    
    fs_comp = np.empty([n,n],dtype='complex128')


    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)



    fs[0:n,0:n] = hs[0:n,0:n] - gs[0:n,0:n]

    rq = n/(4*r)
###    wp.prin2('rq=',rq,1)

#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,rpi,nangls+1)
    dangl = rpi / nangls

    wp.prin2('dangl=',dangl,1)


    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])
#
    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)

#
#    compute the projections with the FINUFFT
#
    fs_comp[0:n,0:n] = fs[0:n,0:n]
    finufft.nufft2d2(xs, ys, fs_comp, czs, eps=1e-15, isign=-1)
    czs = czs / (2*rq)**2

###    wp.prinz('czs=',czs,n*nangls)

    fhats_arr[0:nangls,0:n] = np.reshape(czs[0:nangls*n],[nangls,n])


#
#    evaluate the sliced norm
#
    lnorms_fourier_fast(fhats_arr,ts,nangls,n,-r,r,p,\
        fps_arr,fphats_arr,kfs,freqs,vnorms)

    dww = 1/nangls
    dist = lpnorm_fast(vnorms,nangls,p,1/nangls)


    return(dist)

#
#
#

def lnorms_fourier_fast(fzs,ts,m,n,a,b,p,fps,fpzs,kfs,freqs,vnorms):
#
#    vectorized code for computing the p-norms of m functions,
#    taking as inputs their Fourier transforms on a frequency grid.
#

    zzzs = np.empty(n,dtype='complex128')

    nh2 = np.int(n/2)

#
#    phase factors
#
    zzzs[0:n] = np.exp(2*np.pi*1j*freqs[0:n]*a)


#
#    evaluate partial integrals and find norms
#
    iftrs_fast(fzs,freqs,ts,fps,m,n)

    fpmaxs = np.max(np.abs(fps[0:m,0:n]),axis=1)

    if (p <= 0 or p == np.inf):
        vnorms[0:m] = fpmaxs[0:m]
        return()

    dt = (b-a) / n
    vnorms[0:m] = la.norm(fps[0:m,0:n],ord=p,axis=1)* dt**(1/p)



    return()

#
#
#

def slicedwasser_dumb(gs,hs,ts,n,r,p,nangls,m,wdists):
#
#                            description:
#
#    This code evaluates the sliced Wasserstein distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    slow and dumb, without vectorization or use of the NUFFT; a fast version,
#    whose output should be identical up to round-off, is slicedwasser_nufft.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#  m - the number of points used to discretized the inverse CDFs
#
#
#                        output parameters:
#
#  wdists - nangls-length array containing the distances between each pair
#    of projections
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#
#
#                    return parameters:
#
#  dist - the sliced Wasserstein distance
#
#
    kfs = np.empty(n,dtype='int32')
    freqs = np.empty(n,dtype='float64')

    angls = np.empty(nangls+1,dtype='float64')

    hprojs_arr = np.empty([nangls,n],dtype='float64')
    hhats_arr = np.empty([nangls,n],dtype='complex128')
    hphats_arr = np.empty([nangls,n],dtype='complex128')
    hps_arr = np.empty([nangls,n+1],dtype='float64')

    gprojs_arr = np.empty([nangls,n],dtype='float64')
    ghats_arr = np.empty([nangls,n],dtype='complex128')
    gphats_arr = np.empty([nangls,n],dtype='complex128')
    gps_arr = np.empty([nangls,n+1],dtype='float64')

    hinvs_arr = np.empty([nangls,m+1],dtype='float64')
    ginvs_arr = np.empty([nangls,m+1],dtype='float64')

    difs_arr = np.empty([nangls,m+1],dtype='float64')


    wp.prinf('nangls=',nangls,1)
###    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)

###    wp.prinr2('fs=',fs,n,6)


    rq = n/(4*r)


#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    for k in range(0,nh2):
        kfs[k] = k

    for k in range(nh2,n):
        kfs[k] = k-n

    for k in range(0,n):
        freqs[k] = kfs[k] / (2*r)

###    wp.prinf('kfs=',kfs,n)
###    wp.prin2('freqs=',freqs,n+1)
    wp.prin2('rq=',rq,1)

#
#    angles grid from 0 to pi
#
    rpi = np.pi
    for i in range(0,nangls+1):
        angls[i] = i*rpi/nangls


    dangl = rpi / nangls

    wp.prin2('angls=',angls,nangls+1)
    wp.prin2('dangl=',dangl,1)


#
#    compute the Wasserstein distance between projection pairs
#
    for iang in range(0,nangls):
        angl=angls[iang]

        for i in range(0,n):
            sq = np.cos(angl)*freqs[i]
            tq = np.sin(angl)*freqs[i]
            ghats_arr[iang,i] = ftr_2d_fast(sq,tq,gs,ts,n,r)
            hhats_arr[iang,i] = ftr_2d_fast(sq,tq,hs,ts,n,r)

        cz0 = iftr_fast(hhats_arr[iang,:],freqs,ts,hprojs_arr[iang,:],n)
        cz0 = iftr_fast(ghats_arr[iang,:],freqs,ts,gprojs_arr[iang,:],n)

        wp.prini(0,0)
        wdists[iang] = wasser(hprojs_arr[iang,:],gprojs_arr[iang,:],ts,m,n, \
            -r,r,p,hps_arr[iang,:],gps_arr[iang,:],hinvs_arr[iang,:], \
            ginvs_arr[iang,:],difs_arr[iang,:])
        wp.prini(13,0)

    wp.prin2('wdists=',wdists,nangls)

###    exit()

#
#    evaluate sliced norm
#
    wdist_max = 0.0

    for iang in range(0,nangls):
        if (wdists[iang] > wdist_max):
            wdist_max = wdists[iang]
    wp.prin2('wdist_max=',wdist_max,1)


    if (wdist_max == 0):
        wdist = 0
        return(wdist)

    if (p<=0 or p==np.inf):
        wdist = wdist_max
        return(wdist)


    wdist = 0.0
    for iang in range(0,nangls):
        wdist = wdist + (wdists[iang]/wdist_max)**p / nangls
    wdist = wdist**(1/p)
    wdist = wdist*wdist_max


###    wp.prinr2('fprojs_arr=',fprojs_arr,nangls,6)
    wp.prin2('wdist, inside=',wdist,1)
###    wp.prin2('wdists=',wdists,nangls)


    return(wdist)


    plt.figure(10)
    plt.imshow(fprojs_arr[0:nangls,0:n])
    plt.savefig('allmat.pdf')

    plt.figure(11)
    plt.imshow(fps_arr[0:nangls,0:n])
    plt.savefig('allmat2.pdf')



    return(wdist)

#
#
#

def ftr_2d_fast(sf,tf,fs,ts,n,r):
#
#
#                        description:
#
#    This code uses the values of f on a 2D grid to evaluate the
#    2D Fourier transform at a single frequency pair (sf,tf).
#    The code is vectorized.
#
#
#                    input parameters:
#
#  sf,tf - the real-valued frequency coordinates
#  fs - n-by-n real array with values of f sampled on n-by-n equispaced
#    grid over [-r,r) x [-r,r)
#  ts - length n real array with equispaced values on grid over [-r,r)
#  n - the number of samples per dimension (n^2 samples in total). It is
#    assumed that n is even.
#  r - half the length of each side of f's domain
#
#
#                    return parameters:
#
#  fhat - the 2D Fourier transform of f at (sf,tf)
#
#

    zis = np.empty([1,n],dtype='complex128')
    zjs = np.empty([1,n],dtype='complex128')


###    dt = ts[1] - ts[0]
    dt = 2*r / n


    zis[0,0:n] = np.exp(-2*np.pi*1j*ts[0:n]*sf)
    zjs[0,0:n] = np.exp(-2*np.pi*1j*ts[0:n]*tf)

    zz = np.multiply(np.transpose(zis[0:n]),zjs[0:n])
    pp = np.multiply(fs[0:n,0:n],zz[0:n,0:n])

    fhat = np.sum(np.sum(pp[0:n,0:n])) * dt**2



    return(fhat)

#
#
#

def ftr_2d_dumb(sf,tf,fs,ts,n,r):
#
#
#                        description:
#
#    This code uses the values of f on a 2D grid to evaluate the
#    2D Fourier transform at a single frequency pair (sf,tf).
#    The code is not vectorized or optimized in any way.
#
#
#                    input parameters:
#
#  sf,tf - the real-valued frequency coordinates
#  fs - n-by-n real array with values of f sampled on n-by-n equispaced
#    grid over [-r,r) x [-r,r)
#  ts - length n real array with equispaced values on grid over [-r,r)
#  n - the number of samples per dimension (n^2 samples in total). It is
#    assumed that n is even.
#  r - half the length of each side of f's domain
#
#
#                    return parameters:
#
#  fhat - the 2D Fourier transform of f at (sf,tf)
#
#


###    dt = ts[1] - ts[0]
    dt = 2*r / n

###    wp.prin2('dt=',dt,1)
###    wp.prin2('ts=',ts,n)
###    wp.prin2('r=',r,1)

###    exit()

    fhat = 0.0

    for i in range(0,n):
        for j in range(0,n):
            zi = np.exp(-2*np.pi*1j*sf*ts[i])
            zj = np.exp(-2*np.pi*1j*tf*ts[j])

            fhat = fhat + fs[i,j]*zi*zj*dt**2


    return(fhat)

#
#
#

def wasser(fvals,gvals,tgrid,m,n,a,b,p,fints,gints,finvs,ginvs,difs):


    ivals = np.empty(m+10,dtype='int32')
    yedges=np.empty(m+10,dtype='float64')
    ys=np.empty(m+10,dtype='float64')


    wp.prinf('n=',n,1)
    wp.prinf('m=',m,1)


    wp.prin2('tgrid=',tgrid,n+1)

    wp.prin2('a=',a,1)
    wp.prin2('b=',b,1)

    hh=(b-a)/n

    wp.prin2('hh=',hh,1)


#
#    partial integrals on grid
#
    allints_dumb(fvals,n,hh,fints)
    allints_dumb(gvals,n,hh,gints)


    wp.prin2('fints=',fints,n+1)
    wp.prin2('gints=',gints,n+1)


    ymin = 0.0
    ymax = 1.0

    wp.prin2('ymin=',ymin,1)
    wp.prin2('ymax=',ymax,1)

#
#    edges between 0 and 1 -- includes endpoints
#
    for i in range(0,m+1):
        wp.prinf('i=',i,1)
        yedges[i] = ymin + (ymax-ymin)*i/m

    dy = yedges[1] - yedges[0]

    wp.prin2('yedges=',yedges,m+1)
    wp.prin2('dy=',dy,1)

#
#    midpoint grid between 0 and 1
#
    for i in range(0,m):
        ys[i] = (yedges[i] + yedges[i+1])/2


    wp.prin2('ys=',ys,m)

    dy2 = ys[1] - ys[0]
    wp.prin2('dy2=',dy2,1)
    chk0 = dy2 - dy
    wp.prin2('chk0=',chk0,1)



#
#    inverse CDFs over [0,1]
#
    cdfinvs_march(ys,m,a,b,n,finvs,fints,tgrid,ivals)
    wp.prin2('finvs=',finvs,m)


    cdfinvs_march(ys,m,a,b,n,ginvs,gints,tgrid,ivals)

    wp.prin2('ginvs=',ginvs,m)

#
#    distance between inverse CDFs, by midpoint rule
#
    wdist = 0
    dif_max = 0
    for i in range(0,m):
        difs[i] = finvs[i] - ginvs[i]
        if (np.abs(difs[i]) > dif_max):
            dif_max = np.abs(difs[i])


    wp.prin2('dif_max=',dif_max,1)


    if (dif_max == 0):
        wdist=0
        return(wdist)

    if (p<=0 or p==np.inf):
        wdist = dif_max
        return(wdist)


    for i in range(0,m):
        wdist = wdist + np.abs(difs[i]/dif_max)**p * dy
    wdist = wdist**(1/p) * dif_max



###    wp.prini(13,0)
    wp.prin2('difs=',difs,m)


###    plt.figure(15)
###    plt.plot(ys[0:m],difs[0:m])
###    plt.savefig('dif.pdf')



    wp.prin2('wdist, inside=',wdist,1)

###    exit()



    return(wdist)

#
#
#

def allints_dumb(fvals,n,hh,fints):
#
#    all partial integrals of density on grid, via trapezoidal rule
#
    fints[0]=0
    for m in range(1,n):
        fints[m] = fints[m-1] + fvals[m-1]*hh/2 + fvals[m]*hh/2

    fints[n] = 1



    return()

#
#    . . . stupid way
#
    fints[0]=0
    for m in range(1,n):
        fints[m] = (fvals[0] + fvals[m])*hh/2
        for j in range(1,m):
             fints[m] = fints[m] + fvals[j]*hh

    fints[n] = 1

#
    return()

#
#
#

def cdfinvs_march(yys,m,a,b,n,finvs,ygrid,tgrid,ivals):

    vals_grid = np.empty(n+10,dtype='float64')


###    [finvs[0],ivals[0]] = cdfinv_march(yys[0],a,b,n,tgrid,ygrid)



###    wp.prini(0,0)

    istar=0

    for ijk in range(0,m):

        wp.prinf('ijk=',ijk,1)

###        wp.prinf('istar=',istar,1)
        wp.prinf('n=',n,1)

        vals_grid[istar] = ygrid[istar] - yys[ijk]
#

        for i in range(istar+1,n+1):
            vals_grid[i] = ygrid[i] - yys[ijk]
        #
            if (vals_grid[i]*vals_grid[i-1] < 0):
    ###            t0 = tgrid[i]
                finvs[ijk] = (tgrid[i]+tgrid[i-1])/2
                ivals[ijk] = i-1
                istar = i-1

                wp.prinf('exiting loop at i=',i,1)
                break


###    for ijk in range(0,m):
###        [finvs[ijk],ivals[ijk]] = cdfinv_march(yys[ijk],a,b,n,tgrid,ygrid)

###    wp.prini(13,0)

    wp.prin2('finvs, inside march=',finvs,m)
    wp.prinf('ivals, inside march=',ivals,m)
    wp.prin2('a=',a,1)
    wp.prin2('b=',b,1)


    return()

#
#
#

def slicedwasser_ps(gs,hs,ts,n,r,ps,nps,nangls,m,wdists,freqs,kfs,dists):
#
#                            description:
#
#    This code evaluates the sliced Wasserstein distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    vectorized and uses the FINUFFT to evaluate the projections; a slower,
#    dumber, but more readable code, whose output should be identical
#    up to round-off, is slicedwasser_dumb.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#  m - the number of points used to discretized the inverse CDFs
#
#
#                        output parameters:
#
#  wdists - nangls-length array containing the distances between each pair
#    of projections
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#
#
#                    return parameters:
#
#  dist - the sliced Wasserstein distance
#
#

    angls = np.empty(nangls+1,dtype='float64')

    hprojs_arr = np.empty([nangls,n+1],dtype='float64')
    hhats_arr = np.empty([nangls,n+1],dtype='complex128')
    hphats_arr = np.empty([nangls,n+1],dtype='complex128')
    hps_arr = np.empty([nangls,n+1],dtype='float64')
    hinvs_arr = np.empty([nangls,m+1],dtype='float64')

    gprojs_arr = np.empty([nangls,n+1],dtype='float64')
    ghats_arr = np.empty([nangls,n+1],dtype='complex128')
    gphats_arr = np.empty([nangls,n+1],dtype='complex128')
    gps_arr = np.empty([nangls,n+1],dtype='float64')
    ginvs_arr = np.empty([nangls,m+1],dtype='float64')

    difs_arr = np.empty([nangls,m+1],dtype='float64')


    hczs = np.empty(n*nangls,dtype='complex128')
    gczs = np.empty(n*nangls,dtype='complex128')

    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    
    hs_comp = np.empty([n,n],dtype='complex128')
    gs_comp = np.empty([n,n],dtype='complex128')




    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)

###    wp.prinr2('fs=',fs,n,6)


    rq = n/(4*r)


#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

###    wp.prinf('kfs=',kfs,n)
###    wp.prin2('freqs=',freqs,n+1)
    wp.prin2('rq=',rq,1)

#
#    angles from 0 to pi, and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,rpi,nangls+1)

    dangl = rpi / nangls


###    wp.prin2('angls=',angls,nangls+1)
###    wp.prin2('dangl=',dangl,1)



    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])


    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)



#
#    projections of h
#
    hs_comp[0:n,0:n] = hs[0:n,0:n]
    finufft.nufft2d2(xs, ys, hs_comp, hczs, eps=1e-15, isign=-1)
    hczs = hczs / (2*rq)**2
    hhats_arr[0:nangls,0:n] = np.reshape(hczs[0:nangls*n],[nangls,n])
    iftrs_fast(hhats_arr,freqs,ts,hprojs_arr,nangls,n)


#
#    projections of g
#
    gs_comp[0:n,0:n] = gs[0:n,0:n]
    finufft.nufft2d2(xs, ys, gs_comp, gczs, eps=1e-15, isign=-1)
    gczs = gczs / (2*rq)**2
    ghats_arr[0:nangls,0:n] = np.reshape(gczs[0:nangls*n],[nangls,n])
    iftrs_fast(ghats_arr,freqs,ts,gprojs_arr,nangls,n)



    wp.prinz('hhats_arr=',hhats_arr,22)
    wp.prinz('ghats_arr=',ghats_arr,22)


    wp.prin2('hprojs_arr=',hprojs_arr,30)
    wp.prin2('gprojs_arr=',gprojs_arr,30)

#
#    evaluate sliced distance for each p
#
    dww = 1/nangls
    for i in range(0,nps):
        wassers_fast(hprojs_arr,gprojs_arr,ts,m,n,-r,r,ps[i], \
            hps_arr,gps_arr,hinvs_arr,ginvs_arr,difs_arr,nangls,wdists)
        dists[i] = lpnorm_fast(wdists,nangls,ps[i],dww)

###    wp.prin2('wdist, inside=',wdist,1)



    return()

#
#
#

def slicedcramer_ps(hs,gs,ts,n,r,ps,nps,nangls,vnorms,kfs,freqs,dists):
#
#                            description:
#
#    This code evaluates the sliced Cramer p-distances between two functions
#    h and g evaluated on a 2D grid of equispaced points, for multiple
#    values of p. This code is mostly vectorized and uses the FINUFFT to evaluate
#    the projections.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  ps - array of size nps containing the the norm parameters. To use the infinity
#    norm, set p<=0 or np.inf
#  nps - the number of distances evaluated
#  nangls - the number of projection angles used to evaluate the sliced distance
#
#
#                        output parameters:
#
#  vnorms - nangls-length array containing the distances between each pair
#    of projections
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#  dists - array of size nps containing the sliced Cramer distances
#
#
#
#

    angls = np.empty(nangls+1,dtype='float64')

    fprojs_arr = np.empty([nangls,n],dtype='float64')
    fhats_arr = np.empty([nangls,n],dtype='complex128')
    fphats_arr = np.empty([nangls,n],dtype='complex128')
    fps_arr = np.empty([nangls,n],dtype='float64')

###    fprojs_arr[0:nangls,0:n] = 10000


    fs = np.empty([n,n],dtype='float64')


    czs = np.empty(n*nangls,dtype='complex128')
    xs = np.empty(n*nangls,dtype='float64')    
    ys = np.empty(n*nangls,dtype='float64')    
    fs_comp = np.empty([n,n],dtype='complex128')


    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)



    fs[0:n,0:n] = hs[0:n,0:n] - gs[0:n,0:n]

    rq = n/(4*r)
###    wp.prin2('rq=',rq,1)

#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n
    freqs[0:n] = kfs[0:n] / (2*r)

#
#    angles and 2D frequencies
#
    rpi = np.pi
    angls[0:nangls+1] = np.linspace(0,rpi,nangls+1)
    dangl = rpi / nangls

###    wp.prin2('dangl=',dangl,1)


    sfs_all = np.outer(np.cos(angls[0:nangls]),freqs[0:n])
    tfs_all = np.outer(np.sin(angls[0:nangls]),freqs[0:n])
#
    xs[0:n*nangls] = 2*np.pi*np.reshape(sfs_all,[1,nangls*n]) / (2*rq)
    ys[0:n*nangls] = 2*np.pi*np.reshape(tfs_all,[1,nangls*n]) / (2*rq)

#
#    compute the projections with the FINUFFT
#
    fs_comp[0:n,0:n] = fs[0:n,0:n]
    finufft.nufft2d2(xs, ys, fs_comp, czs, eps=1e-15, isign=-1)
    czs = czs / (2*rq)**2

###    wp.prinz('czs=',czs,n*nangls)

    fhats_arr[0:nangls,0:n] = np.reshape(czs[0:nangls*n],[nangls,n])


#
#    evaluate the sliced norm for each p
#
    dww = 1/nangls

    wp.prin2('dww=',dww,1)

    for i in range(0,nps):
        vnorms_fourier_fast(fhats_arr,ts,nangls,n,-r,r,ps[i],\
            fps_arr,fphats_arr,kfs,freqs,vnorms)
        dists[i] = lpnorm_fast(vnorms,nangls,ps[i],dww)

    return()

#
#
#

def vnorm_fourier_dumb(fzs,ts,n,a,b,p,fps,fpzs,kfs,freqs):
#
#                            description:
#
#    This code evaluates the mean-centered Volterra p-norm using the
#    DFT and spectral integration. It takes as input samples from
#    the Fourier transform of f, rather than values of f itself.
#
#
#                        input parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#  n - the number of samples; assumed to be an even integer (for now)
#  a,b - the endpoints of the interval
#
#
#                        output parameters:
#
#  fzs - n-length complex array containing values of the Fourier transform
#    of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#  fpzs - n-length complex array containing the Fourier coefficients
#    of f's Volterra transform
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#
#
#                    return parameters:
#
#  vnorm - the Volterra p-norm
#
#

    nh2 = np.int(n/2)


#
#    spectral integration
#
    for k in range(1,n):
        fpzs[k] = fzs[k] / (2*np.pi*1j*freqs[k])

    fpzs[nh2] = 0

#
#    zero-th frequency, excluding frequencies n/2 and -n/2
#
    fpzs[0] = 0.0
    for k in range(1,nh2):
        fpzs[0] = fpzs[0] - fpzs[k] * np.exp(2*np.pi*1j*freqs[k]*a)
###        fpzs[0] = fpzs[0] - fpzs[k+nh2] * np.exp(2*np.pi*1j*freqs[k+nh2]*a)

    for k in range(nh2+1,n):
        fpzs[0] = fpzs[0] - fpzs[k] * np.exp(2*np.pi*1j*freqs[k]*a)

#
#    evaluate Vf and find p-norm
#
    cz0=iftr_dumb(fpzs,freqs,ts,fps,n)
###    wp.prin2('cz0=',cz0,1)

###    wp.prin2('fps=',fps,n)

###    plt.plot(ts[0:n],fps[0:n])
###    plt.savefig('pp.pdf')


    fpmax=0.0
    for i in range(0,n):
        if (np.abs(fps[i]) > fpmax):
            fpmax = np.abs(fps[i])


#
#    check if constantly 0, or if p is infinite
#
    if (p <= 0 or p == np.inf):
        vnorm = fpmax
        return(vnorm)
#
    if (fpmax == 0):
        vnorm=0.0
        return(vnorm)


###    wp.prin2('fpmax=',fpmax,1)


#
#    otherwise, find p-norm
#
    dt = (b-a) / n

    vnorm = 0.0
    for i in range(0,n):
        vnorm = vnorm + np.abs(fps[i] / fpmax)**p * dt
    vnorm = vnorm**(1/p)
    vnorm = vnorm * fpmax

###    wp.prin2('vnorm, inside=',vnorm,1)


    return(vnorm)

#
#
#

def lpdist2_fast(hs,gs,m,n,p,ds,dt):

    difs = np.empty(m*n,dtype='float64')
    difs[0:m*n] = np.reshape(hs[0:m,0:n] - gs[0:m,0:n],[1,m*n])
    dist = lpnorm_fast(difs,m*n,p,ds*dt)

    return(dist)

#
#
#

def lpdist_fast(hs,gs,n,p,dt):

    difs = np.empty(n,dtype='float64')
    difs[0:n] = hs[0:n] - gs[0:n]
    dist = lpnorm_fast(difs,n,p,dt)

    return(dist)

#
#
#

def lpnorm_fast(fs,n,p,dt):

###    fmax = 0.0
    fmax = np.max(fs[0:n])

###    wp.prin2('fmax=',fmax,1)

    if (fmax == 0):
        wdist = 0.0
        return(wdist)

    if (p<=0 or p==np.inf):
        wdist = fmax
        return(wdist)


    wdist = la.norm(fs[0:n]/fmax,ord=p)
    wdist = wdist * fmax * dt**(1/p)



    return(wdist)

#
#
#

def cramer1d(hs,gs,ts,n,a,b,p,fps,fzs,fpzs,kfs,freqs):

    fs = np.empty(n,dtype='float64')
    fs[0:n] = hs[0:n] - gs[0:n]

    cdist = vnorm_fast(fs,ts,n,a,b,p,fps,fzs,fpzs,kfs,freqs)

    return(cdist)

#
#
#

def vnorm_fast(fs,ts,n,a,b,p,fps,fzs,fpzs,kfs,freqs):
#
#                            description:
#
#    This code evaluates the mean-centered Volterra p-norm using the
#    DFT and spectral integration.
#
#
#                        input parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#  n - the number of samples; assumed to be an even integer (for now)
#  a,b - the endpoints of the interval
#
#
#                        output parameters:
#
#  fzs - n-length complex array containing values of the Fourier transform
#    of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#  fpzs - n-length complex array containing the Fourier coefficients
#    of f's Volterra transform
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#
#
#                    return parameters:
#
#  vnorm - the Volterra p-norm
#
#

    zzzs = np.empty(n,dtype='complex128')

    nh2 = np.int(n/2)


    dt = (b-a) / n



#
#    Fourier transform of f and Vf
#
    ftr_fast(fzs,freqs,kfs,fs,ts,n,a,b)


###    ftr_dumb(fzs,freqs,kfs,fs,ts,n,a,b)



#
#    spectral integration
#

    fpzs[1:n] = np.divide(fzs[1:n],freqs[1:n])
    fpzs[1:n] = fpzs[1:n] / (2*np.pi*1j)

    fpzs[nh2] = 0

    wp.prinz('fzs=',fzs,6)


#
#    zero-th frequency, excluding frequencies n/2 and -n/2
#
    zzzs[0:n] = np.exp(2*np.pi*1j*freqs[0:n]*a)

    fpzs[0] = -np.sum(np.multiply(fpzs[1:nh2],zzzs[1:nh2]))
    fpzs[0] = fpzs[0]-np.sum(np.multiply(fpzs[nh2+1:n],zzzs[nh2+1:n]))

    wp.prinz('fpzs=',fpzs,2)




#
#    evaluate Vf and find p-norm
#
    cz0=iftr_fast(fpzs,freqs,ts,fps,n)
###    wp.prin2('cz0=',cz0,1)

    fpmax=np.max(np.abs(fps[0:n]))
###    wp.prin2('fpmax=',fpmax,1)
###    exit()


#
#    check if constantly 0, or if p is infinite
#
    if (p <= 0 or p == np.inf):
        vnorm = fpmax
        return(vnorm)
#
    if (fpmax == 0):
        vnorm=0.0
        return(vnorm)


###    wp.prin2('fpmax=',fpmax,1)


#
#    otherwise, find p-norm
#
    dt = (b-a) / n

    vnorm=np.sum(np.abs(fps[0:n] / fpmax)**p) * dt
    vnorm = vnorm**(1/p)
    vnorm = vnorm * fpmax

    wp.prin2('vnorm, inside=',vnorm,1)


###    exit()

    return(vnorm)

#
#
#

def iftr_fast(fzs,freqs,ts,fs,n):
#
#                            description:
#
#    This code evaluates f from its Fourier transform on an
#    equispaced grid between -1/R and 1/R, where R is the radius
#    of the interval (right endpoint 1/R not included).
#    The code uses the IFFT and is vectorized.
#
#
#                        input parameters:
#
#  fzs - n-length complex vector containing the n values of the Fourier
#    transform of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#    on a grid over [-1/r,1/r) (only the left endpoint is included)
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#  n - the number of samples; assumed to be an even integer (for now)
#
#
#                        output parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#
#
#                    return parameters:
#
#  cz0 - the normalized l1 norm of the complex part
#
#


    zzzs = np.empty(n,dtype='complex128')
    gggs = np.empty(n,dtype='complex128')
    fff = np.empty(n,dtype='complex128')

    dpi=np.pi
    zi = 1j

    df = freqs[1] - freqs[0]

    cz0 = 0.0

#
#    phase correction
#
    zzzs[0:n] = np.exp(2*dpi*zi*freqs[0:n]*ts[0])
    gggs[0:n] = np.multiply(fzs[0:n],zzzs[0:n])


#
#    inverse DFT, rescaled
#
    fff[0:n] = dft.ifft(gggs[0:n]) * df * n
    fs[0:n] = np.real(fff[0:n])


###    wp.prinz('fff, inside=',fff,n)
###    wp.prin2('fs=',fs,n)

#
#    size of imaginary part
#
    cz0 = np.sum(np.abs(np.imag(fff[0:n]))) / n

###    wp.prin2('cz0=',cz0,1)

    return(cz0)

#
#
#

def ftr_fast(fzs,freqs,kfs,fs,ts,n,a,b):
#
#                            description:
#
#    This code evaluates the Fourier transform of f on an
#    equispaced grid between -1/R and 1/R, where R is the radius
#    of the interval (right endpoint 1/R not included).
#    The code uses the FFT and is vectorized.
#
#
#                        input parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#  n - the number of samples; assumed to be an even integer (for now)
#  a,b - the endpoints of the interval
#
#
#                        output parameters:
#
#  fzs - n-length complex array containing values of the Fourier transform
#    of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#
#


    zzzs=np.empty(n,dtype='complex128')

    nh2 = np.int(n/2)

#
#    frequencies between -1/R and 1/R, in DFT ordering
#
    kfs[0:n] = range(0,n)
    kfs[nh2:n] = kfs[nh2:n] - n

    freqs[0:n] = kfs[0:n] / (b-a)

    wp.prinf('kfs=',kfs,n)
    wp.prin2('freqs=',freqs,n)

###    exit()


#
#    real-space grid
#
    dt = (b-a) / n
    ts[0:n] = np.linspace(a,b-dt,n)



    fzs[0:n] = dft.fft(fs[0:n]) * dt

#
#    phase correction
#
    zzzs[0:n] = np.exp(-2*np.pi*1j*a*freqs[0:n])
    fzs[0:n] = np.multiply(fzs[0:n],zzzs[0:n])

    wp.prinz('fzs=',fzs,6)


    return()

#
#
#

def vnorm_dumb(fs,ts,n,a,b,p,fps,fzs,fpzs,kfs,freqs):
#
#                            description:
#
#    This code evaluates the mean-centered Volterra p-norm using the
#    DFT and spectral integration.
#
#
#                        input parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#  n - the number of samples; assumed to be an even integer (for now)
#  a,b - the endpoints of the interval
#
#
#                        output parameters:
#
#  fzs - n-length complex array containing values of the Fourier transform
#    of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#  fpzs - n-length complex array containing the Fourier coefficients
#    of f's Volterra transform
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#
#
#                    return parameters:
#
#  vnorm - the Volterra p-norm
#
#



    nh2 = np.int(n/2)

    dt = (b-a) / n



#
#    spectral integration
#
    ftr_dumb(fzs,freqs,kfs,fs,ts,n,a,b)

    for k in range(1,n):
        fpzs[k] = fzs[k] / (2*np.pi*1j*freqs[k])

    fpzs[nh2] = 0

#
#    zero-th frequency, excluding frequencies n/2 and -n/2
#
    fpzs[0] = 0.0
    for k in range(1,nh2):
        fpzs[0] = fpzs[0] - fpzs[k] * np.exp(2*np.pi*1j*freqs[k]*a)
###        fpzs[0] = fpzs[0] - fpzs[k+nh2] * np.exp(2*np.pi*1j*freqs[k+nh2]*a)

    for k in range(nh2+1,n):
        fpzs[0] = fpzs[0] - fpzs[k] * np.exp(2*np.pi*1j*freqs[k]*a)



#
#    evaluate Vf and find p-norm
#
    cz0=iftr_dumb(fpzs,freqs,ts,fps,n)
###    wp.prin2('cz0=',cz0,1)

    wp.prin2('fps=',fps,n)

###    plt.plot(ts[0:n],fps[0:n])
###    plt.savefig('pp.pdf')


    fpmax=0.0
    for i in range(0,n):
        if (np.abs(fps[i]) > fpmax):
            fpmax = np.abs(fps[i])


#
#    check if constantly 0, or if p is infinite
#
    if (p <= 0 or p == np.inf):
        vnorm = fpmax
        return(vnorm)
#
    if (fpmax == 0):
        vnorm=0.0
        return(vnorm)


###    wp.prin2('fpmax=',fpmax,1)


#
#    otherwise, find p-norm
#
    dt = (b-a) / n

    vnorm = 0.0
    for i in range(0,n):
        vnorm = vnorm + np.abs(fps[i] / fpmax)**p * dt
    vnorm = vnorm**(1/p)
    vnorm = vnorm * fpmax

###    wp.prin2('vnorm, inside=',vnorm,1)


    return(vnorm)

#
#
#

def iftr_dumb(fzs,freqs,ts,fs,n):
#
#                            description:
#
#    This code evaluates f from its Fourier transform on an
#    equispaced grid between -1/R and 1/R, where R is the radius
#    of the interval (right endpoint 1/R not included).
#    The code is slow and dumb, using a brute force IDFT
#    and no vectorization.
#
#
#                        input parameters:
#
#  fzs - n-length complex vector containing the n values of the Fourier
#    transform of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#    on a grid over [-1/r,1/r) (only the left endpoint is included)
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#  n - the number of samples; assumed to be an even integer (for now)
#
#
#                        output parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#
#
#                    return parameters:
#
#  cz0 - the normalized l1 norm of the complex part
#
#

    dpi=np.pi
    zi = 1j

    df = freqs[1] - freqs[0]

    cz0 = 0.0

    for i in range(0,n):
        fzz = 0.0
        for k in range(0,n):
            fzz = fzz + fzs[k]*np.exp(2*dpi*zi*freqs[k]*ts[i]) * df
        fs[i] = np.real(fzz)
        cz0 = cz0 + np.abs(np.imag(fzz))
###        wp.prinf('i=',i,1)
###        wp.prinz('fzz=',fzz,1)
###        wp.prin2('cz0=',cz0,1)

    cz0 = cz0 / n

###    wp.prin2('cz0=',cz0,1)

    return(cz0)

#
#
#

def ftr_dumb(fzs,freqs,kfs,fs,ts,n,a,b):
#
#                            description:
#
#    This code evaluates the Fourier transform of f on an
#    equispaced grid between -1/R and 1/R, where R is the radius
#    of the interval (right endpoint 1/R not included).
#    The code is slow and dumb, using a brute force DFT
#    and no vectorization.
#
#
#                        input parameters:
#
#  fs - n-length real vector containing the n equispaced samples of f
#    on [a,b) (only the left endpoint f(a) is included, not f(b))
#  n - the number of samples; assumed to be an even integer (for now)
#  a,b - the endpoints of the interval
#
#
#                        output parameters:
#
#  fzs - n-length complex array containing values of the Fourier transform
#    of f, defined by the convention
#
#          \what{f}[k] = \int_{a}^{b} f(t) \phi_k(t) dt             (1)
#
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#  ts - n-length real vector of equispaced nodes on [a,b) (the left
#    endpoint a is included, but not b)
#
#
    nh2 = np.int(n/2)

#
#    frequencies between -1/R and 1/R, in DFT ordering
#
    for k in range(0,nh2):
        kfs[k] = k

    for k in range(nh2,n):
        kfs[k] = k-n

    wp.prinf('kfs=',kfs,n)

    for k in range(0,n):
        freqs[k] = kfs[k] / (b-a)

    wp.prin2('freqs=',freqs,n)

###    exit()


#
#    real-space grid
#

    for i in range(0,n):
        ts[i] = a + (b-a)*i/n

    dt = (b-a) / n

#
#    Fourier coefficients, via trapezoidal rule (DFT)
#
    for k in range(0,n):
        fzs[k] = 0.0
        for i in range(0,n):
###            fzs[k] = fzs[k] + fs[i] * np.exp(-2*np.pi*1j*k*(ts[i]-a)/(b-a))
            fzs[k] = fzs[k] + fs[i] * np.exp(-2*np.pi*1j*ts[i]*freqs[k])

        fzs[k] = fzs[k] * dt



    return()

#
#
#

def evalmix2d_fast(ts,n,cens,wids,heis,ngaus,fs):
#
#                            description:
#
#    Evaluates samples of 2D gaussian mixture on n-by-n grid.
#    If centers are 3D, it will evaluate the tomographic projection
#    of the 3D mixture onto the x-y plane.
#
#
#                        input parameters:
#
#  ts - length n real array containing the grid points; mixture will be
#    evaluated on points (ts[i],ts[j]), 0 \le i,j \le n-1
#  n - the number of samples on each dimension
#  cens - real array of size ngaus-by-2 or ngaus-by-3
#  wids - length ngaus array containing scale parameters of each gaussian
#  heis - length ngaus array containing heights (weights) of each gaussian
#  ngaus - number of gaussians in the mixture
#
#
#                        output parameters:
#
#  fs - n-by-n array containing the gaussian mixture values
#

    tvec = np.empty(2,dtype='float64')
    ffs0 = np.empty([n,ngaus],dtype='float64')
    ffs1 = np.empty([n,ngaus],dtype='float64')
#
    difs0 = np.empty([n,ngaus],dtype='float64')
    difs1 = np.empty([n,ngaus],dtype='float64')
    gg = np.empty([n,n,ngaus],dtype='float64')


    dpi = np.pi

    fs[0:n,0:n] = 0

#
#    evaluate 1D gaussians
#
    difs0[0:n,0:ngaus] = np.add.outer(ts[0:n] ,-cens[0:ngaus,0])
    difs1[0:n,0:ngaus] = np.add.outer(ts[0:n] ,-cens[0:ngaus,1])

    wp.prinr2('difs0=',difs0,n,ngaus)
    wp.prinr2('difs1=',difs1,n,ngaus)

###    ffs0[0:n,0:ngaus] = np.exp(-np.divide(difs0[0:n,0:ngaus]**2, \
###        wids[None,0:ngaus]))
    ffs0[0:n,0:ngaus] = np.exp(-np.divide(difs0[0:n,0:ngaus]**2, \
        wids[np.newaxis,0:ngaus]))
    ffs0[0:n,0:ngaus] = np.divide(ffs0[0:n,0:ngaus], \
        np.sqrt(dpi*wids[np.newaxis,0:ngaus]))
#
    ffs1[0:n,0:ngaus] = np.exp(-np.divide(difs1[0:n,0:ngaus]**2, \
        wids[np.newaxis,0:ngaus]))
    ffs1[0:n,0:ngaus] = np.divide(ffs1[0:n,0:ngaus], \
        np.sqrt(dpi*wids[np.newaxis,0:ngaus]))


#
#    outer products to get 2D gaussians, and sum
#
    gg[0:n,0:n,0:ngaus] = np.multiply(ffs0[0:n,np.newaxis,0:ngaus], \
        ffs1[np.newaxis,0:n,0:ngaus])

    fs[0:n,0:n] = np.sum(np.multiply(gg[0:n,0:n,0:ngaus], \
        heis[np.newaxis,np.newaxis,0:ngaus]),axis=2)

###    wp.prin2('gg=',gg[0:n,0:n,0:ngaus],n*n*ngaus)


###    wp.prin2('fs=',fs[:,0:n],n*n)


###    wp.prinr2('cens=',cens,ngaus,2)
###    wp.prin2('heis=',heis,ngaus)
###    wp.prin2('wids=',wids,ngaus)


    return()

#
#
#

def slicedlebesg_dumb(gs,hs,ts,n,r,p,nangls,vnorms,kfs,freqs):
#
#                            description:
#
#    This code evaluates the sliced Lebesgue distance between two functions
#    h and g evaluated on a 2D grid of equispaced points. This code is
#    slow and dumb, without vectorization or use of the NUFFT; a fast version,
#    whose output should be identical up to round-off, is slicedlebesg_nufft.
#
#
#                        input parameters:
#
#  hs,gs - n-by-n real arrays containing the samples of h and g
#    on [-r,r) x [-r,r) (only the left endpoints are included)
#  ts - n-length real vector of equispaced nodes on [-r,r) (the left
#    endpoint a is included, but not b)
#  n - the number of samples on each dimension; assumed to be an even integer.
#    To be clear, the total number of samples is n^2
#  r - the radius of the interval
#  p - the norm parameter. To use the infinity norm, set p<=0 or np.inf
#  nangls - the number of projection angles used to evaluate the sliced distance
#
#
#                        output parameters:
#
#  vnorms - nangls-length array containing the distances between each pair
#    of projections
#  kfs - n-length integer array containing the integer frequencies for each
#    Fourier coefficient
#  freqs - n-length real array containing the frequencies for each
#    Fourier coefficient: freqs[k] = kfs[k] / (b-a)
#
#
#                    return parameters:
#
#  dist - the sliced Cramer distance
#
#


    vnorms2 = np.empty(nangls,dtype='float64')
    fprojs_arr2 = np.empty([nangls,n],dtype='float64')
    fhats_arr2 = np.empty([nangls,n],dtype='complex128')
    fphats_arr2 = np.empty([nangls,n],dtype='complex128')
    fps_arr2 = np.empty([nangls,n],dtype='float64')

    angls = np.empty(nangls+1,dtype='float64')
    fprojs_arr = np.empty([nangls,n],dtype='float64')
    fhats_arr = np.empty([nangls,n],dtype='complex128')
    fphats_arr = np.empty([nangls,n],dtype='complex128')
    fps_arr = np.empty([nangls,n],dtype='float64')




    fs = np.empty([n,n],dtype='float64')



    wp.prinf('nangls=',nangls,1)
    wp.prin2('ts=',ts,n+1)
    wp.prin2('r=',r,1)

###    wp.prinr2('fs=',fs,n,6)


    for i in range(0,n):
        for j in range(0,n):
            fs[i,j] = hs[i,j] - gs[i,j]

    rq = n/(4*r)



#
#    frequency indices and values, DFT ordering
#
    nh2=np.int(n/2)

    for k in range(0,nh2):
        kfs[k] = k

    for k in range(nh2,n):
        kfs[k] = k-n

    for k in range(0,n):
        freqs[k] = kfs[k] / (2*r)


    wp.prinf('kfs=',kfs,n)
    wp.prin2('freqs=',freqs,n+1)
    wp.prin2('rq=',rq,1)

#
#    angles grid from 0 to pi
#
    rpi = np.pi

    for i in range(0,nangls+1):
        angls[i] = i*rpi/nangls


    dangl = rpi / nangls

    wp.prin2('angls=',angls,nangls+1)
    wp.prin2('dangl=',dangl,1)


###    exit()



#
#    compute the Lebesgue norm of each projection
#
    for iang in range(0,nangls):
        angl=angls[iang]

        for i in range(0,n):
            sq = np.cos(angl)*freqs[i]
            tq = np.sin(angl)*freqs[i]
            fhats_arr[iang,i] = ftr_2d_fast(sq,tq,fs,ts,n,r)

        cz0 = iftr_fast(fhats_arr[iang,:],freqs,ts,fprojs_arr[iang,:],n)

        vnorms[iang] = lnorm_fourier_dumb(fhats_arr[iang,:],ts,n,\
            -r,r,p,fps_arr[iang,:],fphats_arr[iang,:],kfs,freqs)

###        vnorms2[iang] = vnorm_fourier_dumb(fhats_arr[iang,:],ts,n,\
###            -r,r,p,fps_arr2[iang,:],fphats_arr2[iang,:],kfs,freqs)

###        vnorms[iang] = vnorm_fourier_fast(fhats_arr[iang,:],ts,n,\
###            -r,r,p,fps_arr[iang,:],fphats_arr[iang,:],kfs,freqs)


    wp.prin2('angls=',angls,nangls)
    wp.prin2('vnorms=',vnorms,nangls)

###    exit()
#
#    evaluate sliced norm
#
    vnorm_max = 0.0

    for iang in range(0,nangls):
        if (vnorms[iang] > vnorm_max):
            vnorm_max = vnorms[iang]
    wp.prin2('vnorm_max=',vnorm_max,1)

    if (vnorm_max == 0):
        dist = 0
        return(dist)

    if (p<=0 or p==np.inf):
        dist = vnorm_max
        return(dist)

    dist = 0.0
    for iang in range(0,nangls):
        dist = dist + (vnorms[iang]/vnorm_max)**p / nangls
    dist = dist**(1/p)
    dist = dist*vnorm_max


###    wp.prinr2('fprojs_arr=',fprojs_arr,nangls,6)
    wp.prin2('slnorm, inside=',dist,1)
###    wp.prin2('vnorms=',vnorms,nangls)


    return(dist)

#
#
#

def lnorm_fourier_dumb(fzs,ts,n,a,b,p,fps,fpzs,kfs,freqs):

    cz0=iftr_dumb(fzs,freqs,ts,fps,n)

    fpmax=0.0
    for i in range(0,n):
        if (np.abs(fps[i]) > fpmax):
            fpmax = np.abs(fps[i])


#
#    check if constantly 0, or if p is infinite
#
    if (p <= 0 or p == np.inf):
        vnorm = fpmax
        return(vnorm)
#
    if (fpmax == 0):
        vnorm=0.0
        return(vnorm)


###    wp.prin2('fpmax=',fpmax,1)


#
#    otherwise, find p-norm
#
    dt = (b-a) / n

    vnorm = 0.0
    for i in range(0,n):
        vnorm = vnorm + np.abs(fps[i] / fpmax)**p * dt
    vnorm = vnorm**(1/p)
    vnorm = vnorm * fpmax

###    wp.prin2('vnorm, inside=',vnorm,1)


    return(vnorm)
