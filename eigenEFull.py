# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:26:26 2019

@author: Will Clare

Main program for calculating the eigen-solutions. Use `eigenEF` to generate these.
"""

import numpy as np
from numpy import linalg as LA
import diff
import matplotlib.pyplot as plt
import betafind
import LaplaceCoefficients
LC = LaplaceCoefficients.LC
diffm = diff.diffm
exp = np.exp
pi = np.pi


#These are constants relating to the resonances.
# Beginning with I are for inner resonances. Beginning with O are for outer resonances.
IAs = np.array([0, 0, 0, 0.8332, 2.1863, 3.8918, 5.889, 8.4106, 10.621, 13.3108, 16.195])
OAs = np.array([0,0,0, 0.6072,5.2009,7.3617,9.7628,12.3809,15.1989,18.2033,21.3831])
IBs = np.array([0,0,0,1.5397,3.2019,5.1463,7.3456,9.7756,12.4173,15.2555,18.2777])
OBs = np.array([0,0,0,1.849,3.594,5.6037,7.8592,10.3395,13.0272,15.9081,18.9703])

IC = np.array([0,0,1.0853,2.2367,3.3933,4.5517,5.7109,6.8705,8.0304,9.1905])
OC = np.array([0,0,1.7229,2.9309,4.1107,5.2817,6.4489,7.6141,8.7780,9.9412])
ID = np.array([0,0,0.3906,2.7434,3.9223,5.0930,6.2601,7.4252,8.5890,9.7522])
OD = np.array([0,0,0.6200,3.5949,4.7515,5.9099,7.0692,8.2288,9.3887,10.5488])

ICM = np.array([0,0,0.3365,1.1819,1.1265,1.0970,1.0787,1.0663,1.0572,1.0503])
OCM = np.array([0,0,0.7422,0.8418,0.8854,0.9101,0.9261,0.9372,0.9454,0.9517])

J = np.array([0,1,2,3,4,5,6,7,8,9])
IWC = 4.1*(ICM*(J-1)*0.01)**(1/2)
OWC = 4.1*(OCM*(J-1)*0.01)**(1/2)


def rank(v,times=1,):
    '''Rank the indices from lowest to highest in terms of derivative.
    
    Can be useful for determining non-spurious eigenvectors as these will typically have lower derivatives through continuity.'''
    fn = v
    for i in range(times):
        fn = np.diff(fn, axis=0)
    
    return np.argsort(np.max(abs(fn), axis=0))



def eigenEF(rin,
            rout,  
            eps,
            n=1000,
            a=0,
            qd=0,
            q=0,
            ap=None,
            is3d=True,
            **kwargs):
    '''Calculate the eccentric modes for a disc with `rin`, `rout`, `eps` aspect ratio, `n` steps, `a` bulk viscosity, `qd` disc mass ratio, `q` planet mass ratio, `ap` planet semi-major axis.
    
    It is also possible to pass in a pre-defined univariate functions s and cs - being the surface density and isothermal sound speed respectively. Make sure these have a `**kwargs` argument.
    
    3D can be toggled with `is3d` (default `True`).
    
    By default, this will calculate the 5 lowest modes, but this is not a perfect operation and can be disabled by `plot=False`.
    
    Returns the 5-lowest modes `E`, the mode of the planet `Ep`, the corresponding eigenvalues `w1` to the modes, the `r` coordinates, the full eigenvalues `w`, and the full eigenvectors `v`'''
    if q != 0:
        assert ap > 0
    X = np.linspace(np.log(rin),np.log(rout),n)
    r = exp(X)
    dX = X[1]-X[0]
    beta = betafind.beta1(r,eps)
    
    # Difference matrix
    D = np.zeros(shape=(n,n))
    for i in range(n-1):
        D[i,i] = -1/dX
        D[i,i+1] = 1/dX
    D[n-1,:] = D[n-2,:]
    D2 = np.zeros(shape=(n,n))
    for i in range(1,n-1):
        D2[i,i-1] = 1/dX**2    #-
        D2[i,i] = -2*1/dX**2
        D2[i,i+1] = 1/dX**2
    D2[0,0] = -2*1/dX**2
    D2[0,1] = 1/dX**2
    D2[n-1,n-1] = -2*1/dX**2
    D2[n-1,n-2] = 1/dX**2
    
    # Distributions
    # You can import your own surface density and sound speed distribution. The vars dictionary contains the useful physical information from the code.
    Vars = {'rin' : rin,
             'rout' : rout,
             'eps' : eps,
             'a' : a,
             'qd' : qd,
             'q' : q,
             'ap' : ap,
             'r' : r,
           }
    
    s_func = kwargs.get('s')
    if s_func == None:
        s = r**(-1/2)*(1 - (rin/r)**(1/2))**(5/9)*np.tanh((rout-r)/(0.01*rout))
    else:
        s = s_func(**Vars)
    
    cs_func = kwargs.get('cs')
    if cs_func == None:
        cs = eps*1/r**(1/2)*(1-(rin/r)**(1/2))**(2/9)
    else:
        cs = cs_func(**Vars)
        
    
    # Make M matrix
    # Terms:
    # A1,B1,C1,D1,E1 2D pressure effects
    # F1 3D pressure effects
    # G1 Perturbing effects from companion
    # H viscous effects from alpha_b
    # I Self-gravity
    # R resonances
    M = np.zeros(shape=(n+1,n+1),dtype=complex)
    m = np.zeros(shape=(n))
    LR = np.zeros(shape=(n,11))
    PLR = np.zeros(shape=(n,11))
    CR = np.zeros(shape=(n,11))
    PCR = np.zeros(shape=(n,11))
    IPL = np.zeros(shape=(n,11),dtype=complex)
    OPL = np.zeros(shape=(n,11),dtype=complex)
    PL1 = np.zeros(shape=(11),dtype=complex)
    IPC = np.zeros(shape=(n,11),dtype=complex)
    OPC = np.zeros(shape=(n,11),dtype=complex)
    PC1 = np.zeros(shape=(11),dtype=complex)
    
    #Calculate the Resonance contributions
    ##Lindblad
    for m in range(3,11):
        if q == 0:
            break
        # inner
        Ires = ((m-2)/m)**(2/3)*ap
        Iw = r*(eps**2/(3*(m-2)))**(1/3)
        Idel = (2*pi)**(-1/2)*exp(-((r-Ires)/Iw+1)**2/2)
        # outer
        Ores = ((m-2)/m)**(-2/3)*ap
        Ow = r*(eps**2/(3*(m)))**(1/3)
        Odel = (2*pi)**(-1/2)*exp(-((r-Ores)/Ow-1)**2/2)
        for i in range(n-1):
            ILR = -s[i]*IAs[m]**2*r[i]**(-1)*Iw[i]**(-1)*Idel[i]
            OLR = -s[i]*OAs[m]**2*r[i]**(-1)*Ow[i]**(-1)*Odel[i]
            LR[i,m] = ILR + OLR
            
            PILR = s[i]*IAs[m]*IBs[m]*r[i]**(-1)*Iw[i]**(-1)*Idel[i]
            POLR = s[i]*OAs[m]*OBs[m]*r[i]**(-1)*Ow[i]**(-1)*Odel[i]
            PLR[i,m] = PILR + POLR
            
        IPL1 = -4j*pi/ap*IBs[m]**2*sum(s[k]/Iw[k]*r[k]**2*Idel[k] for k in range(n))*dX
        for k in range(n):
            IPL[k,m] = 4j*pi/ap*IBs[m]*IAs[m]*s[k]/Iw[k]*r[k]**2*Idel[k]*dX
            OPL[k,m] = 4j*pi/ap*OBs[m]*OAs[m]*s[k]/Ow[k]*r[k]**2*Odel[k]*dX
        PL = IPL + OPL
        OPL1 = -sum(OPL[k,m] for k in range(n))*OBs[m]/OAs[m]
        PL1[m] = IPL1 + OPL1
            
    ##Corotation
    for m in range(2,10):
        if q == 0:
            break
        # inner
        Ires = ((m-1)/m)**(2/3)*ap
        Iw = r*q**(1/2)*IWC[m]
        Idel = (2*pi)**(-1/2)*exp(-((r-Ires)/Iw)**2/2)
        # outer
        Ores = ((m-1)/m)**(-2/3)*ap
        Ow = r*q**(1/2)*OWC[m]
        Odel = (2*pi)**(-1/2)*exp(-((r-Ores)/Ow)**2/2)
        for i in range(1,n-2):
            ICR = -s[i]*IC[m]**2*r[i]**(-1)*Iw[i]**(-1)*Idel[i]*(sum(D[i,k]*np.log(s[k]) for k in [i,i+1]) + 3/2)
            OCR = s[i]*OC[m]**2*r[i]**(-1)*Ow[i]**(-1)*Odel[i]*(sum(D[i,k]*np.log(s[k]) for k in [i,i+1]) + 3/2)
            CR[i,m] = ICR + OCR
            
            PICR = -ICR*ID[m]/IC[m]
            POCR = -OCR*OD[m]/OC[m]
            PCR[i,m] = PICR + POCR
        ICR = -s[n-2]*IC[m]**2*r[n-2]**(-1)*Iw[n-2]**(-1)*Idel[n-2]*(sum(D[n-3,k]*np.log(s[k]) for k in [n-3,n-2]) + 3/2)
        OCR = s[n-2]*OD[m]**2*r[n-2]**(-1)*Ow[n-2]**(-1)*Odel[n-2]*(sum(D[n-3,k]*np.log(s[k]) for k in [n-3,n-2]) + 3/2)
        CR[n-2,m] = ICR + OCR
        
        PICR = -ICR*ID[m]/IC[m]
        POCR = -OCR*OD[m]/OC[m]
        PCR[n-2,m] = PICR + POCR
        
        for i in range(n-2):
            if s[i] > 10**(-10):
                IPC[i,m] = 4j*pi/ap*(sum(D[i,k]*np.log(s[k]) for k in [i,i+1]) + 3/2)*ID[m]*IC[m]*s[i]*r[i]**2/Iw[i]*Idel[i]
                OPC[i,m] = - 4j*pi/ap*(sum(D[i,k]*np.log(s[k]) for k in [i,i+1]) + 3/2)*OD[m]*OC[m]*s[i]*r[i]**2/Ow[i]*Odel[i]
            
        PC = IPC + OPC
        IPC1 = -sum(IPC[i,m] for i in range(n))*ID[m]/IC[m]
        OPC1 = -sum(OPC[i,m] for i in range(n))*OD[m]/OC[m]
        PC1[m] = IPC1 + OPC1
    
                
    for i in range(n-1):
        for j in range(i-1,i+2):  #n
            A1 = 1/r[i]**3*D[i,j]*sum(D[i,k]*s[k]*cs[k]**2*r[k]**2 for k in range(n))
            B1 = s[i]*cs[i]**2*r[i]**(-1)*D2[i,j]
            if i==j:
                C1 = 1/r[i]*sum(D[i,k]*s[k]*cs[k]**2 for k in range(n))
                E1 = -sum(D2[i,k]*cs[k]**2 for k in range(n))*s[i]*r[i]**(-1)
                if is3d:
                    F1 = 3/r[i]**3*s[i]*sum(D[i,k]*cs[k]**2*r[k]**2 for k in range(n))
                else:
                    F1 = 0
                H1 = (-3*cs[i]**2/r[i] + sum(D[i,k]*cs[k]**2 for k in range(n)))*s[i]
                H4 = sum(3*D[i,k]*cs[k]**2*s[k] for k in range(n))
                H6 = s[i]*sum(D[i,k]*cs[k]**2 for k in range(n))
                H7 = r[i]*s[i]*sum(D2[i,k]*cs[k]**2 for k in range(n))
                H9 = r[i]*sum(D[i,:]*cs[:]**2)*sum(D[i,:]*s[:])
                if q != 0:
                    G1 = s[i]/(2*ap**2)*LC(3/2,1,betafind.beta(r[i],ap,0))
                I1 = pi*r[i]**(-3/2)*s[i]*sum(s[k]*beta[i,k]**(3/2)*r[k]**(3/2)*LC(3/2,1,beta[i,k]) for k in range(n))*dX
                LR1 = 0
                CR1 = 0
                for m in range(3,11):
                    LR1 = LR1 + 2j*q**2*LR[i,m]
                    if m != 3:
                        CR1 = CR1 + 2j*q**2*CR[i,m-1]
            else:
                C1 = 0
                E1 = 0
                F1 = 0
                H1 = 0
                H4 = 0
                H6 = 0
                H7 = 0
                H9 = 0
                G1 = 0
                I1 = 0
                LR1 = 0
                CR1 = 0
            D1 = -1/r[i]**3*D[i,j]*s[j]*r[j]**2*sum(D[i,k]*cs[k]**2 for k in range(n))
            H2 = r[i]*D[i,j]*sum(D[i,k]*cs[k]**2*s[k] for k in range(n))
            H3 = r[i]*cs[i]**2*s[i]*D2[i,j]
            H5 = 3*cs[i]**2*s[i]*D[i,j]
            H8 = r[i]*s[i]*D[i,j]*sum(D[i,k]*cs[k]**2 for k in range(n))
            I2 = -pi*r[i]**(-3/2)*s[i]*beta[i,j]**(3/2)*r[j]**(3/2)*s[j]*LC(3/2,2,beta[i,j])*dX
            
            if q == 0:
                G = 0
            else:
                G = q*G1
            H = -1j*a*(H1 + H2 + H3 + H4 + H5 + H6 + H7 + H8
             + H9
             )
            
            # Below is 2D version of viscosity 
            if not is3d:
                H1 = sum(D[i,k]*cs[k]**2*s[k]*r[k]**3 for k in range(n))*D[i,j]
                H2 = cs[i]**2*s[i]*r[i]**3*D2[i,j]
                H = -1j/r[i]**2*a*(H1 + H2)
            
            I = qd*(I1 + I2)
            # I = qd*I1
            if s[i]>1e-10:
                M[i,j] = (A1 + B1 + C1 + D1 + E1 + F1 + G + H + I + LR1 + CR1)/(2*s[i])*r[i]**(1/2)
                
        for j in range(n):
            I2 = -pi*r[i]**(-3/2)*s[i]*beta[i,j]**(3/2)*r[j]**(3/2)*s[j]*LC(3/2,2,beta[i,j])*dX
            if s[i] > 1e-10:
                M[i,j] = M[i,j] + I2*qd/(2*s[i])*r[i]**(1/2)
            
    
    # Now deal with planets effects and rows of matrix
    for i in range(1,n-1):
        if q == 0:
            break
        PLR1 = 2j*q**2*sum(PLR[i,m] for m in range(3,11))
        PCR1 = 2j*q**2*sum(PCR[i,m] for m in range(2,10))
        G2 = -1/2*ap**(-2)*q*s[i]*LC(3/2,2,ap/r[i])  #betafind.beta(r[i],ap,eps)
        M[i,n] = (PLR1 + PCR1 + G2)/(2*s[i])*r[i]**(1/2)

    if q != 0:
        BETA = ap/r          #betafind.beta(r,ap,eps)
        PG1 = q**(-1/2)*qd*pi/ap**(3/2)*sum(s[k]*BETA[k]**(3/2)*r[k]**(3/2)*LC(3/2,1,BETA[k]) for k in range(n))*dX
        PL2 = q**(1/2)*qd*sum(PL1[m] for m in range(3,11))
        PC2 = q**(1/2)*qd*sum(PC1[m] for m in range(2,10))
        M[n,n] = (PG1 + PL2 + PC2)/2*ap**(1/2)
    
    for i in range(n-1):
        if q == 0:
            break
        PLR2 = q**(1/2)*qd*sum(PL[i,m] for m in range(3,11))
        PCR2 = q**(1/2)*qd*sum(PC[i,m] for m in range(2,10))
        PG2 = -q**(-1/2)*qd*pi/ap**(3/2)*s[i]*BETA[i]**(3/2)*r[i]**(3/2)*LC(3/2,2,BETA[i])*dX
        M[n,i] = (PLR2 + PG2 + PCR2)/2*ap**(1/2)
    
    #Now find the eigenvalues and vectors
    w, v = LA.eig(M)
    
    v = np.delete(v,[n-2,n-1],axis=1)
    w = np.delete(w,[n-2,n-1])
    
    for p in range(n-2):
        v[0,p] = v[1,p]
        v[n-1,p] = v[n-2,p]
    
    # Make max(abs(v))=1 for plotting
    v = v / np.max(abs(v), axis=1)
    Ep = v[n,:]
    v = np.delete(v,[n],axis=0)
    
    # Checking number of zeros
    l = np.zeros(len(v[0]))
    for k in range(len(l)):
        for i in range(len(l)-1):
            if v[i,k] > 0 and v[i+1,k] < 0:
                l[k] = l[k] + 1
            if v[i,k] < 0 and v[i+1,k] > 0:
                l[k] = l[k] + 1
                
    # Checking how often the absolute value decreases
    p = np.zeros(len(l))
    for k in range(len(l)):
        for i in range(len(l)-1):
            if abs(v[i,k]) > abs(v[i+1,k]):
                p[k] = p[k] + 1
    
    # Finding lowest order mode
    E = np.zeros(shape=(n,5))
    m = np.zeros(5,dtype=int)
    for i in range(len(m)):
        m[i] = np.argmin(abs(l-i))
    if min(l) == 0:
        m[0] = np.argmin(l)
    else:
        m[0] = np.argmax(p)
        if m[0] == np.argmin(l):
            m[0] = np.argmax(np.delete(p,np.argmax(p)))
    
    E = np.column_stack((v[:,m[0]],v[:,m[1]],v[:,m[2]],v[:,m[3]],v[:,m[4]]))
    w1 = np.array([w[m[0]],w[m[1]],w[m[2]],w[m[3]],w[m[4]]])
    
    #Plots
    plot = kwargs.get('plot')
    if plot is None:
        plot = True
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(r,abs(E[:,0]),label='0')
        ax.plot(r,abs(E[:,1]),label='1')
        ax.plot(r,abs(E[:,2]),label='2')
        ax.plot(r,abs(E[:,3]),label='3')
        ax.plot(r,abs(E[:,4]),label='4')
        ax.set_xscale('log')
        ax.set_xlabel('r/au')
        ax.set_ylabel('|E|')
        ax.set_ylim(0,1)
        ax.set_xlim(rin,rout)
        ax.legend(loc='upper right',title='Modes')
        plt.show()
    
    
    return E, Ep, w1, r, w, v