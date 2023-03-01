"""
Created on Wed Feb  8 08:43:25 2023
@author: Peret
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import aurora
from aurora import get_atom_data, get_adas_file_types,get_cs_balance_terms
from scipy.interpolate import CubicSpline, PchipInterpolator
import time

#JG: I;m adding a function  helloworld

def hello_world():
    print("hello word")
    
def radiation_plasma():
    pass

def recycling_plasma():
    pass
#Define plasma parameters
param = {"A": 2, "Z": 1, "Ip": 0.7e6, "BT": 1.7, "R": 1.7, "a":0.5, "kappa": 1.8, "lambdaq": 10e-3, "alphaT": 1, "fL_ux":1,\
                "lpol_xt": 0.1, "fx_xt": 4, "fLFS": 0.5, "fcond": 1, "see": 0, "alphaR": 0.5, "alphaU": 0, "Ei": 25, "PSOL": 2e6,\
                "ne_u":2.5e19, "alpha_Vn": 0.7, "alpha_PFR": 0, "lnolq": 0.5, "Er": 25, "Ecx": 7.5, "alphar": 0, "alphacx": 0,\
                "mu0": 4*math.pi*1e-7, "eps0": 1/36/math.pi*1e-9, "qe": 1.602e-19, "mp": 1.67e-27, "me": 9.11e-31, "K0": 2000,\
                "gamma": 0, "Bp": 0, "qcyl": 0, "Lpar_cyl": 0, "L_ux": 0, "L_xt": 0, "alpha_par": 0, "q0": 0, "ngw": 0, "fGW": 0,\
                "mi": 0, "Cs0": 0}
param["gamma"] = 5+2.5*param["alphaT"]
param["Bp"] = param["mu0"]/2/math.pi/param["a"]*param["Ip"]/np.sqrt(0.5*(param["kappa"]**2+1))
param["qcyl"] = param["a"]*param["BT"]*np.sqrt(0.5*(param["kappa"]**2+1))/param["Bp"]/param["R"]
param["Lpar_cyl"] = math.pi*param["qcyl"]*param["R"]
param["L_ux"] = param["fL_ux"] * param["Lpar_cyl"]
param["L_xt"] = param["lpol_xt"]*param["BT"]/param["Bp"]*param["fx_xt"]
param["alpha_par"] = (param["L_ux"]+param["L_xt"])/param["Lpar_cyl"]
param["q0"] = param["fLFS"]*param["BT"]/param["Bp"]*param["PSOL"]/param["lambdaq"]/(2*math.pi)/param["R"]/param["kappa"]
param["ngw"] = param["Ip"]/1e6/math.pi/param["a"]**2*1e20
param["fGW"] = param["ne_u"]/param["ngw"]
param["mi"] = param["A"]*param["mp"]
param["Cs0"] = np.sqrt(param["Z"]*param["qe"]*(1+param["alphaT"])/param["mi"])

#    param.Z         = 1;     % charge unit of ions
#    param.Ip        = 0.5e6; % plasma current in [A]
#    param.BT        = 4;   % toroidal magnetic field at plasma center in [T]
#    param.a         = 0.5;   % minor radius in [m]
#    param.R         = 2.5;   % major radius of plasma center in [m]
#    param.kappa     = 1.4;   % plasma elongation
#    param.lambdaq   = 20e-3;  % heat flux SOL width at upstream in [m]
#    param.alphaT    = 1;     % Ti/Te in SOL
#    param.fL_ux     = 1;     % parallel connection length multiplicator for upstream to X-point
#    param.lpol_xt   = 0.1;   % poloidal length of outer target divertor leg in [m]
#    param.fx_xt     = 4;     % poloidal flux expansion along the outer target divertor leg
#    param.fLFS      = 0.5;   % ratio of SOL power flowing toward the outer divertor target
#    param.fcond     = 1;     % fraction of conducted power 
#    param.see       = 0;     % secondary eigvH1selectron emission coefficien t at target
#    param.alphaR    = 0.5;   % fraction of target ion flux recycled along the outer divertor leg
#    param.alphaU    = 0;     % fraction of neutrals ionized in the upstream SOL
#    param.Ei        = 13.6;    % energy cost of a ionisation  in [eV]
#    param.PSOL      = 1e6; % power entering the SOL in [W] 
#    param.ne_u      = 1e19;  % upstream plasma density in [m-3]
#    param.alpha_Vn  = 10e-2;  %fraction of energy from ions to neutrals
#    param.alpha_PFR = 0;     % fraction of recycling flux going to PFR and recycled in the outer leg
#    param.lnolq     = 0.5;   % ratio between lambdan and lambdaq
#
#    param.mu0=4*pi*1e-7; %USI
#    param.eps0=1/36/pi*1e-9; %USI
#    param.qe=1.602e-19; %C
#    param.mp=1.67e-27; %kg
#    param.me=9.11e-31; %kg
#    param.K0=2000;
#    param.gamma=5+2.5*param.alphaT;  %sheath exhaust factor

# magnetic field and equilibrium
#    param.Bp=param.mu0/2/pi/param.a*param.Ip/sqrt(0.5*(param.kappa.^2+1));     %T
#    param.qcyl=param.a*param.BT*sqrt(0.5*(param.kappa.^2+1))/param.Bp/param.R;  
#    param.Lpar_cyl=pi*param.qcyl*param.R;
#    param.L_ux=param.fL_ux.*param.Lpar_cyl; %m
#    param.L_xt=param.lpol_xt.*param.BT./param.Bp.*param.fx_xt; %m
#    param.alpha_par=(param.L_ux+param.L_xt)./param.Lpar_cyl;
#    param.q0=param.fLFS.*param.BT./param.Bp.*param.PSOL./param.lambdaq./(2*pi)./param.R./param.kappa;
#    param.ngw=param.Ip./1e6./pi./param.a.^2.*1e20;
#    param.fGW=param.ne_u./param.ngw;
#    param.mi=param.A*param.mp; %kg
#    param.Cs0=sqrt(param.Z*param.qe*(1+param.alphaT)/param.mi);


    
    
#Define functions
def f1(x,param):
    return np.sqrt(x)/(1-param["alphaR"]+param["alphar"])*(1-np.sqrt(1-(1-param["alphaR"]+param["alphar"])**(2)/x))


def f2(x,param):
    return param["ne_u"]*(1+param["L_ux"]/param["L_xt"]*(1-x**(7/2)))**(2/7)/(1+(f1(x,param))**2)

def f3(x,param):
    return f2(x,param)/2*(1+(f1(x,param))**2)/x

def f4(x,param):
#% y=2.*x./param.ne_u.*f3(x,param);
    return (1+param["L_ux"]/param["L_xt"]*(1-x**(7/2)))**(2/7)


def f5(x,param):
    beta = (param["Ei"]*(param["alphaR"]+param["alphaU"]*(1-param["alphaR"]))+param["Er"]*(param["alphar"]+param["alphaU"]*(1-param["alphar"]))\
            +param["Ecx"]*(param["alphacx"]+param["alphaU"]*(1-param["alphacx"])))/param["gamma"]

    d = param["q0"]**2*param["mi"]/f3(x,param)**2/param["Z"]/param["qe"]**3/param["gamma"]**2/(1+param["alphaT"])
    ff = (0.5*(3*np.sqrt(3)*np.sqrt(4*beta**3*d+27*d**2)+2*beta**3+27*d))**(1/3) 
    return 1/3*(ff+beta**2/ff-2*beta)



def f6(x,param):
    return x**3-f5(x,param)**3*(2/7)*(1-param["alphaR"]+param["alphar"])*param["K0"]/param["L_xt"]/f1(x,param)/f2(x,param)/\
        param["Cs0"]*(1-x**(7/2))/(param["gamma"]*param["qe"]*f5(x,param)+param["qe"]*param["Ei"]*param["alphaR"]\
                                   +param["qe"]*param["Er"]*param["alphar"]+param["qe"]*param["Ecx"]*param["alphacx"])


def sigvH1se(ne,T):

    # % Ionisation H par electron impact
    B5 = np.zeros(9)
    # %Ionisation Ho reaction 2.1.5   Janev ref 
    # %************************
    B5[0]=-3.271396786375e1
    B5[1]=1.353655609057e1
    B5[2]=-5.739328757388e0
    B5[3]=1.563154982022e0
    B5[4]=-2.877056004391e-1
    B5[5]=3.482559773737e-2
    B5[6]=-2.631976175590e-3
    B5[7]=1.119543953861e-4
    B5[8]=-2.039149852002e-6



    logT1=np.log(T);
    logT2=np.zeros(9)
    for jn in range(9):
        logT2[jn]=logT1**(jn);

    return (np.exp(np.sum(B5*logT2)))*1e-6;
    
def sigvH1se2(ne,T):
    h_data = get_atom_data("H",files=["acd","scd","plt","ccd"])
    a = get_cs_balance_terms(h_data, ne_cm3=[np.squeeze(ne)*1e-6], Te_eV= [np.squeeze(T)], Ti_eV=None, include_cx=True)
    nu_iz = np.squeeze(a[1])
    nu_rec = a[2]
    nu_cx = a[3]  
    return nu_iz / ne, nu_rec/ne, nu_cx/ne

def lmfp(n_t,T_t,param):
    #%lambda_mfp = V_neutral ./ (n_target * sigvH1se(T_target)*1.e-6);
    nu_iz, nu_rec, nu_cx = sigvH1se2(n_t,T_t)
    
    return param["alpha_Vn"]*param["Cs0"]*np.sqrt(T_t)/(n_t*nu_iz), param["Cs0"]*np.sqrt(T_t)/(n_t*nu_rec), param["alpha_Vn"]*param["Cs0"]*np.sqrt(T_t)/(n_t*nu_cx)     
    



def out_3pt_calculator(param):
    #Define ouput sturcture and variables

    out = {"neU": param["ne_u"], "TeU": 0, "neX": 0, "TeX": 0, "MX": 0, "neT": 0, "TeT": 0, "PT": 0, "Prad":0, "nustarU": 0,\
         "nustarX1": 0, "nustarX2": 0, "nustarT": 0, "l_mfp": 0, "l_mfp_rec": 0,"l_mfp_cx": 0}  
        #Calculate the 3pts model predictions for upstream/X-pt/target conditions   
    #print(param["alphaR"],param["alphacx"],param["alphar"])
    x0 = np.arange((1-param["alphaR"]+param["alphar"])**2,1.01,0.001)
    y0 = f6(x0,param)
    #print(param["alphaR"],param["alphacx"],param["alphar"])
    t = np.append(y0[1:],0)
    t = t*y0
    t = t<0
    ind0 = []
    ind0 = np.max(np.squeeze(np.where(t==True)))
    # print(test)
    # for i in range(len(t)):
    #     if t[i]:
    #         ind0 = i

    #find(y0*[y0(2:end);0]<0)
    #% plot(x0,y0)
    #% pause
    #% close
    if not ind0:
        plt.plot(x0,x0**3-y0)
        plt.ylim(-0.01, 0.01)
        ne_u=param["ne_u"]    
        Mx=np.nan
        ne_x=np.nan
        ne_t=np.nan
        Tt=np.nan
        Tx=np.nan
        Tu=np.nan
    else:
        ind = np.max(ind0)
        x = x0[ind]
        ne_u=param["ne_u"]
        Mx=f1(x,param)
        ne_x=f2(x,param)
        ne_t=f3(x,param)
        Tu_oTx=f4(x,param)
        Tt=f5(x,param)
        Tx=Tt/x
        Tu=Tu_oTx*Tx

    #print(x)
    out["neU"] = ne_u
    out["TeU"] = Tu
    out["neX"] = ne_x
    out["TeX"] = Tx
    out["MX"] = Mx
    out["neT"] = ne_t
    out["TeT"] = Tt
    out["PT"] = param["gamma"]*param["qe"]*out["neT"]*param["Cs0"]*out["TeT"]**(3/2)*4*math.pi*\
                param["R"]*param["lambdaq"]*param["Bp"]/param["BT"]
    out["Prad"] = param["PSOL"]-out["PT"]
    #out["Prad2"] = (param["q0"]-param["qe"]*out["neT"]*param["Cs0"]*out["TeT"]**(3/2)*param["gamma"])*param["PSOL"]/param["q0"]
    out["Prad2"] = param["PSOL"]/param["q0"]*out["neT"]*param["Cs0"]*np.sqrt(out["TeT"])*param["qe"]*(param["alphaR"]*param["Ei"]+param["alphacx"]*param["Ecx"]+param["alphar"]*param["Er"])

    out["nustarU"] = param["L_ux"]/(param["K0"]/(param["Z"]*3.16*np.sqrt(param["qe"]/param["me"])*param["qe"]*out["neU"]/out["TeU"]**2))
    out["nustarX1"] = param["L_ux"]/(param["K0"]/(param["Z"]*3.16*np.sqrt(param["qe"]/param["me"])*param["qe"]\
                    *out["neX"]/out["TeX"]**2))
    out["nustarX2"] = param["L_xt"]/(param["K0"]/(param["Z"]*3.16*np.sqrt(param["qe"]/param["me"])*param["qe"]\
                  *out["neX"]/out["TeX"]**2))
    out["nustarT"]=param["L_xt"]/(param["K0"]/(param["Z"]*3.16*np.sqrt(param["qe"]/param["me"])*param["qe"]\
                  *out["neT"]/out["TeT"]**2))


    out["l_mfp"], out["l_mfp_rec"], out["l_mfp_cx"]=lmfp(out["neT"],out["TeT"],param)
    
    out["l_mfp_rec"] = np.squeeze(np.squeeze(out["l_mfp_rec"]))
    out["l_mfp_cx"] = np.squeeze(np.squeeze(out["l_mfp_cx"]))
    
    return out



def param_SF_model(out, param):
    SF_param = {"alpha_g":0.85, "ws": 0, "rhoS": 0, "g": 0, "sig_n":0, "taueT": 0, "sig_PHI": 0, "lambdaq":0, "lambdan": 0,\
                "lambdaT": 0, "tau": 0, "Lambda": 0, "s": 0, "kx": 0, "alphac":0, "sig_res": 0, "kx_mag": 0.6, "kx_corr": 0,\
                "lq_geom": 0.2, "taueX": 0}
    SF_param["lambdaq"] = param["lambdaq"]
    SF_param["lambdaT"] = 7/2*SF_param["lambdaq"]
    SF_param["lambdan"] = param["lnolq"]/SF_param["lambdaq"]
    
    SF_param["ws"]=param["Z"]*param["qe"]*param["BT"]/param["mi"]
    SF_param["rhoS"]=param["Cs0"]*np.sqrt(out["TeU"])/SF_param["ws"]

    SF_param["g"]=SF_param["alpha_g"] * 1.5*SF_param["rhoS"]/param["R"]
    
    #SF_param["sig_N"]=2*SF_param["rhoS"]/param["L_ux"]*out["neX"]/out["neU"]*out["MX"]*np.sqrt(out["TeX"]/out["TeU"])
    SF_param["sig_N"]=SF_param["rhoS"]/param["L_ux"]*out["MX"]*np.sqrt(out["TeX"]/out["TeU"])*out["neX"]/out["neU"]
    
    
    SF_param["tau"]=0.43*np.sqrt(SF_param["lambdaT"]/SF_param["g"]/SF_param["rhoS"])
    SF_param["Lambda"]=-0.5*np.log(0.5*param["me"]/param["mi"]*(1+param["alphaT"]))
    SF_param["s"]=SF_param["Lambda"]*SF_param["rhoS"]**2/SF_param["lambdaT"]**2
    
    SF_param["kx"]=SF_param["s"]*SF_param["tau"] + SF_param["kx_mag"];
    alpha_s = SF_param["kx"]*SF_param["s"]*SF_param["tau"]/np.sqrt(1+SF_param["kx"]**2)
    SF_param["kx_corr"]=(1+SF_param["kx"]**2)**(-9/11)*(1+1.13*alpha_s-0.01*alpha_s**2)**(-4/11)
    
#%alphac=14.*(out.neT.*1e-19)./out.TeT.^(2).*out.param.L_xt./out.param.Z.^2;
    SF_param["taueT"]=3*(2*math.pi)**(3/2)*param["eps0"]**2*np.sqrt(param["me"])\
         /15/param["qe"]**4/out["neT"]*(param["qe"]*out["TeT"])**(3/2)

    # SF_param["sig_res"]=np.sqrt(1836)*SF_param["ws"]*SF_param["taueT"]*SF_param["rhoS"]**2/param["L_ux"]**2;
    # alphac=1/0.51*param["L_xt"]*param["Cs0"]*np.sqrt(out["TeT"])/(param["qe"]/param["me"]*out["TeT"])/SF_param["taueT"]
    # SF_param["sig_PHI"]=2*SF_param["rhoS"]/param["L_ux"]*out["neX"]/out["neU"]*(out["TeU"]/out["TeT"])**(1/2)/(1+alphac)
    
    SF_param["taueX"]=3*(2*math.pi)**(3/2)*param["eps0"]**2*np.sqrt(param["me"])\
        /15/param["qe"]**4/out["neX"]*(param["qe"]*out["TeX"])**(3/2)
    SF_param["sig_res"]=np.sqrt(1836)*SF_param["ws"]*SF_param["taueX"]*SF_param["rhoS"]**2/param["L_ux"]**2;
    alphac=1/0.51*param["L_xt"]*param["Cs0"]*np.sqrt(out["TeT"])/(param["qe"]/param["me"]*out["TeT"])/SF_param["taueT"]*out["TeU"]/out["TeT"]
    SF_param["sig_PHI"]=SF_param["rhoS"]/param["L_ux"]*np.sqrt(out["TeU"]/out["TeT"])/(1+alphac)
    
    
   
    SF_param["lambdan"]=3.9*SF_param["kx_corr"]*SF_param["lq_geom"]*SF_param["g"]**(3/11)\
        *SF_param["sig_N"]**(-4/11)*SF_param["sig_PHI"]**(-2/11)*SF_param["rhoS"]
        
    SF_param["lambdaq"] = SF_param["lambdan"]/(1+3/7/param["lnolq"]/(1+alphac)**(2/11))
   

    SF_param["lambdaT"] = 7/2 * param["lnolq"] * SF_param["lambdan"]*(1+alphac)**(2/11)

      
    
    # SF_param["lambdaq"]=param["lnolq"]*3.9*SF_param["kx_corr"]*SF_param["lq_geom"]*SF_param["g"]**(3/11)\
    #     *SF_param["sig_N"]**(-4/11)*SF_param["sig_PHI"]**(-2/11)*SF_param["rhoS"]        
    # SF_param["lambdan"]=1/param["lnolq"]*SF_param["lambdaq"]
    # SF_param["lambdaT"] = 7/2* * SF_param["lambdaq"]  

     
    SF_param["tau"]=0.43*np.sqrt(SF_param["lambdaT"]/SF_param["g"]/SF_param["rhoS"])
    SF_param["alphac"]=1+alphac;
    return SF_param


def alphaR_cal(out,param):
    alphaR=1-2/math.pi*np.arctan(out["l_mfp"]/param["lpol_xt"])
    #alphaR=1-2/pi*atan(out.l_mfp./(out.param.fx_xt.*out.param.lambdaq));
    #alphaR=1-0.5*2/math.pi*(np.arctan(np.sqrt(out["l_mfp"]**2+(param["fx_xt"]*param["lambdaq"])**2)\
    #                               /param["lpol_xt"]+np.arctan(out["l_mfp"]/(param["lpol_xt"]))))
    alphaR=param["alpha_PFR"]+alphaR*(1-param["alpha_PFR"])
    #alphaR = param["alphaR"]+1/1000*(alphaR-param["alphaR"])
    return alphaR

def alphar_cal(out,param):
    alphar=1-2/math.pi*np.arctan(out["l_mfp_rec"]/param["lpol_xt"])
    #alphaR=1-2/pi*atan(out.l_mfp./(out.param.fx_xt.*out.param.lambdaq));
    #alphaR=1-0.5*2/math.pi*(np.arctan(np.sqrt(out["l_mfp"]**2+(param["fx_xt"]*param["lambdaq"])**2)\
    #                               /param["lpol_xt"]+np.arctan(out["l_mfp"]/(param["lpol_xt"]))))
    alphar=param["alpha_PFR"]+alphar*(1-param["alpha_PFR"])
    #alphar = param["alphar"]+1/1000*(alphar-param["alphar"])
    
    return alphar

def alphacx_cal(out,param):
    alphacx=1-2/math.pi*np.arctan(out["l_mfp_cx"]/param["lpol_xt"])
    #alphaR=1-2/pi*atan(out.l_mfp./(out.param.fx_xt.*out.param.lambdaq));
    #alphaR=1-0.5*2/math.pi*(np.arctan(np.sqrt(out["l_mfp"]**2+(param["fx_xt"]*param["lambdaq"])**2)\
    #                               /param["lpol_xt"]+np.arctan(out["l_mfp"]/(param["lpol_xt"]))))
    alphacx=param["alpha_PFR"]+alphacx*(1-param["alpha_PFR"])
    #alphacx = param["alphacx"]+1/1000*(alphacx-param["alphacx"])

    return alphacx
    
#Need a loop to converge
i_max = 30
j_max = 41

lq = []
neU=[]
g = []
sig_PHI = []
sig_N = []
alphaR = []
alphar = []
alphacx = []
l_mfp = []
l_mfp_rec =[]
l_mfp_cx = []
Gamma_t = []
f_GW = []
TeU = []
TeX = []
TeT = []
MX = []
neU = []
neX = []
neT = []
frad = []
q0 = []

deltan = 2* 0.0056e19

for j in range(j_max):
    t = time.time()
    print(j+1, '/', j_max)
    param["ne_u"] = 0.6e19+deltan*(j)
    
    if (j<2):
        param["alphaR"] = 0.95
        param["alphar"] = 0
        param["alphacx"] = 0.95
        param["lambdaq"] = 1e-2
    else:
        splineR =PchipInterpolator(neU, alphaR)
        spliner =PchipInterpolator(neU, alphar)
        splinecx =PchipInterpolator(neU, alphacx)
        splinelq =PchipInterpolator(neU, lq)
        # splineR =CubicSpline(neU, alphaR)
        # spliner =CubicSpline(neU, alphar)
        # splinecx =CubicSpline(neU, alphacx)
        # splinelq =CubicSpline(neU, lq)        
        
        param["alphaR"] = splineR(param["ne_u"])
        param["alphar"] = spliner(param["ne_u"])
        param["alphacx"] = splinecx(param["ne_u"])
        param["lambdaq"] = splinelq(param["ne_u"])

        
    for i in range(i_max):
        out = out_3pt_calculator(param)
        SF_param = param_SF_model(out, param)
        param["lambdaq"] = SF_param["lambdaq"]
        param["q0"] = param["fLFS"]*param["BT"]/param["Bp"]*param["PSOL"]/param["lambdaq"]/(2*math.pi)/param["R"]/param["kappa"]
        param["alphaR"] = alphaR_cal(out,param)
        param["alphar"] = alphar_cal(out,param)
        param["alphacx"] = alphacx_cal(out,param)  
        

        
        
    # lq = np.append(lq, param["lambdaq"])
    # g = np.append(g, SF_param["g"])
    # sig_N = np.append(sig_N, SF_param["sig_N"])
    # sig_PHI = np.append(sig_PHI, SF_param["sig_PHI"])

    lq = np.append(lq, param["lambdaq"])
    alphaR = np.append(alphaR, param["alphaR"])
    alphar = np.append(alphar, param["alphar"])
    alphacx = np.append(alphacx, param["alphacx"])
    neU = np.append(neU, param["ne_u"])
    g = np.append(g, SF_param["g"])
    sig_N = np.append(sig_N, SF_param["sig_N"])
    sig_PHI = np.append(sig_PHI, SF_param["sig_PHI"])
    l_mfp = np.append(l_mfp, out["l_mfp"])
    l_mfp_rec = np.append(l_mfp_rec, out["l_mfp_rec"])
    l_mfp_cx = np.append(l_mfp_cx, out["l_mfp_cx"])
    Gamma_t = np.append(Gamma_t, param["qe"]*out["neT"]*param["Cs0"]*np.sqrt(out["TeT"]))
    f_GW = np.append(f_GW, 4*param["ne_u"]/param["ngw"])
    TeU = np.append(TeU, out["TeU"])
    TeX = np.append(TeX, out["TeX"])
    TeT = np.append(TeT, out["TeT"])
    neX = np.append(neX, out["neX"])
    neT = np.append(neT, out["neT"])
    MX = np.append(MX, out["MX"])
    frad = np.append(frad, out["Prad2"]/param["PSOL"]*100)
    q0 = np.append(q0, param["q0"])
    
    print("Done in ", time.time()-t, "s")

plt.figure()
plt.plot(f_GW,lq*1000,'ob')
plt.xlabel("Greenwald fraction")
plt.ylabel("$\lambda_{q} [mm]$")
plt.show()

plt.figure()
plt.plot(f_GW,alphaR,'ob')
plt.plot(f_GW,alphar,'or')
plt.plot(f_GW,alphacx,'og')
plt.xlabel("Greenwald fraction")
plt.ylabel("alpha")
plt.legend(["ionisation", "recombination", "CX"])
plt.show()

plt.figure()
plt.plot(f_GW,g/100,'+b')
plt.plot(f_GW,sig_N,'or')
plt.plot(f_GW,sig_PHI,'ok')
plt.xlabel("Greenwald fraction")
plt.ylabel("SSF params")
plt.legend(["g/100", "$\sigma_{\parallel}^{N}$", "$\sigma_{\parallel}^{\Phi}$"])
plt.show()

plt.figure()
plt.plot(f_GW, Gamma_t, "ko")
plt.xlabel("Greenwald fraction")
plt.ylabel("target particle flux")
plt.show()

plt.figure()
plt.plot(f_GW, param["lpol_xt"]/l_mfp, "bo")
plt.plot(f_GW, param["lpol_xt"]/l_mfp_rec, "ro")
plt.plot(f_GW, param["lpol_xt"]/l_mfp_cx, "go")
plt.xlabel("Greenwald fraction")
plt.show()

plt.figure()
plt.plot(f_GW, l_mfp, "bo")
plt.plot(f_GW, l_mfp_rec, "ro")
plt.plot(f_GW, l_mfp_cx, "go")
plt.xlabel("Greenwald fraction")
plt.show()

plt.figure()
plt.plot(f_GW, TeU, "ko")
plt.plot(f_GW, TeX, "ro")
plt.plot(f_GW, TeT, "bo")
plt.xlabel("Greenwald fraction")
plt.show()

plt.figure()
plt.plot(f_GW, TeU/TeX, "ko")
plt.plot(f_GW, TeX/TeX, "ro")
plt.plot(f_GW, TeT/TeX, "bo")
plt.xlabel("Greenwald fraction")
plt.show()

plt.figure()
plt.plot(f_GW, neU, "ko")
plt.plot(f_GW, neX, "ro")
plt.plot(f_GW, neT, "bo")
plt.xlabel("Greenwald fraction")
plt.show()

plt.figure()
plt.plot(f_GW, MX, "ko")
plt.plot(f_GW,(1-alphaR+alphar)/2/np.sqrt(TeT/TeX),'ob')
#plt.plot(f_GW,np.sqrt(TeT/TeX),'ob')
plt.xlabel("Greenwald fraction")
plt.show()

plt.figure()
plt.plot(f_GW, 2*neT*TeT/(neU*TeU)-1,'ko')
plt.plot(f_GW, neX*TeX*(1+MX**2)/(neU*TeU)-1,'bo')
plt.plot(f_GW, neT*np.sqrt(TeT)*(1-alphaR+alphar)/(neX*np.sqrt(TeX)*MX)-1,'go')
plt.plot(f_GW, 2/7*param["K0"]*(TeX**(7/2)-TeT**(7/2))/param["L_xt"]/q0-2/7*param["K0"]*(TeU**(7/2)-TeX**(7/2))/param["L_ux"]/q0,'mo')
plt.plot(f_GW, TeU/TeX/(1+(7*q0*param["L_ux"])/(2*param['K0']*TeX**(7/2)))**(2/7)-1,'ro')

#plt.plot(f_GW, 2/7*param["K0"]*(TeU**(7/2)-TeX**(7/2))/param["L_ux"]/param["q0"]-1,'co')
plt.plot(f_GW, neT*param["Cs0"]*np.sqrt(TeT)*(param["gamma"]*param["qe"]*TeT+alphaR*param["qe"]*param["Ei"]+\
                                              alphar*param["qe"]*param["Er"]+alphacx*param["qe"]*param["Ecx"])/q0-1,'yo')
plt.xlabel("Greenwald fraction")
plt.show()


plt.figure()
plt.plot(f_GW, frad, "ko")
plt.xlabel("Greenwald fraction")
plt.show()

#print(SF_param["kx_corr"])
#print(param["alphaR"])
#print(param["fGW"])
#print(SF_param["lambdan"])