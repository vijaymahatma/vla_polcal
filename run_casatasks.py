# This module generates the full CASA python script as a string and saves it.
fluxdict = {}
def generate_casa_script(
    msin, msout, msout_target,
    all_spws, band_spws, final_spws,
    sources, refant, band,
    cal_leakage, cal_leakage_im, cal_leakage_newgains,
    cal_leakage_I_model, cal_leakage_I_model_spix, cal_leakage_ref_freq,
    cal_polangle, cal_polangle_ref_freq,
    skip_initcal, baseband_list,
    solve_bbc1, solve_bbc2, solve_bbc3, band_ref, target
):
    """
    Returns a complete CASA script string.
    """

    return f"""
from casatasks import gaincal, applycal, setjy, flagdata, polcal, split, statwt
import numpy as np
import matplotlib.pyplot as plt

# --- parameter values injected by Python ---
msin = '{msin}'
msout = '{msout}'
msout_target = '{msout_target}'
all_spws = '{all_spws}'
band_spws = '{band_spws}'
final_spws = '{final_spws}'
refant = '{refant}'
cal_leakage = '{cal_leakage}'
cal_leakage_im = '{cal_leakage_im}'
cal_leakage_newgains = '{cal_leakage_newgains}'
cal_leakage_I_model = {cal_leakage_I_model}
cal_leakage_I_model_spix = {cal_leakage_I_model_spix}
cal_leakage_ref_freq = '{cal_leakage_ref_freq}'
cal_polangle = '{cal_polangle}'
cal_polangle_ref_freq = '{cal_polangle_ref_freq}'
skip_initcal = {skip_initcal}
baseband_list = {baseband_list}
solve_bbc1 = '{solve_bbc1}'
solve_bbc2 = '{solve_bbc2}'
solve_bbc3 = '{solve_bbc3}'
band_ref = {band_ref}
band = '{band}'
sources = '{sources}'
target = '{target}'

# --------------- CASA LOGIC HERE ---------------
# (paste your entire CASA script block unchanged)
# -----------------------------------------------
import sys

if {skip_initcal}:
    pass
else:
    applycal(vis='{msin}', field='{sources}', intent='CALIBRATE_POL_ANGLE#UNSPECIFIED,SYSTEM_CONFIGURATION#UNSPECIFIED,OBSERVE_TARGET#UNSPECIFIED,CALIBRATE_POL_LEAKAGE#UNSPECIFIED,CALIBRATE_BANDPASS#UNSPECIFIED,CALIBRATE_AMPLI#UNSPECIFIED,CALIBRATE_PHASE#UNSPECIFIED', 
    spw='{all_spws}', antenna='0~26', gaintable=['{msin}.hifv_priorcals.s5_2.gc.tbl', 
    '{msin}.hifv_priorcals.s5_3.opac.tbl', 
    '{msin}.hifv_priorcals.s5_4.rq.tbl', 
    '{msin}.hifv_finalcals.s13_2.finaldelay.tbl', 
    '{msin}.hifv_finalcals.s13_4.finalBPcal.tbl', 
    '{msin}.hifv_finalcals.s13_5.averagephasegain.tbl', 
    '{msin}.hifv_finalcals.s13_7.finalampgaincal.tbl', 
    '{msin}.hifv_finalcals.s13_8.finalphasegaincal.tbl'], 
    gainfield=['', '', '', '', '', '', '', ''], 
    spwmap=[[], [], [], [], [], [], [], []], interp=['', '', '', '', 'linear,linearflag', '', '', ''], 
    parang=False, applymode='calflagstrict', flagbackup=False)

    if {cal_leakage_newgains}:
        #Now set the model for 3C147 as the pipeline did not do this
        setjy(vis='{msin}', field='{cal_leakage}', standard='Perley-Butler 2017', 
            model='{cal_leakage_im}', usescratch=True, scalebychan=True)

        #Calibrate on 3C147 as it's resolved, carrying previous caltables
        gaincal(vis='{msin}', caltable='{cal_leakage_newgains}', field='{cal_leakage}', 
            refant='{refant}', 
            spw='', gaintype='G',calmode='p', solint='int',
            gaintable=['{msin}.hifv_priorcals.s5_2.gc.tbl', 
        '{msin}.hifv_priorcals.s5_3.opac.tbl', 
        '{msin}.hifv_priorcals.s5_4.rq.tbl', 
        '{msin}.hifv_finalcals.s13_2.finaldelay.tbl', 
        '{msin}.hifv_finalcals.s13_4.finalBPcal.tbl', 
        '{msin}.hifv_finalcals.s13_5.averagephasegain.tbl', 
        '{msin}.hifv_finalcals.s13_7.finalampgaincal.tbl', 
        '{msin}.hifv_finalcals.s13_8.finalphasegaincal.tbl'])

        #now apply with the 3C147 new gain solutions
        applycal(vis='{msin}', field='{sources}',intent='CALIBRATE_POL_ANGLE#UNSPECIFIED,SYSTEM_CONFIGURATION#UNSPECIFIED,OBSERVE_TARGET#UNSPECIFIED,CALIBRATE_POL_LEAKAGE#UNSPECIFIED,CALIBRATE_BANDPASS#UNSPECIFIED,CALIBRATE_AMPLI#UNSPECIFIED,CALIBRATE_PHASE#UNSPECIFIED', 
        spw='{all_spws}', antenna='0~26', gaintable=['{msin}.hifv_priorcals.s5_2.gc.tbl', 
        '{msin}.hifv_priorcals.s5_3.opac.tbl', 
        '{msin}.hifv_priorcals.s5_4.rq.tbl', 
        '{msin}.hifv_finalcals.s13_2.finaldelay.tbl', 
        '{msin}.hifv_finalcals.s13_4.finalBPcal.tbl', 
        '{msin}.hifv_finalcals.s13_5.averagephasegain.tbl', 
        '{msin}.hifv_finalcals.s13_7.finalampgaincal.tbl', 
        '{msin}.hifv_finalcals.s13_8.finalphasegaincal.tbl',
        '{cal_leakage_newgains}'], 
        gainfield=['', '', '', '', '', '', '', '','{cal_leakage}'], 
        spwmap=[[], [], [], [], [], [], [], []], 
        interp=['', '', '', '', 'linear,linearflag', '', '', '',''], 
        parang=False, applymode='calflagstrict', flagbackup=False)
    else:
        pass

    #Do some basic flagging to prepare for cross-hand calibration
    flagdata(vis='{msin}', mode='rflag', correlation='ABS_RR,LL', 
    intent='*CALIBRATE*', datacolumn='corrected', ntime='scan', 
    combinescans=False,extendflags=False, winsize=3, 
    timedevscale=4.0, freqdevscale=4.0, action='apply', flagbackup=True, savepars=True)

    flagdata(vis='{msin}', mode='rflag', correlation='ABS_RR,LL', 
    intent='*TARGET*', datacolumn='corrected', ntime='scan', combinescans=False,extendflags=False, winsize=3, 
    timedevscale=4.0, freqdevscale=4.0, action='apply', flagbackup=True, savepars=True)

    statwt(vis='{msin}',minsamp=8,datacolumn='corrected')

    #Split parallel-hand calibration to new MS
    split(vis='{msin}',outputvis='{msout}',datacolumn='corrected',spw='{band_spws}')

    #ea11 has some bad data for 3C123
    #flagdata(vis='{msout}', antenna='ea11&ea22', spw='0~2')
    #flagdata(vis='{msout}', mode='rflag', antenna='ea11') #amp outliers only show up when averaging data in frequency, so likely some few channels with strong RFI

import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Need to set polangle calibrator model
def S(f,S,alpha,beta):
    return S*(f/{band_ref})**(alpha+beta*np.log10(f/{band_ref})) #find spectral index at band_ref GHz
def PF(f,a,b,c,d):
    return a+b*((f-{band_ref})/{band_ref})+c*((f-{band_ref})/{band_ref})**2+d*((f-{band_ref})/{band_ref})**3
def PA(f,a,b,c,d,e,g):
    return a+b*((f-{band_ref})/{band_ref})+c*((f-{band_ref})/{band_ref})**2+d*((f-{band_ref})/{band_ref})**3+e*((f-{band_ref})/{band_ref})**4+g*((f-{band_ref})/{band_ref})**5

#3C138 is flaring, so need values then apply scaling factor
if '{cal_polangle}':
    if '{cal_polangle}' == '3C138' or '{cal_polangle}' == 'J0521+1638' or '{cal_polangle}' == '0521+166=3C138':
        data=np.loadtxt('3c138_2019.txt')
        if '{band}' == 'C':
            #scaling=[1.030489,1.030489,1.11295196043943,1.11295196043943,1.11295196043943] 
            scaling=[1.11295196043943] #value at 01.12.2024
        elif '{band}' == 'Ku':
            #scaling for Ku-band at 01/02/2021
            scaling=[1.058064]
        elif '{band}' == 'L':
            scaling=[1.009391] #value at 01.12.2024
        else:
            scaling=[1.0]
    if '{cal_polangle}' == '3C48' or '{cal_polangle}' == 'J0137+3309':
        data=np.loadtxt('3C48_2019.txt')
        scaling=[1.0]

    extrap_scaling = scaling.copy()
    #Give two indices from "scaling" corresponding to frequencies from "data" with known fluxes, rest will be averaged
    known_fluxes=[0,3]
    if len(scaling)>1:
        for i in range(1,len(scaling)-1):#skip the first and last elements
            extrap_scaling[i] = (extrap_scaling[i-1]+extrap_scaling[i]+extrap_scaling[i+1])/3
            print(extrap_scaling[i])
        #Ensure these go back to normal as they should be fixed
        extrap_scaling[0]=scaling[0]
        extrap_scaling[3]=scaling[3]
        scaling = extrap_scaling
        print('Extrapolated scaling factors: ',extrap_scaling)
    else:
        #Don't scale
        scaling=1.0

    if '{band}' == 'Ku':
        row_min=7
        row_max=13
    if '{band}' == 'C':
        row_min=2
        row_max=9
    if '{band}' == 'L':
        row_min=0
        row_max=5

    popt_I,pcov=curve_fit(S,data[row_min:row_max,0],data[row_min:row_max,1]*scaling) 
    print("I@6GHz: ",popt_I[0], ' Jy')
    print("alpha: ",popt_I[1])
    print("beta", popt_I[2])
    print('Covariance: ',pcov)

    #Clear any plots that may exist
    plt.close()
    plt.plot(data[row_min:row_max,0],data[row_min:row_max,1]*scaling,'ro',label='data')
    plt.plot(np.arange(1,20,0.1),S(np.arange(1,20,0.1), *popt_I), 'r-', label='fit')

    plt.title('3C138')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux Density (Jy)')
    plt.savefig('FluxvFreq.png')


    popt_pf,pcov=curve_fit(PF,data[row_min:row_max,0],data[row_min:row_max,2])
    print("Polfrac Polynomial: ",popt_pf)
    print("Covariance: ", pcov)
    plt.close()
    plt.plot(data[row_min:row_max,0],data[row_min:row_max,2],'ro',label='data')
    plt.plot(np.arange(1,20,0.1),PF(np.arange(1,20,0.1), *popt_pf), 'r-', label='fit')

    plt.title('3C138')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Lin. Pol. Fraction')
    plt.savefig('LinPolFracvFreq.png')

    popt_pa,pcov=curve_fit(PA,data[row_min:row_max,0],data[row_min:row_max,3])
    print("Polangle Polynomial: ",popt_pa)
    print("Covariance: ", pcov)
    plt.close()
    plt.plot(data[row_min:row_max,0],data[row_min:row_max,3],'ro',label='data')
    plt.plot(np.arange(1,20,0.1),PA(np.arange(1,20,0.1), *popt_pa), 'r-', label='fit')

    plt.title('3C138')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Lin. Pol. Angle (rad)')
    plt.savefig('LinPolAnglevFreq.png')
    plt.close()

    I=popt_I[0]
    alpha=[popt_I[1],popt_I[2]]
    polfrac=popt_pf
    polangle=popt_pa
    print(polfrac,polangle)

reffreq = '{cal_polangle_ref_freq}'

#set model for polangle cal
setjy(vis='{msout}',field='{cal_polangle}',scalebychan=True,standard="manual",model="",
    listmodels=False,fluxdensity=[I,0,0,0],spix=alpha,reffreq='{cal_polangle_ref_freq}',polindex=polfrac,
    polangle=polangle,rotmeas=0,fluxdict={fluxdict},useephemdir=False,interpolation='nearest',
    usescratch=True, ismms=False)

#Set model for leakage cal
#Get from calibration weblog pipeline-*/stage12/casapy.log
setjy(vis='{msout}',field='{cal_leakage}',scalebychan=True,standard="manual",model="",
    listmodels=False,fluxdensity=[{cal_leakage_I_model},0,0,0],spix={cal_leakage_I_model_spix},reffreq='{cal_leakage_ref_freq}',polindex=[],
    polangle=[],rotmeas=0,fluxdict={fluxdict},useephemdir=False,interpolation='nearest',
    usescratch=True, ismms=False)

#Solve for the RL phase difference on the reference antenna
#Need to solve for each baseband at a time
kcross_sbd = '{msout}.Kcross_sbd'
gaincal(vis='{msout}', caltable=kcross_sbd,field='{cal_polangle}',spw='{solve_bbc1}', 
    refant='{refant}', 
    refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan',calmode='ap',append=False, gaintable='',
    gainfield='',interp='', spwmap=[[]], parang=True)
gaincal(vis='{msout}', caltable=kcross_sbd,field='{cal_polangle}',spw='{solve_bbc2}', 
    refant='{refant}', 
    refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan',calmode='ap',append=True, gaintable='',
    gainfield='',interp='', spwmap=[[]], parang=True)
if '{solve_bbc3}':
    gaincal(vis='{msout}', caltable=kcross_sbd,field='{cal_polangle}',spw='{solve_bbc3}', 
        refant='{refant}', 
        refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan',calmode='ap',append=True, gaintable='',
        gainfield='',interp='', spwmap=[[]], parang=True)
#Also try a multi-band solution (over all basebands):
kcross_mbd = '{msout}.Kcross_mbd'

gaincal(vis='{msout}', caltable=kcross_mbd,field='{cal_polangle}',spw='{solve_bbc1}', 
    refant='{refant}', 
    refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan,spw',calmode='ap',append=False, gaintable='',
    gainfield='',interp='', spwmap=[[]], parang=True)
gaincal(vis='{msout}', caltable=kcross_mbd,field='{cal_polangle}',spw='{solve_bbc2}', 
    refant='{refant}', 
    refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan,spw',calmode='ap',append=True, gaintable='',
    gainfield='',interp='', spwmap=[[]], parang=True)
if '{solve_bbc3}':
    gaincal(vis='{msout}', caltable=kcross_mbd,field='{cal_polangle}',spw='{solve_bbc3}', 
        refant='{refant}', 
        refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan,spw',calmode='ap',append=True, gaintable='',
        gainfield='',interp='', spwmap=[[]], parang=True)
#Do this mbd as sbd solution looks too erratic
#Now solve for leakages
dtab = '{msout}.Df'
polcal(vis='{msout}',
       caltable=dtab,
       field='{cal_leakage}',
       spw='{final_spws}',
       refant='{refant}',
       poltype='Df',
       solint='inf,2MHz',
       combine='scan',
       gaintable=[kcross_mbd],
       gainfield=[''],
       spwmap=[{baseband_list}],
       append=False)

xtab = '{msout}.Xf'
polcal(vis='{msout}', caltable=xtab, spw='{final_spws}',
field='{cal_polangle}', solint='inf,2MHz', combine='scan', poltype='Xf',
refant='{refant}',
gaintable=[kcross_mbd,dtab],
gainfield=['',''],
spwmap=[{baseband_list},[]],
append=False)

applycal(vis='{msout}', field='',gainfield=['','',''],
    flagbackup=True, interp=['','',''],gaintable=[kcross_mbd,dtab,xtab],
    spw='{final_spws}', calwt=[False,False,False],applymode='calflag',antenna='*&*',
    spwmap=[{baseband_list},[],[]],
    parang=True)

split(vis='{msout}',outputvis='{msout_target}',
    datacolumn='corrected',field='{target}')

statwt(vis='{msout_target}',datacolumn='data',minsamp=8)

"""