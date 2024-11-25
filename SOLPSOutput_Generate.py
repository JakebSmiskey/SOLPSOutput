#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:50:23 2024

@author: Jakeb Smiskey

--------------------------------------------------------------------------
This script reads the output data from SOLPS-ITER 
and creates a centralized file with the .npy structure (dictionary).
This file is called SOLPS_var.npy unless otherwise changed.

The files are read with classes from the readers.py file. 

The SOLPS-ITER files read:
 -b2fstate 
 -b2fplasmf
 -b2fgmtry
 -input.dat
 -fort.44
 -fort.46
 -*.sno file (only nxiso)
 
The ionization_potentials file is static and contains the ionization energy
for all charge states up to Rd.

The data contained in the file is stored as a dictionary with a set 
variable name specified by SOLPS or otherwise specififed in the readers.py file
Therefore the SOLPS-ITER Manual should be used when determining what variable name
should be used.

The flags are:
    -g = gfile name
    -s = SOLPS-ITER run directory

"""

import Balance_Reader,B2fgmtry_Reader,B2fplasmf_Reader,B2fstate_Reader,GEQDSK_Reader,Fort44_Reader,Fort46_Reader,IonizationPotential_Reader
import numpy as np
import SOLPSOutput
import os 
import argparse
        
parser = argparse.ArgumentParser()
parser.add_argument('-g', nargs='?', const='gfile', type=str, default='.', help='Name of the gfile')
parser.add_argument('-s', nargs='?', const='.', type=str, default='.', help='SOLPS-ITER run directory')
args = parser.parse_args()
directory = args.s
gfile = args.g
f44 = Fort44_Reader.Fort44(directory +"/fort.44", directory+"/input.dat")
f46 = Fort46_Reader.Fort46(directory +"/fort.46")
gfile = GEQDSK_Reader.GEQDSK(directory + gfile)
gmtry = B2fgmtry_Reader.B2fgmtry(directory +"/b2fgmtry")
b2fstate = B2fstate_Reader.B2fstate(directory +"/b2fstate")
b2fplasmf = B2fplasmf_Reader.B2fplasmf(directory +"/b2fplasmf", b2fstate.nx, b2fstate.ny, b2fstate.ns)
balance = Balance_Reader.BalanceNC(directory+"/balance.nc")
ion_pots = IonizationPotential_Reader.IonizationPotential("/fusion/projects/codes/imas/c8/SOLPS-ITER/solps-iter_3.0.8_develop_test/modules/B2.5/Database/ionization_potentials")
variable = SOLPSOutput.SOLPSOutput(f44,f46,gfile,gmtry,b2fstate,b2fplasmf,ion_pots,balance)
np.save(directory+"SOLPS_vars.npy",variable, allow_pickle=True)
