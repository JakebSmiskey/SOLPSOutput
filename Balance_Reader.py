#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:12:38 2024

@author: jakeb
"""

import netCDF4 as nc
import numpy as np

#dset.variables.keys() 

class BalanceNC:
    def __init__(self,fname):
        self.load_netcdf_to_dict(fname)

    def load_netcdf_to_dict(self,file_path):
        # Open the netCDF file
        dataset = nc.Dataset(file_path, mode='r')

        #Dictionary to store data

        #Loop through each variable in the file and store it in the dictionary
        for var_name in dataset.variables:
            # Extract the variable data as a NumPy array
            data = np.array(dataset.variables[var_name][:])
            if data.ndim == 1:
                setattr(self,var_name,data)
            elif data.ndim == 2:
                data = np.transpose(data,axes=(1,0))
                setattr(self,var_name,data)
            elif data.ndim == 3:
                data = np.transpose(data,axes=(2,1,0))
                setattr(self,var_name,data)
            elif data.ndim == 4:
                data = np.transpose(data,axes=(3,2,1,0))
                setattr(self,var_name,data)
            else:
                setattr(self,var_name,data)

if __name__ == "__main__":
    test_balance = BalanceNC("balance.nc")
