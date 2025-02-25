# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:15:46 2024

@author: Jakeb Smiskey and Ray Mattis

This class populates a dictionary with the outputs of the reader classes 
and derives some quantities that might be useful

Add new quanitites you want derived in this file as it will speed up future
post processing!

Best,
Jakeb
"""
import numpy as np
import B2fstate_Reader
import GEQDSK_Reader
import Fort44_Reader
import Fort46_Reader
import IonizationPotential_Reader
import Balance_Reader


class SOLPSOutput:
    chrgstate_D = 2  # Ground State + Charge States
    chrgstate_H = 2
    chrgstate_He = 5
    chrgstate_C = 7
    chrgstate_N = 8
    chrgstate_Ne = 11
    chrgstate_Ar = 19
    puffing_map = {
        "H": chrgstate_H,
        "D": chrgstate_D,
        "C": chrgstate_C,
        "HE": chrgstate_He,
        "NE": chrgstate_Ne,
        "N": chrgstate_N,
        "AR": chrgstate_Ar
    }

    def __init__(self, f44, f46, gfile, b2fstate, ion_pots, balance):
        self.__dict__.update(**f44.__dict__)
        self.__dict__.update(**f46.__dict__)
        self.__dict__.update(**gfile.__dict__)
        self.__dict__.update(**b2fstate.__dict__)
        self.__dict__.update(**ion_pots.__dict__)
        self.__dict__.update(**balance.__dict__)
        self.calc_derived()

    def calc_derived(self,):
        # Derived stuff
        crx=self.crx
        cry=self.cry
        nx = self.nx
        ny = self.ny
        hx = self.hx
        hy = self.hy
        hz = self.hz
        vol = self.vol
        B = self.bb[:, :, -1]
        Bx = self.bb[:, :, 0]
        crx = np.reshape(crx,((nx*ny),1,4),order='F')
        cry = np.reshape(cry,((nx*ny),1,4),order='F')
        ll = np.concatenate((crx[:,:,0],cry[:,:,0]), axis = 1)
        lr = np.concatenate((crx[:,:,1],cry[:,:,1]), axis = 1)
        ul = np.concatenate((crx[:,:,2],cry[:,:,2]), axis = 1)
        ur = np.concatenate((crx[:,:,3],cry[:,:,3]), axis = 1)
        mesh_verts = [[tuple(lr[i]), tuple(ll[i]), tuple(ul[i]), tuple(ur[i])]
                 for i in range(nx*ny)]
        setattr(self, "lower_right_vertex", lr)
        setattr(self, "lower_left_vertex", ll)
        setattr(self, "upper_right_vertex", ul)
        setattr(self, "upper_left_vertex", ur)
        setattr(self, "mesh_verticies", mesh_verts)

        A_perp = (hy*hz)
        A_par = (hy*hz) * (Bx/B)
        A_rad = (hx*hz)

        # qperp
        qtot_e = self.fhe[:, :, 0]
        qtot_i = self.fhi[:, :, 0]
        qtot = qtot_e + qtot_i
        qtote_perp = qtot_e/A_perp
        qtote_par = qtot_e/A_par
        qtoti_perp = qtot_i/A_perp
        qtoti_par = qtot_i/A_par
        qtot_perp = qtot/A_perp
        qtot_par = qtot/A_par
        setattr(self, "derived_qtote_perp", qtote_perp * 1e-6)  # [W/cm^2]
        setattr(self, "derived_qtote_par", qtote_par * 1e-6)  # [W/cm^2]
        setattr(self, "derived_qtoti_perp", qtoti_perp * 1e-6)  # [W/cm^2]
        setattr(self, "derived_qtoti_par", qtoti_par * 1e-6)  # [W/cm^2]
        setattr(self, "derived_qtot_perp", qtot_par * 1e-6)  # [W/cm^2]
        setattr(self, "derived_qtot_par", qtot_par * 1e-6)  # [W/cm^2]

        # Ion current
        main_ion_flux = qtot_e = self.fna[:, :, 0, 1]
        # main_ion_flux = qtot_e = self.dict["state"]["fna"][:,:,0,1]
        # imp_ionsource = self.dict['output']['b2npc9_fnax003']*1 + self.dict['output']['b2npc9_fnax004']*2 + self.dict['output']['b2npc9_fnax005']*3 + self.dict['output']['b2npc9_fnax006']*4 + self.dict['output']['b2npc9_fnax007']*5+self.dict['output']['b2npc9_fnax008']*6

        j_perp = main_ion_flux * 1.602e-19 / A_perp
        j_par = (main_ion_flux) * 1.602e-19 / A_par
        setattr(self, "derived_jperp", j_perp * 1e-4)  # [A/cm^2]
        setattr(self, "derived_jpar", j_par * 1e-4)  # [A/cm^2]

        # Main ion flux
        setattr(self, "derived_main_ion_flux", main_ion_flux)

        # Make total radiation
        b2br = self.rqbrm
        b2ra = self.rqrad
        brhe = self.b2stbr_she
        brhi = self.b2stbr_shi
        brna = self.b2stbr_sna*1.602E-19
        pot = self.pot
        vol = self.vol
        b2ra_tot = np.sum(b2ra, axis=2)
        b2br_tot = np.sum(b2br, axis=2)
        brna_tot = brna.clip(0)
        sindx = 0
        findx = 0
        for groups in self.particles:
            findx += self.puffing_map[groups]
            num = len(self.pot[self.puffing_map[groups]-2])
            frst = 0
            for i in range(num):
                if i == frst:
                    brna[:, :, sindx+1+i] *= self.pot[self.puffing_map[groups]-2][i]
                else:
                    brna[:, :, sindx+1 +
                         i] *= np.sum(self.pot[self.puffing_map[groups]-2][frst:i+1])
            sindx += self.puffing_map[groups]

        brna_tot = np.sum(brna, axis=2)
        total_rad = (b2br_tot+b2ra_tot-brhe-brna_tot-brhi)/vol * 1E-6
        total_rad = np.reshape(total_rad, (self.nx, self.ny), order="F")
        setattr(self, "derived_total_rad", total_rad)

        # Main Ion Radiation
        if "D" in self.particles:
            b2br_D = np.sum(b2br[:, :, :1], axis=2)
            b2ra_D = np.sum(b2ra[:, :, :1], axis=2)
            brna_D = np.sum(brna[:, :, :1], axis=2)
            D_rad = (b2br_D+b2ra_D-brna_D)/vol * 1E-6
            D_rad = np.reshape(D_rad, (self.nx, self.ny), order="F")
            setattr(self, "derived_D_rad", D_rad)
        # Make Carbon radiation
        if "C" in self.particles:
            b2br_carbon = np.sum(b2br[:, :, 3:9], axis=2)
            b2ra_carbon = np.sum(b2ra[:, :, 3:9], axis=2)
            brna_carbon = np.sum(brna[:, :, 3:9], axis=2)
            carbon_rad = (b2br_carbon+b2ra_carbon-brna_carbon)/vol * 1E-6
            carbon_rad = np.reshape(carbon_rad, (self.nx, self.ny), order="F")
            setattr(self, "derived_carbon_rad", carbon_rad)
    # Make Nitrogen radiation
        if "N" in self.particles:
            b2br_nitrogen = np.sum(b2br[:, :, 10:17], axis=2)
            b2ra_nitrogen = np.sum(b2ra[:, :, 10:17], axis=2)
            brna_nitrogen = np.sum(brna[:, :, 10:17], axis=2)
            nitrogen_rad = (b2br_nitrogen+b2ra_nitrogen -
                            brna_nitrogen)/vol * 1E-6
            nitrogen_rad = np.reshape(
                nitrogen_rad, (self.nx, self.ny), order="F")
            setattr(self, "derived_nitrogen_rad", nitrogen_rad)

        # Make ion sources
        brna = self.b2stbr_sna
        vol = self.vol
        # area = (self.solps_data["hz"]*self.solps_data["hy"] *
        #  self.solps_data["qc"])
        if "D" in self.particles:
            source = brna[:, :, 1] / \
                (vol*1e2)*1.602e-19
            setattr(self, "derived_D_source", source)
        if "C" in self.particles:
            source = np.sum(brna[:, :, 3:9],axis=2) / \
                (vol*1e2)*1.602e-19
            setattr(self, "derived_C_source", source)
        if "N" in self.particles:
            source = np.sum(brna[:, :, 10:17],axis = 2) / \
                (vol*1e2)*1.602e-19
            setattr(self, "derived_N_source", source)
        
        #Ion Densities
        if "D" in self.particles:
            D_density = self.na[:,:,0:2]
            setattr(self,"derived_D_na",D_density)
        if "C" in self.particles:
            C_density = self.na[:,:,2:9]
            setattr(self,"derived_C_na",C_density)
        if "N" in self.particles:
            N_density = self.na[:,:,9:17]
            setattr(self,"derived_N_na",N_density)
        
        
