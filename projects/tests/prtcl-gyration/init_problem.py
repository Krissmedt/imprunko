from __future__ import print_function
from configSetup import Configuration

from numpy import sqrt, pi
import numpy as np


class Configuration_Gyro(Configuration):

    def __init__(self, *file_names, do_print=False):
        Configuration.__init__(self, *file_names)

        #-------------------------------------------------- 
        # problem specific initializations
        if do_print:
            print("Initializing gyration setup...")
    
        #-------------------------------------------------- 
        # particle initialization

        #local variables just for easier/cleaner syntax
        c   = self.cfl

        self.larmor = self.gamma*c**2
        self.beta = sqrt(1-1/self.gamma**2.) 
        self.vy = sqrt(self.gamma**2.-1)
        self.vy = self.beta*self.gamma
        
	#plasma reaction & subsequent normalization
        self.omp=c/self.c_omp
        self.qe = -1

        #-------------------------------------------------- 
        # field initialization
	
        self.Nx = 1
        self.Ny = 1
        self.Nz = 1
        self.NxMesh = int(self.larmor*2.2)
        self.NyMesh = int(self.larmor*2.2)
        self.NzMesh =1

        self.dx=1.0 
        self.dy=1.0 
        self.dz=1.0 

        self.x_start = np.floor(self.larmor*0.1)

        print(self.larmor)
        print(self.NxMesh)
        print(self.x_start)

        #---------cold plasma-----------
        self.bphi=90.0  #Bfield z angle (bphi=0  bz, bphi=90 -> x-y plane)
        self.btheta=0.0   #Bfield x-y angle: btheta=0 -> parallel

        # set external magnetic field strength

        self.binit = 1.0


