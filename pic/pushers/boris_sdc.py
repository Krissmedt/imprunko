#!/usr/bin/env python3

## Dependencies
import numpy as np
import scipy.sparse as sps
import scipy.interpolate as scint
from math import sqrt, fsum, pi
from gauss_legendre import CollGaussLegendre
from gauss_lobatto import CollGaussLobatto
import time
import copy as cp

## Class
class kpps_analysis:
    def __init__(self,**kwargs):

        
        self.particleIntegration = False
        self.particleIntegrator = 'boris_SDC'
        self.nodeType = 'lobatto'
        self.M = 2
        self.K = 1
        self.rhs_dt = 1
        self.gather = self.none
        self.bound_cross_methods = []
        self.looped_axes = []
        self.calc_residuals = self.calc_residuals_max
        self.display_residuals = self.display_residuals_max
        
        ## Iterate through keyword arguments and store all in object (self)
        self.params = cp.deepcopy(kwargs)
        for key, value in self.params.items():
            setattr(self,key,value)
            
            
        # check for other intuitive parameter names
        name_dict = {}
        name_dict['looped_axes'] = ['periodic_axes','mirrored_axes']
        
        for key, value in name_dict.items():
            for name in value:
                try:
                    setattr(self,key,getattr(self,name))
                except AttributeError:
                    pass   
    
######################## Particle Analysis Methods ############################
    def boris(self, vel, E, B, dt, alpha, ck=0):
        """
        Applies Boris' trick for given velocity, electric and magnetic 
        field for vector data in the shape (N x 3), i.e. particles as rows 
        and x,y,z components for the vector as the columns.
        k = delta_t * alpha / 2
        """ 
        
        k = dt*alpha/2
        
        tau = k*B

        vMinus = vel + dt/2 * (alpha*E + ck)
        
        tauMag = np.linalg.norm(tau,axis=1)
        vDash = vMinus + np.cross(vMinus,tau)
        
        tm = 2/(1+tauMag**2)

        for col in range(0,3):
            vDash[:,col] = tm[:] * vDash[:,col]

        vPlus = vMinus + np.cross(vDash,tau)
        
        vel_new = vPlus + dt/2 * (alpha*E + ck)
        
        return vel_new
    
    
    def boris_staggered(self,species_list,mesh,controller,**kwargs):
        dt = controller.dt
        self.run_fieldIntegrator(species_list,mesh,controller)
        for species in species_list:
            alpha = species.a
    
            self.fieldGather(species,mesh)
            species.vel = self.boris(species.vel,species.E,species.B,dt,alpha)
            species.pos = species.pos + controller.dt * species.vel
            self.check_boundCross(species,mesh,**kwargs)

        return species_list

    
    def boris_synced(self,species_list,mesh,controller,**kwargs):
        dt = controller.dt
        for species in species_list:
            alpha = species.a
    
            species.pos = species.pos + dt * (species.vel + dt/2 * self.lorentz_std(species,mesh))
            self.check_boundCross(species,mesh,**kwargs)
            
        self.run_fieldIntegrator(species_list,mesh,controller)
            
        for species in species_list:
            E_old = species.E
            self.fieldGather(species,mesh)
            E_new = species.E
    
            species.E_half = (E_old+E_new)/2
            
            species.vel = self.boris(species.vel,species.E_half,species.B,dt,alpha)

        return species_list
        
    
    def collSetup(self,species_list,fields,controller=None,**kwargs):
        M = self.M
        K = self.K
        dt = controller.dt
        
        if self.nodeType == 'lobatto':
            self.ssi = 1    #Set sweep-start-index 'ssi'
            self.collocationClass = CollGaussLobatto
            self.updateStep = self.lobatto_update
            self.rhs_dt = (self.M - 1)*self.K
            
        elif self.nodeType == 'legendre':
            self.ssi = 0 
            self.collocationClass = CollGaussLegendre
            self.updateStep = self.legendre_update
            self.rhs_dt = (self.M + 1)*self.K
        
        coll = self.collocationClass(self.M,0,1) #Initialise collocation/quadrature analysis object (class is Daniels old code)
        self.nodes = coll._getNodes
        self.weights = coll._getWeights(coll.tleft,coll.tright) #Get M  nodes and weights 


        self.Qmat = coll._gen_Qmatrix           #Generate q_(m,j), i.e. the large weights matrix
        self.Smat = coll._gen_Smatrix           #Generate s_(m,j), i.e. the large node-to-node weights matrix

        self.delta_m = coll._gen_deltas         #Generate vector of node spacings
        
        for species in species_list:
            self.fieldGather(species,fields)
            species.F = species.a*(species.E + np.cross(species.vel,species.B))

        self.coll_params = {}
        
        d = 3*species.nq
        
        self.coll_params['dt'] = controller.dt
        
        #Remap collocation weights from [0,1] to [tn,tn+1]
        #nodes = (t-dt) + self.nodes * dt
        self.coll_params['weights'] = self.weights * dt 
        
        Qmat = self.Qmat * dt
        Smat = self.Smat * dt
        delta_m = self.delta_m * dt

        self.coll_params['Qmat'] = Qmat
        self.coll_params['Smat'] = Smat
        self.coll_params['dm'] = delta_m

        #Define required calculation matrices
        QE = np.zeros((M+1,M+1),dtype=np.float)
        QI = np.zeros((M+1,M+1),dtype=np.float)
        QT = np.zeros((M+1,M+1),dtype=np.float)
        
        SX = np.zeros((M+1,M+1),dtype=np.float)
        
        for i in range(0,M):
            QE[(i+1):,i] = delta_m[i]
            QI[(i+1):,i+1] = delta_m[i] 
        
        QT = 1/2 * (QE + QI)
        QX = QE @ QT + (QE*QE)/2
        SX[:,:] = QX[:,:]
        SX[1:,:] = QX[1:,:] - QX[0:-1,:]      
        
        self.coll_params['SX'] = SX
        self.coll_params['SQ'] = Smat @ Qmat
        
        self.coll_params['x0'] = np.zeros((d,M+1),dtype=np.float)
        self.coll_params['v0'] = np.zeros((d,M+1),dtype=np.float)
        
        self.coll_params['xn'] = np.zeros((d,M+1),dtype=np.float)
        self.coll_params['vn'] = np.zeros((d,M+1),dtype=np.float)
        
        self.coll_params['F'] = np.zeros((d,M+1),dtype=np.float)
        self.coll_params['Fn'] = np.zeros((d,M+1),dtype=np.float)

        for species in species_list:
            species.x0 = np.copy(self.coll_params['x0'])
            species.v0 = np.copy(self.coll_params['v0'])
            
            species.xn = np.copy(self.coll_params['xn'])
            species.vn = np.copy(self.coll_params['vn'])
            
            species.F = np.copy(self.coll_params['Fn'])
            species.Fn = np.copy(self.coll_params['F'])
            
            species.x_con = np.zeros((K,M))
            species.x_res = np.zeros((K,M))
            species.v_con = np.zeros((K,M))
            species.v_res = np.zeros((K,M))
                

    def boris_SDC(self, species_list,fields, controller,**kwargs):
        M = self.M
        K = self.K
        
        #Remap collocation weights from [0,1] to [tn,tn+1]
        #nodes = (t-dt) + self.nodes * dt
        weights =  self.coll_params['weights']

        Qmat =  self.coll_params['Qmat']
        Smat =  self.coll_params['Smat']

        dm =  self.coll_params['dm']

        SX =  self.coll_params['SX'] 

        SQ =  self.coll_params['SQ']

        for species in species_list:
            ## Populate node solutions with x0, v0, F0 ##
            species.x0[:,0] = self.toVector(species.pos)
            species.v0[:,0] = self.toVector(species.vel)
            species.F[:,0] = self.toVector(species.lntz)
            species.En_m0 = species.E

            for m in range(1,M+1):
                species.x0[:,m] = species.x0[:,0]
                species.v0[:,m] = species.v0[:,0]
                species.F[:,m] = species.F[:,0]
            #############################################
            
            species.x = np.copy(species.x0)
            species.v = np.copy(species.v0)
            
            species.xn[:,:] = species.x[:,:]
            species.vn[:,:] = species.v[:,:]
            species.Fn[:,:] = species.F[:,:]

        #print()
        #print(simulationManager.ts)
        for k in range(1,K+1):
            #print("k = " + str(k))
            for species in species_list:
                species.En_m = species.En_m0 #reset electric field values for new sweep

            for m in range(self.ssi,M):
                for species in species_list:
                    #print("m = " + str(m))
                    #Determine next node (m+1) positions
                    sumSQ = 0
                    for l in range(1,M+1):
                        sumSQ += SQ[m+1,l]*species.F[:,l]
                    
                    sumSX = 0
                    for l in range(1,m+1):
                        sumSX += SX[m+1,l]*(species.Fn[:,l] - species.F[:,l])
    
                    species.xQuad = species.xn[:,m] + dm[m]*species.v[:,0] + sumSQ
                              
                    ### POSITION UPDATE FOR NODE m/SWEEP k ###
                    species.xn[:,m+1] = species.xQuad + sumSX 
                    
                    ##########################################
                    
                    sumS = 0
                    for l in range(1,M+1):
                        sumS += Smat[m+1,l] * species.F[:,l]
                    
                    species.vQuad = species.vn[:,m] + sumS
                    
                    species.ck_dm = -1/2 * (species.F[:,m+1]
                                            +species.F[:,m]) + 1/dm[m] * sumS
                    
                    ### FIELD GATHER FOR m/k NODE m/SWEEP k ###
                    species.pos = self.toMatrix(species.xn[:,m+1],3)
                    self.check_boundCross(species,fields,**kwargs)
                    
                
                self.run_fieldIntegrator(species_list,fields,controller)
                
                for species in species_list:
                    self.fieldGather(species,fields)
                    ###########################################
                    
                    #Sample the electric field at the half-step positions (yields form Nx3)
                    half_E = (species.En_m+species.E)/2
                    species.En_m = species.E              #Save m+1 value as next node's m value
                    
                    #Resort all other 3d vectors to shape Nx3 for use in Boris function
                    
                    v_oldNode = self.toMatrix(species.vn[:,m])
                    species.ck_dm = self.toMatrix(species.ck_dm)
                    
                    ### VELOCITY UPDATE FOR NODE m/SWEEP k ###
                    v_new = self.boris(v_oldNode,half_E,species.B,dm[m],species.a,species.ck_dm)
                    species.vn[:,m+1] = self.toVector(v_new)
                    ##########################################
                    
                    self.calc_residuals(species,m,k)
                    
                    ### LORENTZ UPDATE FOR NODE m/SWEEP k ###
                    species.vel = species.toMatrix(species.vn[:,m+1])
                    
                    species.lntz = species.a*(species.E + np.cross(species.vel,species.B))
                    species.Fn[:,m+1] = species.toVector(species.lntz)
                    
                    #########################################
                
            for species in species_list:
                species.F[:,:] = species.Fn[:,:]
                species.x[:,:] = species.xn[:,:]
                species.v[:,:] = species.vn[:,:]
            
                
        species_list = self.updateStep(species_list,fields,weights,Qmat)
            
        return species_list
    
   
    
    def lobatto_update(self,species_list,mesh,*args,**kwargs):
        for species in species_list:
            pos = species.x[:,-1]
            vel = species.v[:,-1]
            
            species.pos = species.toMatrix(pos)
            species.vel = species.toMatrix(vel)
            self.check_boundCross(species,mesh,**kwargs)

        return species_list
    
    
    def legendre_update(self,species_list,mesh,weights,Qmat,**kwargs):
        for species in species_list:
            M = self.M
            d = 3*species.nq
            
            Id = np.identity(d)
            q = np.zeros(M+1,dtype=np.float)
            q[1:] = weights
            q = np.kron(q,Id)
            qQ = q @ np.kron(Qmat,Id)
            
            V0 = self.toVector(species.v0.transpose())
            F = self.FXV(species,mesh)
            
            vel = species.v0[:,0] + q @ F
            pos = species.x0[:,0] + q @ V0 + qQ @ F
            
            species.pos = species.toMatrix(pos)
            species.vel = species.toMatrix(vel)
            self.check_boundCross(species,mesh,**kwargs)
        return species_list
    
    
    def lorentzf(self,species,mesh,m,**kwargs):
        species.pos = species.toMatrix(species.x[:,m])
        species.vel = species.toMatrix(species.v[:,m])
        self.check_boundCross(species,mesh,**kwargs)

        self.fieldGather(species,mesh)

        F = species.a*(species.E + np.cross(species.vel,species.B))
        F = species.toVector(F)
        return F
    
    def lorentz_std(self,species,fields):
        F = species.a*(species.E + np.cross(species.vel,species.B))

        return F
    
    
    
    def FXV(self,species,fields):
        dxM = np.shape(species.x)
        d = dxM[0]
        M = dxM[1]-1
        
        F = np.zeros((d,M+1),dtype=np.float)
        for m in range(0,M+1):
            F[:,m] = self.lorentzf(species,fields,m)
        
        F = self.toVector(F.transpose())
        return F
    
    
    
    def gatherE(self,species,mesh,x,**kwargs):
        species.pos = self.toMatrix(x,3)
        self.check_boundCross(species,mesh,**kwargs)
        
        self.fieldGather(species,mesh)
        
        return species.E
    
    def gatherB(self,species,mesh,x,**kwargs):
        species.pos = self.toMatrix(x,3)
        self.check_boundCross(species,mesh,**kwargs)
        
        self.fieldGather(species,mesh)
        
        return species.B
    
    
####################### Boundary Analysis Methods #############################
    def check_boundCross(self,species,mesh,**kwargs):
        for method in self.bound_cross_methods:
                method(species,mesh,**kwargs)
        return species
    
        
################################ Hook methods #################################
    def ES_vel_rewind(self,species_list,mesh,controller=None):
        dt = controller.dt
        for species in species_list:
            self.fieldGather(species,mesh)
            species.vel = species.vel - species.E * species.a * dt/2
 
        
    def calc_residuals_avg(self,species,m,k):
        s = species
        s.x_con[k-1,m] = np.average(np.abs(s.xn[:,m+1] - s.x[:,m+1]))
        s.x_res[k-1,m] = np.average(np.linalg.norm(s.xn[:,m+1]-s.xQuad))
        
        s.v_res[k-1,m] = np.average(np.linalg.norm(s.vn[:,m+1]-s.vQuad))
        s.v_con[k-1,m] = np.average(np.abs(s.vn[:,m+1] - s.v[:,m+1]))
        
    def calc_residuals_max(self,species,m,k):
        s = species
        s.x_con[k-1,m] = np.max(np.abs(s.xn[:,m+1] - s.x[:,m+1]))
        s.x_res[k-1,m] = np.max(np.linalg.norm(s.xn[:,m+1]-s.xQuad))
        
        s.v_res[k-1,m] = np.max(np.linalg.norm(s.vn[:,m+1]-s.vQuad))
        s.v_con[k-1,m] = np.max(np.abs(s.vn[:,m+1] - s.v[:,m+1]))
    
    
    def display_convergence(self,species_list,fields,**kwargs):
        for species in species_list:
            print("Position convergence, " + str(species.name) + ":")
            print(species.x_con)
            
            print("Velocity convergence, " + str(species.name) + ":")  
            print(species.v_con)
        
        
    def display_residuals_full(self,species_list,fields,**kwargs):
        for species in species_list:
            print("Position residual, " + str(species.name) + ":")
            print(species.x_res)
            
            print("Velocity residual, " + str(species.name) + ":")
            print(species.v_res)
            
    def display_residuals_max(self,species_list,fields,**kwargs):
        for species in species_list:
            print("Position residual, " + str(species.name) + ":")
            print(np.max(species.x_res,1))
            
            print("Velocity residual, " + str(species.name) + ":")
            print(np.max(species.v_res,1))
        
        
    def get_u(self,x,v):
        assert len(x) == len(v)
        d = len(x)
        
        Ix = np.array([1,0])
        Iv = np.array([0,1])
        Id = np.identity(d)
        
        u = np.kron(Id,Ix).transpose() @ x + np.kron(Id,Iv).transpose() @ v
        return u

############################ Misc. functionality ##############################
        
    def toVector(self,storageMatrix):
        rows = storageMatrix.shape[0]
        columns = storageMatrix.shape[1]
        vector = np.zeros(rows*columns)
        
        for i in range(0,columns):
            vector[i::columns] = storageMatrix[:,i]
        return vector
    
    
    def toMatrix(self,vector,columns=3):
        rows = int(len(vector)/columns)
        matrix = np.zeros((rows,columns))
        
        for i in range(0,columns):
            matrix[:,i] = vector[i::columns]
        return matrix
    
    def meshtoVector(self,mesh):
        shape = np.shape(mesh)
        x = np.zeros(shape[0]*shape[1]*shape[2],dtype=np.float)
        xi = 0
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    x[xi] = mesh[i,j,k]
                    xi += 1
        return x
    
    
    def vectortoMesh(self,x,shape):
        mesh = np.zeros(shape,dtype=np.float)
        xi = 0
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    mesh[i,j,k] = x[xi]
                    xi += 1
        return mesh
    
    def stringtoMethod(self,front):
        try:
            function = getattr(self,front)
            front = function
        except TypeError:
            pass
        
        return front
        
    def none(self,*args,**kwargs):
        pass
