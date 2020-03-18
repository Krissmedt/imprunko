from __future__ import print_function

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm


import pyplasma as plasma
import pyplasmaDev as pdev



import initialize as init
import injector
from visualize_amr import plot2DSlice


def filler(xloc, uloc, ispcs, conf):

    delgam = np.sqrt(1.0)

    mean = [3.0, 0.0, 0.0]
    cov  = np.zeros((3,3))
    cov[0,0] = 1.0 
    cov[1,1] = 2.0 
    cov[2,2] = 5.0

    f = multivariate_normal.pdf(uloc, mean, cov)

    return f


def insert_em(node, conf):

    for i in range(node.getNx()):
        for j in range(node.getNy()):
            c = node.getCellPtr(i,j)
            yee = c.getYee(0)

            for q in range(conf.NxMesh):
                for k in range(conf.NyMesh):
                    yee.ex[q,k,0] =  0.0
                    yee.ey[q,k,0] =  0.0
                    yee.ez[q,k,0] =  0.0

                    yee.bx[q,k,0] =  0.0
                    yee.by[q,k,0] =  0.0
                    yee.bz[q,k,0] =  0.1



class Conf:

    outdir = "out"


    #-------------------------------------------------- 
    # space parameters
    Nx = 1
    Ny = 1

    dt = 1.0
    dx = 1.0
    dy = 1.0
    dz = 1.0

    NxMesh = 1
    NyMesh = 1
    NzMesh = 1

    qm = -1.0

    #-------------------------------------------------- 
    # velocity space parameters
    vxmin = -10.0
    vymin = -10.0
    vzmin = -10.0

    vxmax =  10.0
    vymax =  10.0
    vzmax =  10.0

    Nxv = 20
    Nyv = 20
    Nzv = 20

    #vmesh refinement
    refinement_level = 2
    clip = True
    clipThreshold = 1.0e-5


def saveVisz(lap, conf):

    slap = str(lap).rjust(4, '0')
    fname = conf.outdir + '/amr2_{}.png'.format(slap)
    plt.savefig(fname)


def plotAll(axs, node, conf, lap):

    #for i in range(node.getNx()):
    #    for j in range(node.getNy()):
    #        cid = node.cellId(i,j)
    #        c = node.getCellPtr(cid) 
    #        blocks = c.getPlasma(0,0)
    #        vmesh = block[0,0,0]
    #        print("xx", vmesh.get_cells(True))

    
    #get first of first velomesh from cell
    cid   = node.cellId(0,0)
    c     = node.getCellPtr(cid) #get cell ptr
    block = c.getPlasmaSpecies(0,0)       # timestep 0
    vmesh = block[0,0,0]
    #print("xx", vmesh.get_cells(True))


    rfl = conf.refinement_level
    args = {"dir":"xy", 
            "q":  "mid",
            "rfl": rfl }
    plot2DSlice(axs[0], vmesh, args)

    args = {"dir":"xz", 
            "q":   "mid",
            "rfl": rfl }
    plot2DSlice(axs[1], vmesh, args)

    args = {"dir":"yz", 
            "q":   "mid",
            "rfl": rfl }
    plot2DSlice(axs[2], vmesh, args)

    #set_xylims(axs)
    saveVisz(lap, conf)


# dig out velocity meshes from containers and feed to actual adapter
def adapt(node):
    cid  = node.cellId(0,0)
    cell = node.getCellPtr(cid)
    block = cell.getPlasmaSpecies(0,0)
    vmesh = block[0,0,0]
    
    adaptVmesh(vmesh)



def adaptVmesh(vmesh):
    adapter = pdev.Adapter();

    sweep = 1
    while(True):
        adapter.check(vmesh)
        adapter.refine(vmesh)

        print("cells to refine: {}".format( len(adapter.cells_to_refine)))
        for cid in adapter.cells_created:
            rfl = vmesh.get_refinement_level(cid)
            indx = vmesh.get_indices(cid)
            uloc = vmesh.get_center(indx, rfl)

            val = pdev.tricubic_interp(vmesh, indx, [0.5,0.5,0.5], rfl)
            vmesh[indx[0], indx[1], indx[2], rfl] = val



        adapter.unrefine(vmesh)
        print("cells to be removed: {}".format( len(adapter.cells_removed)))

        sweep += 1
        if sweep > conf.refinement_level: break

    if conf.clip:
        print("clipping...")
        vmesh.clip_cells(conf.clipThreshold)



if __name__ == "__main__":

    ################################################## 
    # set up plotting and figure
    plt.fig = plt.figure(1, figsize=(12,20))
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick')
    plt.rc('ytick')
    
    gs = plt.GridSpec(3, 1)
    gs.update(hspace = 0.5)
    
    axs = []
    axs.append( plt.subplot(gs[0,0]) )
    axs.append( plt.subplot(gs[1,0]) )
    axs.append( plt.subplot(gs[2,0]) )



    ################################################## 
    # node configuration
    conf = Conf()

    node = plasma.Grid(conf.Nx, conf.Ny)
    xmin = 0.0
    xmax = conf.dx*conf.Nx*conf.NxMesh
    ymin = 0.0
    ymax = conf.dy*conf.Ny*conf.NyMesh

    node.setGridLims(xmin, xmax, ymin, ymax)

    init.loadCells(node, conf)



    ################################################## 
    # load values into cells
    injector.inject(node, filler, conf)

    insert_em(node, conf)


    #visualize initial condition
    plotAll(axs, node, conf, 0)

    vsol = pdev.AmrMomentumLagrangianSolver()
    
    print("solving momentum space push")
    cid  = node.cellId(0,0)
    cell = node.getCellPtr(cid)


    for lap in range(1,20):
        print("-------lap {} -------".format(lap))

        vsol.solve(cell)

        cell.cycle()

        if conf.clip:
            cell.clip()

        plotAll(axs, node, conf, lap)







