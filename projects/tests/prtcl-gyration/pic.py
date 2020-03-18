from __future__ import print_function
from mpi4py import MPI

import numpy as np
import sys, os
import h5py

import pycorgi.twoD as corgi
import pyrunko.tools.twoD as pytools
import pyrunko.vlv.twoD as pyvlv
import pyrunko.pic.twoD as pypic
import pyrunko.fields.twoD as pyfld


from configSetup import Configuration
import argparse
import initialize as init
from initialize_pic import loadTiles
from initialize_pic import initialize_virtuals
from initialize_pic import globalIndx
from sampling import boosted_maxwellian
from initialize_pic import spatialLoc
from injector_pic import inject
from injector_pic import insert_em
from time import sleep
from visualize import get_yee

#global simulation seed
np.random.seed(1)

try:
    import matplotlib.pyplot as plt
    from visualize import plotNode
    from visualize import plotJ, plotE, plotDens
    from visualize import saveVisz
    
    from visualize import getYee2D
    from visualize import plot2dYee
    from visualize_pic import plot2dParticles

except:
    pass

from timer import Timer

from init_problem import Configuration_Gyro as Configuration

debug = False

def debug_print(n, msg):
    if debug:
        print("{}: {}".format(n.rank(), msg))
        sys.stdout.flush()


def filler(xloc, ispcs, conf):
    # perturb position between x0 + RUnif[0,1)

    #electrons
    if ispcs == 0:
        delgam  = conf.delgam #* np.abs(conf.mi / conf.me) * conf.temp_ratio

        xx = xloc[0] + np.random.rand(1)
        yy = xloc[1] + np.random.rand(1)
        #zz = xloc[2] + np.random.rand(1)
        zz = 0.5

    #positrons/ions/second species
    if ispcs == 1:
        delgam  = conf.delgam

        #on top of electrons
        xx = xloc[0]
        yy = xloc[1]
        zz = 0.5

    gamma = conf.gamma
    direction = -1
    ux, uy, uz, uu = boosted_maxwellian(delgam, gamma, direction=direction, dims=3)

    x0 = [xx, yy, zz]
    u0 = [ux, uy, uz]
    return x0, u0


def direct_inject(grid, conf):
    cid = grid.id(0,0)
    c = grid.get_tile(cid)
    container = c.get_container(0)

    x = conf.x_start
    y = conf.NyMesh/2.
    z = 0.5
    x0 = [x,y,z]
    
    vx = 0
    vy = conf.vy
    vz = 0
    u0 = [vx,vy,vz]

    container.add_particle(x0,u0,1.0)


# Field initialization (guide field)
def insert_em(grid, conf):

    #into radians
    btheta = conf.btheta/180.*np.pi
    bphi   = conf.bphi/180.*np.pi
    beta   = conf.beta

    kk = 0
    for cid in grid.get_tile_ids():
        tile = grid.get_tile(cid)
        yee = tile.get_yee(0)

        ii,jj = tile.index

        for n in range(conf.NzMesh):
            for m in range(-3, conf.NyMesh+3):
                for l in range(-3, conf.NxMesh+3):
                    # get global coordinates
                    iglob, jglob, kglob = globalIndx( (ii,jj), (l,m,n), conf)

                    yee.bx[l,m,n] = 0. #conf.binit*np.cos(btheta) 
                    yee.by[l,m,n] = 0. #conf.binit*np.sin(btheta)*np.sin(bphi)
                    yee.bz[l,m,n] = 1.0 #conf.binit*np.sin(btheta)*np.cos(bphi)   

                    yee.ex[l,m,n] = 0.0
                    yee.ey[l,m,n] = 0. #-beta*yee.bz[l,m,n]
                    yee.ez[l,m,n] = 0. #beta*yee.by[l,m,n]


if __name__ == "__main__":

    x = []
    y = []
    vx = []
    vy = []
    t = []

    do_plots = True
    do_print = False

    if MPI.COMM_WORLD.Get_rank() == 0:
        do_print =True

    if do_print:
        print("Running with {} MPI processes.".format(MPI.COMM_WORLD.Get_size()))

    ################################################## 
    # set up plotting and figure
    try:
        if do_plots:
            pass
    except:
        #print()
        pass


    # Timer for profiling
    timer = Timer()
    timer.start("total")
    timer.start("init")

    timer.do_print = do_print


    # parse command line arguments
    parser = argparse.ArgumentParser(description='Simple PIC-Maxwell simulations')
    parser.add_argument('--conf', dest='conf_filename', default=None,
                       help='Name of the configuration file (default: None)')
    args = parser.parse_args()
    if args.conf_filename == None:
        conf = Configuration('gyration.ini', do_print=do_print) 
    else:
        if do_print:
            print("Reading configuration setup from ", args.conf_filename)
        conf = Configuration(args.conf_filename, do_print=do_print)


    grid = corgi.Grid(conf.Nx, conf.Ny, conf.Nz)

    xmin = 0.0
    xmax = conf.Nx*conf.NxMesh #XXX scaled length
    ymin = 0.0
    ymax = conf.Ny*conf.NyMesh
    grid.set_grid_lims(xmin, xmax, ymin, ymax)

    #init.loadMpiRandomly(grid)
    #init.loadMpiXStrides(grid)
    debug_print(grid, "load mpi 2d")
    init.loadMpi2D(grid)
    debug_print(grid, "load tiles")
    loadTiles(grid, conf)

    ################################################## 
    # Path to be created 
    if grid.master():
        if not os.path.exists( conf.outdir ):
            os.makedirs(conf.outdir)
        if not os.path.exists( conf.outdir+"/restart" ):
            os.makedirs(conf.outdir+"/restart")
        if not os.path.exists( conf.outdir+"/full_output" ):
            os.makedirs(conf.outdir+"/full_output")

    do_initialization = True

    #check if this is the first time and we do not have any restart files
    if not os.path.exists( conf.outdir+'/restart/laps.txt'):
        conf.laprestart = -1 #to avoid next if statement

    # restart from latest file
    deep_io_switch = 0
    if conf.laprestart >= 0:
        do_initialization = False

        #switch between automatic restart and user-defined lap
        if conf.laprestart == 0:

            #get latest restart file from housekeeping file
            with open(conf.outdir+"/restart/laps.txt", "r") as lapfile:
                #lapfile.write("{},{}\n".format(lap, deep_io_switch))
                lines = lapfile.readlines()
                slap, sdeep_io_switch = lines[-1].strip().split(',')
                lap = int(slap)
                deep_io_switch = int(sdeep_io_switch)

            read_lap = deep_io_switch
            odir = conf.outdir + '/restart'

        elif conf.laprestart > 0:
            lap = conf.laprestart
            read_lap = lap
            odir = conf.outdir + '/full_output'

        debug_print(grid, "read")
        if do_print:
            print("...reading Yee lattices (lap {}) from {}".format(read_lap, odir))
        pyvlv.read_yee(grid, read_lap, odir)

        if do_print:
            print("...reading particles (lap {}) from {}".format(read_lap, odir))
        pyvlv.read_particles(grid, read_lap, odir)

        lap += 1 #step one step ahead

    # initialize
    if do_initialization:
        debug_print(grid, "inject")
        lap = 0
        np.random.seed(1)
        #inject(grid, filler, conf) injecting plasma particles
        direct_inject(grid,conf) #inject plasma particles individually by loc,vel
        insert_em(grid, conf)

    #static load balancing setup; communicate neighbor info once
    debug_print(grid, "analyze bcs")
    grid.analyze_boundaries()
    debug_print(grid, "send tiles")
    grid.send_tiles()
    debug_print(grid, "recv tiles")
    grid.recv_tiles()
    MPI.COMM_WORLD.barrier()

    #sys.exit()

    debug_print(grid, "init virs")
    initialize_virtuals(grid, conf)


    timer.stop("init") 
    timer.stats("init") 


    # end of initialization
    ################################################## 
    debug_print(grid, "solvers")


    # visualize initial condition
    if do_plots:
        try:
            plotNode( axs[0], grid, conf)
            #plotXmesh(axs[1], grid, conf, 0, "x")
            saveVisz(-1, grid, conf)
        except:
            pass


    Nsamples = conf.Nt
    #pusher   = pypic.BorisPusher()
    pusher   = pypic.VayPusher()


    #fldprop  = pyfld.FDTD2()
    fldprop  = pyfld.FDTD4()
    fintp    = pypic.LinearInterpolator()
    currint  = pypic.ZigZag()
    analyzer = pypic.Analyzator()
    flt      = pyfld.Binomial2(conf.NxMesh, conf.NyMesh, conf.NzMesh)

    #enhance numerical speed of light slightly to suppress numerical Cherenkov instability
    fldprop.corr = 1.02

    debug_print(grid, "mpi_e")
    grid.send_data(1) 
    grid.recv_data(1) 
    grid.wait_data(1) 

    debug_print(grid, "mpi_b")
    grid.send_data(2) 
    grid.recv_data(2) 
    grid.wait_data(2) 

    ################################################## 
    sys.stdout.flush()

    #simulation loop
    time = lap*(conf.cfl/conf.c_omp)
    for lap in range(lap, conf.Nt+1):
        debug_print(grid, "lap_start")

        ################################################## 
        # advance Half B

        #--------------------------------------------------
        # comm B
        timer.start_comp("mpi_b1")
        debug_print(grid, "mpi_b1")

        grid.send_data(2) 
        grid.recv_data(2) 
        grid.wait_data(2) 

        timer.stop_comp("mpi_b1")

        #--------------------------------------------------
        #update boundaries
        timer.start_comp("upd_bc0")
        debug_print(grid, "upd_bc0")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            tile.update_boundaries(grid)

        timer.stop_comp("upd_bc0")

        #--------------------------------------------------
        #push B half
        timer.start_comp("push_half_b1")
        debug_print(grid, "push_half_b1")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            fldprop.push_half_b(tile)

        timer.stop_comp("push_half_b1")

        #--------------------------------------------------
        # comm B
        timer.start_comp("mpi_b2")
        debug_print(grid, "mpi_b2")

        grid.send_data(2) 
        grid.recv_data(2) 
        grid.wait_data(2) 

        timer.stop_comp("mpi_b2")

        #--------------------------------------------------
        #update boundaries
        timer.start_comp("upd_bc1")
        debug_print(grid, "upd_bc1")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            tile.update_boundaries(grid)

        timer.stop_comp("upd_bc1")


        ################################################## 
        # move particles (only locals tiles)

        #--------------------------------------------------
        #interpolate fields (can move to next asap)
        timer.start_comp("interp_em")
        debug_print(grid, "interp_em")

        for cid in grid.get_local_tiles():
            tile = grid.get_tile(cid)
            fintp.solve(tile)

        timer.stop_comp("interp_em")
        #--------------------------------------------------

        #--------------------------------------------------
        #push particles in x and u
        timer.start_comp("push")
        debug_print(grid, "push")

        for cid in grid.get_local_tiles():
            tile = grid.get_tile(cid)
            pusher.solve(tile)

        timer.stop_comp("push")


        # advance B half

        #--------------------------------------------------
        #push B half
        timer.start_comp("push_half_b2")
        debug_print(grid, "push_half_b2")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            fldprop.push_half_b(tile)

        timer.stop_comp("push_half_b2")


        #--------------------------------------------------
        # comm B
        timer.start_comp("mpi_e1")
        debug_print(grid, "mpi_e1")

        grid.send_data(1) 
        grid.recv_data(1) 
        grid.wait_data(1) 

        timer.stop_comp("mpi_e1")

        #--------------------------------------------------
        #update boundaries
        timer.start_comp("upd_bc2")
        debug_print(grid, "upd_bc2")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            tile.update_boundaries(grid)

        timer.stop_comp("upd_bc2")


        ##################################################
        # advance E 

        #--------------------------------------------------
        #push E
        timer.start_comp("push_e")
        debug_print(grid, "push_e")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            fldprop.push_e(tile)

        timer.stop_comp("push_e")


        ##################################################
        # particle communication (only local/boundary tiles)

        #--------------------------------------------------
        #local particle exchange (independent)
        timer.start_comp("check_outg_prtcls")
        debug_print(grid, "check_outg_prtcls")

        for cid in grid.get_local_tiles():
            tile = grid.get_tile(cid)
            tile.check_outgoing_particles()

        timer.stop_comp("check_outg_prtcls")

        #--------------------------------------------------
        # global mpi exchange (independent)
        timer.start_comp("pack_outg_prtcls")
        debug_print(grid, "pack_outg_prtcls")

        for cid in grid.get_boundary_tiles():
            tile = grid.get_tile(cid)
            tile.pack_outgoing_particles()

        timer.stop_comp("pack_outg_prtcls")

        #--------------------------------------------------
        # MPI global particle exchange
        # transfer primary and extra data
        timer.start_comp("mpi_prtcls")
        debug_print(grid, "mpi_prtcls")

        debug_print(grid, "mpi_prtcls: send3")
        grid.send_data(3) 

        debug_print(grid, "mpi_prtcls: recv3")
        grid.recv_data(3) 

        debug_print(grid, "mpi_prtcls: wait3")
        grid.wait_data(3) 

        # orig just after send3
        debug_print(grid, "mpi_prtcls: send4")
        grid.send_data(4) 

        debug_print(grid, "mpi_prtcls: recv4")
        grid.recv_data(4) 

        debug_print(grid, "mpi_prtcls: wait4")
        grid.wait_data(4) 

        timer.stop_comp("mpi_prtcls")

        #--------------------------------------------------
        # global unpacking (independent)
        timer.start_comp("unpack_vir_prtcls")
        debug_print(grid, "unpack_vir_prtcls")

        for cid in grid.get_virtual_tiles(): 
            tile = grid.get_tile(cid)
            tile.unpack_incoming_particles()
            tile.check_outgoing_particles()

        timer.stop_comp("unpack_vir_prtcls")

        #--------------------------------------------------
        # transfer local + global
        timer.start_comp("get_inc_prtcls")
        debug_print(grid, "get_inc_prtcls")

        for cid in grid.get_local_tiles():
            tile = grid.get_tile(cid)
            tile.get_incoming_particles(grid)

        timer.stop_comp("get_inc_prtcls")

        #--------------------------------------------------
        # delete local transferred particles
        timer.start_comp("del_trnsfrd_prtcls")
        debug_print(grid, "del_trnsfrd_prtcls")

        for cid in grid.get_local_tiles():
            tile = grid.get_tile(cid)
            tile.delete_transferred_particles()

        timer.stop_comp("del_trnsfrd_prtcls")

        #--------------------------------------------------
        # delete all virtual particles (because new prtcls will come)
        timer.start_comp("del_vir_prtcls")
        debug_print(grid, "del_vir_prtcls")

        for cid in grid.get_virtual_tiles(): 
            tile = grid.get_tile(cid)
            tile.delete_all_particles()

        timer.stop_comp("del_vir_prtcls")



        ##################################################
        #filter
        timer.start_comp("filter")

        #sweep over npasses times
        for fj in range(conf.npasses):

            #update global neighbors (mpi)
            grid.send_data(0)
            grid.recv_data(0) 
            grid.wait_data(0)

            #get halo boundaries
            for cid in grid.get_local_tiles():
                tile = grid.get_tile(cid)
                tile.update_boundaries(grid)

            #filter each tile
            for cid in grid.get_local_tiles():
                tile = grid.get_tile(cid)
                flt.solve(tile)

            MPI.COMM_WORLD.barrier() # sync everybody 


        #--------------------------------------------------
        timer.stop_comp("filter")


        #--------------------------------------------------
        #add current to E
        timer.start_comp("add_cur")
        debug_print(grid, "add_cur")

        for cid in grid.get_tile_ids():
            tile = grid.get_tile(cid)
            tile.deposit_current()

        timer.stop_comp("add_cur")

        #comm E
        timer.start_comp("mpi_e2")
        debug_print(grid, "mpi_e2")

        grid.send_data(1) 
        grid.recv_data(1) 
        grid.wait_data(1) 

        timer.stop_comp("mpi_e2")


        ##################################################
        # data reduction and I/O
	
        cid = grid.id(0,0)
        c = grid.get_tile(cid)
        container = c.get_container(0)

        x.append(container.loc(0))
        y.append(container.loc(1))
        vx.append(container.vel(0))
        vy.append(container.vel(1))
        t.append(time)


        timer.lap("step")
        if (lap % conf.interval == 0):
            debug_print(grid, "io")
            if do_print:
                print("--------------------------------------------------")
                print("------ lap: {} / t: {}".format(lap, time)) 

            print("------------------------------------------------------")
            print("x-position:" + str(x[lap]))
            print("y-position:" + str(y[lap]))
            print("x-vel:" + str(vx[lap]))
            print("y-vel:" + str(vy[lap]))
            print("------------------------------------------------------")

            #for cid in grid.get_tile_ids():
            #    tile = grid.get_tile(cid)
            #    tile.erase_temporary_arrays()

            timer.stats("step")
            timer.comp_stats()
            timer.purge_comps()
            
            #analyze (independent)
            timer.start("io")


            #--------------------------------------------------
            #2D plots
            if do_plots:
                try:
                    pass
			
                except:
                    #print()
                    pass
            timer.stop("io")


            timer.stats("io")
            timer.start("step") #refresh lap counter (avoids IO profiling)

            sys.stdout.flush()

        #next step
        time += conf.cfl/conf.c_omp
    #end of loop

    timer.stop("total")
    timer.stats("total")

    fig_traj = plt.figure(1)
    ax_traj = fig_traj.add_subplot(111)
    ax_traj.plot(x,y)

    fig_vel = plt.figure(2)
    ax_vel = fig_vel.add_subplot(111)
    ax_vel.plot(t,vx,t,vy)
                    

    fig_traj.savefig(str(pusher)+'_trajectory.png')
    fig_vel.savefig(str(pusher)+'_velocity.png')