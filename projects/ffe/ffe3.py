# -*- coding: utf-8 -*-

# system libraries
from __future__ import print_function
from mpi4py import MPI
import numpy as np
import sys, os

# runko + auxiliary modules
import pytools  # runko python tools


# problem specific modules
from problem import Configuration_Reconnection as Configuration
from antenna2 import Antenna

np.random.seed(1)  # global simulation seed


# quick and dirty plotting
try:
    from visualize import get_yee
    import matplotlib.pyplot as plt
    from visualize import plotNode
    from visualize import plotJ, plotE
    from visualize import saveVisz

    from visualize import getYee2D
    from visualize import plot2dYee
except:
    pass


# Field initialization
def insert_em_fields(grid, conf):

    # into radians
    # btheta = conf.btheta / 180.0 * np.pi
    # bphi = conf.bphi / 180.0 * np.pi
    beta = conf.beta

    for cid in grid.get_tile_ids():
        tile = grid.get_tile(cid)
        yee = tile.get_yee(0)

        if conf.twoD:
            ii, jj = tile.index
            kk = 0
        elif conf.threeD:
            ii, jj, kk = tile.index

        # insert values into Yee lattices; includes halos from -3 to n+3
        for n in range(conf.NzMesh):
            for m in range(conf.NyMesh):
                for l in range(conf.NxMesh):
                    # get global coordinates
                    # iglob, jglob, kglob = pytools.ind2loc((ii, jj, kk), (l, m, n), conf)
                    # r = np.sqrt(iglob ** 2 + jglob ** 2 + kglob ** 2)

                    yee.bx[l, m, n] = 0.0  # conf.binit * np.cos(bphi)
                    yee.by[l, m, n] = 0.0  # conf.binit * np.sin(bphi) * np.sin(btheta)
                    yee.bz[l, m, n] = conf.binit  # * np.sin(bphi) * np.cos(btheta)

                    yee.ex[l, m, n] = 0.0
                    yee.ey[l, m, n] = -beta * yee.bz[l, m, n]
                    yee.ez[l, m, n] = beta * yee.by[l, m, n]
    return


# Field initialization
def insert_em_harris_sheet(grid, conf):
    from numpy import pi, tanh, sin, cos, sinh, cosh, sqrt

    delta = conf.sheet_thickness / (2.0 * pi)  # sheet thickness incl. 2pi
    pinch_delta = conf.pinch_width / (2.0 * pi)  # pinch thickness incl. 2pi
    eta = conf.sheet_density
    beta = 0.0  # conf.beta #sheet bulk flow; NOTE: for force-free setup no flow
    sigma = conf.sigma  # magnetization

    # field angles
    btheta = 1.0
    bphi = pi / 2.0  # conf.bphi/180. * pi

    # note: if periodicx then mxhalf is actually Lx/4 else Lx/2
    mxhalf = conf.mxhalf
    myhalf = conf.myhalf
    mzhalf = conf.mzhalf
    Lx = conf.lstripe

    # binit = conf.binit  # initial B_0
    binit = 0.1

    for tile in pytools.tiles_all(grid):
        ii, jj, kk = tile.index if conf.threeD else (*tile.index, 0)
        yee = tile.get_yee(0)

        # insert values into Yee lattices; includes halos from -3 to n+3
        for n in range(conf.NzMesh):
            for m in range(conf.NyMesh):
                for l in range(conf.NxMesh):

                    # get global coordinates
                    iglob, jglob, kglob = pytools.ind2loc((ii, jj, kk), (l, m, n), conf)

                    # trigger field modulation in z coordinate
                    if conf.threeD:
                        triggerz = cosh((kglob - mzhalf) / pinch_delta)  # 3D
                    else:
                        triggerz = 1.0

                    # inifnitely small thickness
                    if conf.sheet_thickness == 0.0:
                        if iglob <= mxhalf:
                            stripetanh = -1.0
                        elif mxhalf < iglob <= 3.0 * mxhalf:
                            stripetanh = +1.0
                        elif 3.0 * mxhalf < iglob:
                            stripetanh = -1.0

                        # one cell flip of zero
                        # if iglob == mxhalf or iglob == 3.*mxhalf:
                        #    stripetanh = 0.0

                    # flipping harris sheet stripe
                    else:
                        if not (conf.periodicx):
                            stripetanh = tanh((iglob - mxhalf) / delta)
                        else:
                            stripetanh = tanh(
                                Lx
                                * sin(2.0 * pi * (iglob - mxhalf) / Lx)
                                / delta
                                / 2.0
                                / pi
                            )

                    if conf.trigger:

                        # plasma bulk velocity modulation factor;
                        # NOTE: true velocity v/c= velstripe*beta

                        # velstripe = tanh((iglob-mxhalf)/pinch_delta)/cosh((iglob-mxhalf)/pinch_delta)
                        velstripe = tanh((iglob - mxhalf) / pinch_delta)
                        # velstripe = tanh((iglob - mxhalf) / pinch_delta) / (
                        #    cosh((jglob - myhalf) / pinch_delta)
                        #    * cosh((iglob - mxhalf) / pinch_delta)
                        # )

                        if conf.sheet_thickness == 0.0:
                            pinch_corr = cosh((jglob - myhalf) / pinch_delta) * triggerz
                            if iglob != mxhalf + 1:  # or iglob == 3.0 * mxhalf + 1:
                                pinch_corr = 1.0

                        else:
                            pinch_corr = (
                                cosh((jglob - myhalf) / pinch_delta)
                                * cosh((iglob - mxhalf) / delta)
                                * triggerz
                            )

                        # by
                        yee.by[l, m, n] = binit * sin(bphi) * stripetanh
                        yee.by[l, m, n] += (
                            binit
                            * cos(bphi)
                            * btheta
                            * cos(bphi)
                            * (1.0 - 1.0 / pinch_corr)
                        )

                        # bz
                        yee.bz[l, m, n] = binit * cos(bphi) * stripetanh
                        yee.bz[l, m, n] += (
                            binit * sin(bphi) * btheta * (1.0 - 1.0 / pinch_corr)
                        )

                        # ey
                        yee.ey[l, m, n] = (-beta) * velstripe * yee.bz[l, m, n]

                        # ez
                        yee.ez[l, m, n] = (+beta) * velstripe * yee.by[l, m, n]

                        # drive to trigger reconnection in the middle of the box;
                        # the coefficient should be the desired ExB speed
                        yee.ez[l, m, n] += (
                            conf.trigger_field * yee.by[l, m, n] / pinch_corr
                        )

                        # trigger point
                        if conf.sheet_thickness == 0.0:
                            if jglob == myhalf:
                                if iglob == mxhalf + 1 or iglob == 3.0 * mxhalf + 1:
                                    yee.ez[l, m, n] += conf.trigger_field

                    else:
                        yee.by[l, m, n] = binit * sin(bphi) * stripetanh
                        yee.by[l, m, n] += binit * cos(bphi) * btheta

                        yee.bz[l, m, n] = binit * cos(bphi) * stripetanh
                        yee.bz[l, m, n] += binit * sin(bphi) * btheta

                        yee.ey[l, m, n] = (-beta) * yee.bz[l, m, n]
                        yee.ez[l, m, n] = (+beta) * yee.by[l, m, n]

                    yee.ex[l, m, n] = 0.0

                    # add external non-evolving guide field
                    yee.bz[l, m, n] += binit * sqrt(conf.sigma_ext)

                    # one zell thin current sheet to balance the flip
                    # if iglob == mxhalf or iglob == 3.*mxhalf:
                    #    yee.ez[l, m, n] +=  binit
                    # if iglob == mxhalf+1 or iglob == 3.*mxhalf+1:
                    #    yee.ez[l, m, n] +=  binit

                    if False:
                        # hot current sheet
                        # beta_drift = sqrt(sigma)
                        beta_drift = 0.5
                        if not (conf.periodicx):
                            num_plasma = 1.0 / (cosh((iglob - mxhalf) / delta)) ** 2.0
                        else:
                            # num_plasma = 1.0/(cosh(dstripe*lstripe*sin(2.*pi*(iglob-mxhalf)/lstripe)))**2.*stripecosh
                            num_plasma = (
                                1.0
                                / cosh(
                                    Lx
                                    * sin(2.0 * pi * (iglob - mxhalf) / Lx)
                                    / delta
                                    / 2.0
                                    / pi
                                )
                                ** 2.0
                            )

                        gamma_drift = sqrt(1.0 / (1.0 - beta_drift ** 2.0))
                        if conf.periodicx:
                            gamma_drift = gamma_drift * np.sign(
                                cos(2.0 * pi * (iglob - mxhalf) / Lx)
                            )
                            beta_drift = sqrt(1.0 - 1.0 / gamma_drift ** 2) * np.sign(
                                gamma_drift
                            )

                        yee.ez[l, m, n] += beta_drift * num_plasma * binit

            # copy values to boundary cells
            # FIXME
            # try:
            #    for n in range(conf.NzMesh):
            #        for m in range(conf.NyMesh):
            #            for l in range(conf.NxMesh):
            #                c.ex_ref[l,m,n] = yee.ex[l,m,n]
            #                c.ey_ref[l,m,n] = yee.ey[l,m,n]
            #                c.ez_ref[l,m,n] = yee.ez[l,m,n]

            #                c.bx_ref[l,m,n] = yee.bx[l,m,n]
            #                c.by_ref[l,m,n] = yee.by[l,m,n]
            #                c.bz_ref[l,m,n] = yee.bz[l,m,n]
            # except:
            #    #print("cell ({},{}) is not boundary cell".format(ii,jj))
            #    pass
    return


# Field initialization
def insert_em_waves(grid, conf):

    # into radians
    # btheta = conf.btheta / 180.0 * np.pi
    # bphi = conf.bphi / 180.0 * np.pi
    # beta = conf.beta
    # beta = 0.1

    bpar = 1.0
    bperp = 1.0
    Lx = conf.NxMesh * conf.Nx
    beta = 0.1

    modes = 1.0
    kx = 2.0 * np.pi * modes / Lx

    for cid in grid.get_tile_ids():
        tile = grid.get_tile(cid)
        yee = tile.get_yee(0)

        if conf.twoD:
            ii, jj = tile.index
            kk = 0
        elif conf.threeD:
            ii, jj, kk = tile.index

        # insert values into Yee lattices; includes halos from -3 to n+3
        for n in range(conf.NzMesh):
            for m in range(conf.NyMesh):
                for l in range(conf.NxMesh):
                    # get global coordinates
                    iglob, jglob, kglob = pytools.ind2loc((ii, jj, kk), (l, m, n), conf)
                    # r = np.sqrt(iglob ** 2 + jglob ** 2 + kglob ** 2)

                    if True:
                        # 1D Alfven wave packet
                        yee.bx[l, m, n] = 0.0  # bpar
                        if 0 <= iglob <= 0.5 * Lx:
                            # yee.bz[l, m, n] = bperp*np.sin(kx*iglob)
                            # yee.ey[l, m, n] = bperp*np.sin(kx*iglob)

                            yee.by[l, m, n] = bperp * np.sin(kx * iglob)
                            yee.ez[l, m, n] = bperp * np.sin(kx * iglob)

                    # fast mode wave packet
                    if False:
                        if 0 <= iglob <= 0.5 * Lx:
                            yee.by[l, m, n] = bperp * np.sin(kx * iglob)
                            # yee.ez[l, m, n] = bperp*np.sin(kx*iglob)

                    if False:
                        # 1D Alfven wave packet collisions
                        yee.bx[l, m, n] = bpar
                        if 0 <= iglob < 0.5 * Lx:
                            yee.bz[l, m, n] = bperp * np.sin(kx * iglob)
                            yee.ey[l, m, n] = bperp * np.sin(kx * iglob)

                        if 0.5 * Lx <= iglob < Lx:
                            yee.bz[l, m, n] = bperp * np.sin(kx * iglob)
                            yee.ey[l, m, n] = -bperp * np.sin(kx * iglob)

    return


# Field initialization
def insert_em_3D_wave_packet(grid, conf):
    from numpy import arctan2, sin, cos, sqrt

    b0 = 1.0

    # bperp = 1.0
    # Lx = conf.NxMesh*conf.Nx
    # beta = 0.1
    # modes = 1.
    # kx = 2.0*np.pi*modes/Lx

    # middle of the box
    x0 = conf.Nx * conf.NxMesh * 0.5
    y0 = conf.Ny * conf.NyMesh * 0.5


    z1 = 0.0  # conf.Ny*conf.NyMesh*0.25
    z2 = conf.Ny * conf.NyMesh * 0.75

    print("x0 {} {} {}".format(x0, y0, z1))


    # position of the centers as Stagger objects
    pkg_loc1 = pytools.Stagger(x0, y0, z1)
    pkg_loc2 = pytools.Stagger(x0, y0, z2)

    zeta = 1.0  # perturbation amplitude
    ell = 10.0  # perturbation length

    for cid in grid.get_tile_ids():
        tile = grid.get_tile(cid)
        yee = tile.get_yee(0)

        if conf.twoD:
            ii, jj = tile.index
            kk = 0
        elif conf.threeD:
            ii, jj, kk = tile.index

        # insert values into Yee lattices; includes halos from -3 to n+3
        for n in range(conf.NzMesh):
            for m in range(conf.NyMesh):
                for l in range(conf.NxMesh):
                    # get global coordinates
                    iglob, jglob, kglob = pytools.ind2loc((ii, jj, kk), (l, m, n), conf)
                    loc = pytools.Stagger(iglob, jglob, kglob)

                    # distance from the center of the packets
                    d1 = pkg_loc1 - loc
                    d2 = pkg_loc2 - loc

                    # add stationary guide field
                    yee.bz[l, m, n] = b0

                    # amplitude for the perturbations
                    #
                    # gauss = zeta * ell * np.exp(-r1 / ell ** 2)
                    # bphi = (2.0 * b0 * w / ell ** 2) * gauss

                    # bphi = {}
                    # theta = {}

                    bpkg1 = {}
                    bpkg2 = {}

                    # build exact initial amplitude for different staggered grid locations
                    for st in [
                        "rh",
                        "bx",
                        "by",
                        "bz",
                        "ex",
                        "ey",
                        "ez",
                        #'jx', 'jy', 'jz',
                    ]:

                        # spherical distance
                        r1 = d1.at(st).x ** 2 + d1.at(st).y ** 2 + d1.at(st).z ** 2
                        r2 = d2.at(st).x ** 2 + d2.at(st).y ** 2 + d2.at(st).z ** 2

                        gauss_profile1 = zeta * ell * np.exp(-r1 / ell ** 2)
                        gauss_profile2 = zeta * ell * np.exp(-r2 / ell ** 2)

                        bpkg1[st] = (2 * b0 / ell ** 2) * gauss_profile1
                        bpkg2[st] = (2 * b0 / ell ** 2) * gauss_profile2

                    # add fields (staggered)
                    yee.bx[l, m, n] = -bpkg1["bx"] * d1.at("bx").y
                    yee.by[l, m, n] = +bpkg1["by"] * d1.at("by").x

                    yee.ex[l, m, n] = -bpkg1["ex"] * d1.at("ex").x
                    yee.ey[l, m, n] = -bpkg1["ey"] * d1.at("ey").y

                    # non staggered
                    #st = 'rh'
                    #yee.bx[l, m, n] = -bpkg1[st] * d1.at(st).y
                    #yee.by[l, m, n] = +bpkg1[st] * d1.at(st).x

                    #yee.ex[l, m, n] = -bpkg1[st] * d1.at(st).x
                    #yee.ey[l, m, n] = -bpkg1[st] * d1.at(st).y
    return


def plot_waves(ax, yee, mode):

    ax.cla()
    ax.set_ylim((-1, 1))

    if mode == "x":
        e = yee["ex"]
        b = yee["bx"]
        j = yee["jx"]
    if mode == "y":
        e = yee["ey"]
        b = yee["by"]
        j = yee["jy"]
    if mode == "z":
        e = yee["ez"]
        b = yee["bz"]
        j = yee["jz"]

    ax.plot(e, "b")
    ax.plot(b, "r")
    ax.plot(j, "g")


if __name__ == "__main__":

    ##################################################
    # set up plotting and figure
    # TODO: remove
    do_plots = True
    try:
        plconf = dict(curval=0.05, elval=0.2, bfval=0.4)
        if do_plots:
            plt.fig = plt.figure(1, figsize=(8, 10))
            plt.rc("font", family="serif", size=12)
            plt.rc("xtick")
            plt.rc("ytick")

            gs = plt.GridSpec(4, 3)
            gs.update(hspace=0.5)

            axs = []
            for ai in range(12):
                axs.append(plt.subplot(gs[ai]))
    except:
        # print()
        pass

    # --------------------------------------------------
    # initial setup
    do_print = False
    if MPI.COMM_WORLD.Get_rank() == 0:
        do_print = True

    if do_print:
        print("Running ffe.py with {} MPI processes.".format(MPI.COMM_WORLD.Get_size()))

    # --------------------------------------------------
    # Timer for profiling
    timer = pytools.Timer()

    timer.start("total")
    timer.start("init")
    timer.do_print = do_print

    # --------------------------------------------------
    # parse command line arguments
    args = pytools.parse_args()

    # create conf object with simulation parameters based on them
    conf = Configuration(args.conf_filename, do_print=do_print)

    # --------------------------------------------------
    # load runko

    if conf.threeD:
        # 3D modules
        import pycorgi.threeD as pycorgi  # corgi ++ bindings
        import pyrunko.ffe.threeD as pyffe  # runkko ffe c++ bindings
        import pyrunko.fields.threeD as pyfld  # runko fld c++ bindings

    elif conf.twoD:
        # 2D modules
        import pycorgi.twoD as pycorgi  # corgi ++ bindings
        import pyrunko.ffe.twoD as pyffe  # runkko ffe c++ bindings
        import pyrunko.fields.twoD as pyfld  # runko fld c++ bindings

    # --------------------------------------------------
    # setup grid
    grid = pycorgi.Grid(conf.Nx, conf.Ny, conf.Nz)
    grid.set_grid_lims(conf.xmin, conf.xmax, conf.ymin, conf.ymax, conf.zmin, conf.zmax)

    # compute initial mpi ranks using Hilbert's curve partitioning
    pytools.balance_mpi(grid, conf)

    # load ffe tiles into grid
    pytools.ffe.load_tiles(grid, conf)

    # --------------------------------------------------
    # simulation restart

    # create output folders
    if grid.master():
        pytools.create_output_folders(conf)

    # get current restart file status
    io_stat = pytools.check_for_restart(conf)

    # no restart file; initialize simulation
    if io_stat["do_initialization"]:
        if do_print:
            print("initializing simulation...")
        lap = 0

        np.random.seed(1)  # sync rnd generator seed for different mpi ranks

        # inserting em grid
        # insert_em_fields(grid, conf)
        # insert_em_harris_sheet(grid, conf)
        # insert_em_waves(grid, conf)
        insert_em_3D_wave_packet(grid, conf)

    else:
        if do_print:
            print("restarting simulation from lap {}...".format(io_stat["lap"]))

        # read restart files
        pyfld.read_yee(grid, io_stat["read_lap"], io_stat["read_dir"])

        # step one step ahead
        lap = io_stat["lap"] + 1

    # --------------------------------------------------
    # static load balancing setup; communicate neighborhood info once

    grid.analyze_boundaries()
    grid.send_tiles()
    grid.recv_tiles()
    MPI.COMM_WORLD.barrier()

    # load virtual mpi halo tiles
    pytools.ffe.load_virtual_tiles(grid, conf)

    # --------------------------------------------------
    # load physics solvers

    # reduced 2nd order FFE algorithm
    algo = pyffe.rFFE2(conf.NxMesh, conf.NyMesh, conf.NzMesh)

    # --------------------------------------------------
    # I/O objects

    # quick field snapshots
    fld_writer = pyfld.FieldsWriter(
        conf.outdir,
        conf.Nx,
        conf.NxMesh,
        conf.Ny,
        conf.NyMesh,
        conf.Nz,
        conf.NzMesh,
        conf.stride,
    )

    ##################################################
    # Langeving antenna
    if io_stat["do_initialization"]:
        # direct B_{x,y} perturbation
        if False:
            conf.min_mode = 1
            conf.max_mode = 4
            conf.drive_ampl = 1.0
            antenna = Antenna(conf.min_mode, conf.max_mode, conf)
            for tile in pytools.tiles_local(grid):
                antenna.add_driving(tile)

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------
    # end of initialization

    timer.stop("init")
    timer.stats("init")
    # timer.verbose = 1  # 0 normal; 1 - debug mode

    # --------------------------------------------------
    # sync e and b fields

    # mpi e
    grid.send_data(1)
    grid.recv_data(1)
    grid.wait_data(1)

    # mpi b
    grid.send_data(2)
    grid.recv_data(2)
    grid.wait_data(2)

    for tile in pytools.tiles_all(grid):
        tile.update_boundaries(grid)

    ##################################################
    # simulation time step loop

    sys.stdout.flush()

    if True:
        if do_plots:
            # plotNode(axs[0], grid, conf)

            yee = getYee2D(grid, conf)
            plot2dYee(
                axs[3],
                yee,
                grid,
                conf,
                "jx",
                vmin=-plconf["curval"],
                vmax=+plconf["curval"],
            )
            plot2dYee(
                axs[4],
                yee,
                grid,
                conf,
                "jy",
                vmin=-plconf["curval"],
                vmax=+plconf["curval"],
            )
            plot2dYee(
                axs[5],
                yee,
                grid,
                conf,
                "jz",
                vmin=-plconf["curval"],
                vmax=+plconf["curval"],
            )

            plot2dYee(
                axs[6],
                yee,
                grid,
                conf,
                "ex",
                vmin=-plconf["elval"],
                vmax=+plconf["elval"],
            )
            plot2dYee(
                axs[7],
                yee,
                grid,
                conf,
                "ey",
                vmin=-plconf["elval"],
                vmax=+plconf["elval"],
            )
            plot2dYee(
                axs[8],
                yee,
                grid,
                conf,
                "ez",
                vmin=-plconf["elval"],
                vmax=+plconf["elval"],
            )

            plot2dYee(
                axs[9],
                yee,
                grid,
                conf,
                "bx",
                vmin=-plconf["bfval"],
                vmax=+plconf["bfval"],
            )
            plot2dYee(
                axs[10],
                yee,
                grid,
                conf,
                "by",
                vmin=-plconf["bfval"],
                vmax=+plconf["bfval"],
            )
            plot2dYee(
                axs[11],
                yee,
                grid,
                conf,
                "bz",
                vmin=-plconf["bfval"],
                vmax=+plconf["bfval"],
            )
            saveVisz(-1, grid, conf)

    # simulation loop
    time = lap * (conf.cfl / conf.c_omp)
    for lap in range(lap, conf.Nt + 1):

        # initialize Y^n-1 = Y
        t1 = timer.start_comp("copy_eb")
        for tile in pytools.tiles_all(grid):
            algo.copy_eb(tile)
        timer.stop_comp(t1)

        ###################################################
        # rk steps
        rks = 0

        # RK1
        # rk_coeffs = [(1.0,   0.0,   1.0,   1.0),]

        # RK2
        # rk_coeffs = [(1.0,   0.0,   1.0,   1.0),
        #             (0.5,   0.5,   0.5,   1.0),]

        # RK3
        rk_coeffs = [
            (1.0, 0.0, 1.0, 1.0),
            (0.75, 0.25, 0.25, 0.5),
            (1 / 3, 2 / 3, 2 / 3, 1.0),
        ]

        for (rk_c1, rk_c2, rk_c3, rk_dt) in rk_coeffs:

            # RK substep
            rks += 1

            # rho = div E
            t1 = timer.start_comp("comp_rho")
            for tile in pytools.tiles_local(grid):
                algo.comp_rho(tile)
            timer.stop_comp(t1)

            # dE = dt * curl B
            # dB = dt * curl E
            t1 = timer.start_comp("push_eb")
            for tile in pytools.tiles_local(grid):
                algo.push_eb(tile)
            timer.stop_comp(t1)

            # drift current j_perp
            # dE -= dt*j_perp
            t1 = timer.start_comp("add_jperp")
            for tile in pytools.tiles_local(grid):
                algo.add_jperp(tile)
            timer.stop_comp(t1)

            # update fields according to RK scheme
            # Y^n+1 = c1 * Y^n-1 + c2 * Y^n + c3 * dY
            t1 = timer.start_comp("update_eb")
            for tile in pytools.tiles_local(grid):
                algo.update_eb(tile, rk_c1, rk_c2, rk_c3)
            timer.stop_comp(t1)

            if True:

                # comm e & b
                t1 = timer.start_comp("mpi_eb3")
                grid.send_data(1)
                grid.recv_data(1)

                grid.send_data(2)
                grid.recv_data(2)

                grid.wait_data(1)
                grid.wait_data(2)
                timer.stop_comp(t1)

                # update boundaries
                t1 = timer.start_comp("upd_bc3")
                for tile in pytools.tiles_local(grid):
                    tile.update_boundaries(grid)
                timer.stop_comp(t1)

                # parallel current j_par
                # dE -= j_par
                t1 = timer.start_comp("remove_jpar")
                for tile in pytools.tiles_local(grid):
                    algo.remove_jpar(tile)
                timer.stop_comp(t1)

            if True:
                # comm E
                t1 = timer.start_comp("mpi_eb1")
                grid.send_data(1)
                grid.recv_data(1)

                grid.send_data(2)
                grid.recv_data(2)

                grid.wait_data(1)
                grid.wait_data(2)
                timer.stop_comp(t1)

                # update boundaries
                t1 = timer.start_comp("upd_bc1")
                for tile in pytools.tiles_local(grid):
                    tile.update_boundaries(grid)
                timer.stop_comp(t1)

                # enforce E < B
                # dE = dE_lim
                t1 = timer.start_comp("limit_e")
                for tile in pytools.tiles_local(grid):
                    algo.limit_e(tile)
                timer.stop_comp(t1)

            ##################################################
            # TODO: boundary conditions

            ##################################################
            # update field halos

            # comm e & b
            t1 = timer.start_comp("mpi_eb2")
            grid.send_data(1)
            grid.recv_data(1)

            grid.send_data(2)
            grid.recv_data(2)

            grid.wait_data(1)
            grid.wait_data(2)
            timer.stop_comp(t1)

            # update boundaries
            t1 = timer.start_comp("upd_bc2")
            for tile in pytools.tiles_local(grid):
                tile.update_boundaries(grid)
            timer.stop_comp(t1)

        ##################################################
        # data reduction and I/O

        timer.lap("step")
        if lap % conf.interval == 0:
            if do_print:
                print("--------------------------------------------------")
                print("------ lap: {} / t: {}".format(lap, time))

            timer.stats("step")
            timer.comp_stats()
            timer.purge_comps()

            # analyze (independent)
            timer.start("io")

            # shallow IO
            fld_writer.write(grid, lap)  # quick field snapshots

            # deep IO
            if (
                conf.full_interval != -1
                and (lap % conf.full_interval == 0)
                and (lap > 0)
            ):
                pyfld.write_yee(grid, lap, conf.outdir + "/full_output/")

            # restart IO (overwrites)
            if (lap % conf.restart == 0) and (lap > 0):
                # flip between two sets of files
                io_stat["deep_io_switch"] = 1 if io_stat["deep_io_switch"] == 0 else 0

                pyfld.write_yee(
                    grid, io_stat["deep_io_switch"], conf.outdir + "/restart/"
                )

                # if successful adjust info file
                MPI.COMM_WORLD.barrier()
                if grid.rank() == 0:
                    with open(conf.outdir + "/restart/laps.txt", "a") as lapfile:
                        lapfile.write("{},{}\n".format(lap, io_stat["deep_io_switch"]))

            # --------------------------------------------------
            # 2D plots
            # try:
            if True:
                if do_plots:
                    # plotNode(axs[0], grid, conf)

                    yee = getYee2D(grid, conf)

                    # plot_waves(axs[0], yee, 'x')
                    plot_waves(axs[1], yee, "y")
                    plot_waves(axs[2], yee, "z")

                    plot2dYee(
                        axs[0],
                        yee,
                        grid,
                        conf,
                        "rho",
                        # vmin=-plconf["curval"],
                        # vmax=+plconf["curval"],
                    )

                    plot2dYee(
                        axs[3],
                        yee,
                        grid,
                        conf,
                        "jx",
                        vmin=-plconf["curval"],
                        vmax=+plconf["curval"],
                    )
                    plot2dYee(
                        axs[4],
                        yee,
                        grid,
                        conf,
                        "jy",
                        vmin=-plconf["curval"],
                        vmax=+plconf["curval"],
                    )
                    plot2dYee(
                        axs[5],
                        yee,
                        grid,
                        conf,
                        "jz",
                        vmin=-plconf["curval"],
                        vmax=+plconf["curval"],
                    )

                    plot2dYee(
                        axs[6],
                        yee,
                        grid,
                        conf,
                        "ex",
                        vmin=-plconf["elval"],
                        vmax=+plconf["elval"],
                    )
                    plot2dYee(
                        axs[7],
                        yee,
                        grid,
                        conf,
                        "ey",
                        vmin=-plconf["elval"],
                        vmax=+plconf["elval"],
                    )
                    plot2dYee(
                        axs[8],
                        yee,
                        grid,
                        conf,
                        "ez",
                        vmin=-plconf["elval"],
                        vmax=+plconf["elval"],
                    )

                    plot2dYee(
                        axs[9],
                        yee,
                        grid,
                        conf,
                        "bx",
                        vmin=-plconf["bfval"],
                        vmax=+plconf["bfval"],
                    )
                    plot2dYee(
                        axs[10],
                        yee,
                        grid,
                        conf,
                        "by",
                        vmin=-plconf["bfval"],
                        vmax=+plconf["bfval"],
                    )
                    plot2dYee(
                        axs[11],
                        yee,
                        grid,
                        conf,
                        "bz",
                        vmin=-plconf["bfval"],
                        vmax=+plconf["bfval"],
                    )
                    saveVisz(lap, grid, conf)

            # except:
            #    print()
            #    pass
            timer.stop("io")

            timer.stats("io")
            timer.start("step")  # refresh lap counter (avoids IO profiling)

            sys.stdout.flush()

        # MPI.COMM_WORLD.barrier()
        # sleep(0.2)
        time += conf.cfl / conf.c_omp
    # end of loop

    timer.stop("total")
    timer.stats("total")
