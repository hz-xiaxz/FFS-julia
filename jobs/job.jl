#!/usr/bin/env -S julia --color=yes --startup-file=none

using Carlo
using Carlo.JobTools
using FastFermionSampling
using Dates

tm = TaskMaker()
tm.thermalization = 0
tm.sweeps = 100000
tm.binsize = 100
tm.t = 1.0
tm.W = 1.0
tm.U = 1.0
tm.nx = 18
tm.ny = 18
ns = tm.nx * tm.ny
tm.N_up = ns ÷ 2
tm.N_down = ns ÷ 2
tm.B = "Periodic"

if tm.B == "Periodic"
    lat = LatticeRectangular(tm.nx, tm.ny, Periodic())
elseif tm.B == "Open"
    lat = LatticeRectangular(tm.nx, tm.ny, Open())
else
    throw(ArgumentError("Boundary condition not recognized"))
end


model = AHmodel(lat, tm.t, tm.W, tm.U, tm.N_up, tm.N_down)
task(tm, omega = model.omega)

dir = @__DIR__
# savepath = dir * "/../data/" * Dates.format(Dates.now(), "mm-ddTHH-MM-SS")
savepath = dir * "/../data/" * "mpirun$(tm.nx)x$(tm.ny)"
job = JobInfo(
    savepath,
    FastFermionSampling.MC;
    tasks = make_tasks(tm),
    checkpoint_time = "5:00",
    run_time = "24:00:00"
)

Carlo.start(job, ARGS)
