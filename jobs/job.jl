#!/usr/bin/env -S julia --color=yes --startup-file=non

using Carlo
using Carlo.JobTools
using FastFermionSampling
using Dates

tm = TaskMaker()
tm.thermalization = 0
tm.sweeps = 100
tm.binsize = 10
tm.t = 1.0
tm.W = 1.0
tm.U = 1.0
tm.N_up = 2^2 ÷ 2
tm.N_down = 2^2 ÷ 2
tm.nx = 2
tm.ny = 2
tm.B = "Periodic"

if tm.B == "Periodic"
    lat = LatticeRectangular(tm.nx, tm.ny, Periodic())
elseif tm.B == "Open"
    lat = LatticeRectangular(tm.nx, tm.ny, Open())
else
    throw(ArgumentError("Boundary condition not recognized"))
end

model = AHmodel(lat, tm.t, tm.W, tm.U, tm.N_up, tm.N_down)
conf = vcat(FFS(model.U_up), FFS(model.U_down))

tm.omega = model.omega
tm.conf = Vector{Bool}(conf)

task(tm)

dir = @__DIR__
savepath = dir * "/../data/" * Dates.format(Dates.now(), "mm-ddTHH-MM-SS")
job = JobInfo(
    savepath,
    FastFermionSampling.MC;
    tasks=make_tasks(tm),
    checkpoint_time="30:00",
    run_time="24:00:00"
)

Carlo.start(job, ARGS)
