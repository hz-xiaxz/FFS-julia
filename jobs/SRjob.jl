#!/usr/bin/env -S julia --color=yes --startup-file=none
# 

using Carlo
using Carlo.JobTools
using FastFermionSampling
using Dates
using DataFrames
using Carlo.ResultTools
using Measurements
using Logging

nx = 2
ny = 2
SRsteps = 10
g = 0.39118426172566841
B = "Open"
t = 1.0
W = 1.0
U = 1.0
ns = nx * ny
N_up = ns ÷ 2
N_down = ns ÷ 2
if B == "Periodic"
    lat = LatticeRectangular(nx, ny, Periodic())
elseif B == "Open"
    lat = LatticeRectangular(nx, ny, Open())
else
    throw(ArgumentError("Boundary condition not recognized"))
end
model = FastFermionSampling.fixedAHmodel(lat, t, W, U, N_up, N_down)
eta = 0.1
process_time = Dates.format(Dates.now(), "mm-ddTHH-MM-SS")
for _ in 1:SRsteps
    tm = TaskMaker()
    tm.thermalization = 0
    tm.sweeps = 100000
    tm.binsize = 100
    tm.t = 1.0
    tm.W = 1.0
    tm.U = 1.0
    tm.nx = nx
    tm.ny = ny
    tm.ns = tm.nx * tm.ny
    tm.N_up = tm.ns ÷ 2
    tm.N_down = tm.ns ÷ 2
    tm.B = B
    tm.omega = model.omega
    tm.g = g
    task(tm)

    dir = @__DIR__
    savepath = dir * "/../data/" * process_time *
               "/$(tm.nx)x$(tm.ny)g=$(tm.g)"
    job = JobInfo(
        savepath,
        FastFermionSampling.MC;
        tasks = make_tasks(tm),
        checkpoint_time = "30:00",
        run_time = "24:00:00"
    )
    Carlo.cli_delete(job, Dict())
    with_logger(Carlo.default_logger()) do
        start(Carlo.SingleScheduler, job)
    end

    df = DataFrame(ResultTools.dataframe(savepath * ".results.json"))
    # temporarily omit the discipline of `fg`

    fg = Measurements.value(df[!, :fg][1])
    if abs(Measurements.uncertainty(df[!, :fg][1])) > abs(fg)
        @warn "fg has big error"
        break
    end
    fisherScalar = Measurements.value(df[!, :fisherScalar][1])
    global g
    g += eta * fg / fisherScalar
    @show g
end
# TODO:Merge the results 