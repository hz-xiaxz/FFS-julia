#!/usr/bin/env -S julia --color=yes --startup-file=non

using Carlo
using Carlo.JobTools
using FastFermionSampling

tm = TaskMaker()
tm.thermalization = 0
tm.sweeps = 10
tm.binsize = 1

params = Dict(
    :t => 1.0,
    :W => 1.0,
    :U => 1.0,
    :N_up => 2,
    :N_down => 2,
    :nx => 2,
    :ny => 2,
    :boundary => Periodic()
)
task(tm, model=model, conf=conf)

job = JobInfo(
    @__FILE__,
    FastFermionSampling.MC;
    tasks=make_tasks(tm),
    checkpoint_time="30:00",
    run_time="24:00:00"
)

Carlo.start(job, ARGS)