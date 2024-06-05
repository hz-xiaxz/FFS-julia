using FastFermionSampling
using Documenter

DocMeta.setdocmeta!(FastFermionSampling, :DocTestSetup, :(using FastFermionSampling); recursive=true)

makedocs(;
    modules=[FastFermionSampling],
    authors="XuanzheXia",
    sitename="FastFermionSampling.jl",
    format=Documenter.HTML(;
        canonical="https://hz-xiaxz.github.io/FastFermionSampling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/XuanzheXia/FastFermionSampling.jl",
    devbranch="main",
)
