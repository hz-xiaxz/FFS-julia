using FastFermionSampling
using Documenter

DocMeta.setdocmeta!(FastFermionSampling, :DocTestSetup, :(using FastFermionSampling); recursive=true)

makedocs(;
    modules=[FastFermionSampling],
    authors="XuanzheXia",
    sitename="FFS-julia",
    format=Documenter.HTML(;
        canonical="https://hz-xiaxz.github.io/FFS-julia",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hz-xiaxz/FFS-julia.git",
    devbranch="main",
    push_preview=true,
)
