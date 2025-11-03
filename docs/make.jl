using FastFermionSampling
using Documenter

DocMeta.setdocmeta!(
    FastFermionSampling,
    :DocTestSetup,
    :(using FastFermionSampling);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [FastFermionSampling],
    authors = "XuanzheXia",
    repo = "https://github.com/hz-xiaxz/FFS-julia/blob/{commit}{path}#{line}",
    sitename = "FFS-julia",
    format = Documenter.HTML(; canonical = "https://hz-xiaxz.github.io/FFS-julia"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/hz-xiaxz/FFS-julia")
