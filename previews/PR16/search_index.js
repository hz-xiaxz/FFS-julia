var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FastFermionSampling","category":"page"},{"location":"#FastFermionSampling","page":"Home","title":"FastFermionSampling","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FastFermionSampling.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FastFermionSampling]","category":"page"},{"location":"#FastFermionSampling.AHmodel","page":"Home","title":"FastFermionSampling.AHmodel","text":"Anderson-Hubbard Model\n\nlattice: LatticeRectangular{B}     The lattice structure    t: Float64     Hopping parameter W: Float64     Disorder strength U: Float64     On-site interaction strength Nup: Int     Number of up spins Ndown: Int     Number of down spins omega: Vector{Float64}     Random on-site energies Uup: Matrix     Unitary matrix for up spins Udown: Matrix     Unitary matrix for down spins\n\n\n\n\n\n","category":"type"},{"location":"#FastFermionSampling.AHmodel-Union{Tuple{B}, Tuple{LatticeRectangular{B}, Float64, Float64, Float64, Int64, Int64}} where B","page":"Home","title":"FastFermionSampling.AHmodel","text":"generate Anderson-Hubbard model and get the sampling ensemble\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.Gutzwiller-Union{Tuple{B}, Tuple{AHmodel{B}, BitVector, BitVector, Float64}} where B","page":"Home","title":"FastFermionSampling.Gutzwiller","text":"add Gutzwiller Ansatz where G  = exp(-g/2 ∑i (ni - nmean)^2), ΨG = G Ψ_0\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.LatticeRectangular-Tuple{Int64, Int64, Periodic}","page":"Home","title":"FastFermionSampling.LatticeRectangular","text":"LatticeRectangular(nx::Int, ny::Int, B::Periodic)\nLatticeRectangular(nx::Int, ny::Int, B::Open)\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.FFS-Tuple{Random.AbstractRNG, AbstractMatrix}","page":"Home","title":"FastFermionSampling.FFS","text":"Employing Fast Fermion Sampling Algorithm to sample free Fermions\n\nFFS([rng=default_rng()], U::AbstractMatrix)\n\nU: the sampling ensemble, a matrix of size L x N, where L is the number of energy states and N is the number of Fermions \n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.fast_G_update-Union{Tuple{T}, Tuple{N}, Tuple{BitBasis.BitStr{N, T}, BitBasis.BitStr{N, T}, Float64, Float64}} where {N, T}","page":"Home","title":"FastFermionSampling.fast_G_update","text":"Fast Gutzwiller Factor update technique from Becca and Sorella 2017 Should input whole conf\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.fast_update-Union{Tuple{T}, Tuple{N}, Tuple{AbstractMatrix, AbstractMatrix, BitBasis.BitStr{N, T}, BitBasis.BitStr{N, T}}} where {N, T}","page":"Home","title":"FastFermionSampling.fast_update","text":"Fast computing technique from Becca and Sorella 2017\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.getHmat-Union{Tuple{B}, Tuple{LatticeRectangular{B}, Float64, Vector{Float64}, Int64, Int64}} where B","page":"Home","title":"FastFermionSampling.getHmat","text":"Get the non-interacting Anderson model Hamiltonian Matrix to construct HF states\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.getOL-Tuple{AHmodel, BitVector, BitVector, Float64}","page":"Home","title":"FastFermionSampling.getOL","text":"The observable OL = <x|H|ΨG>/<x|ΨG> \n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.getxprime-Union{Tuple{T}, Tuple{N}, Tuple{B}, Tuple{AHmodel{B}, BitBasis.BitStr{N, T}}} where {B, N, T}","page":"Home","title":"FastFermionSampling.getxprime","text":"return |x'> = H|x> where H = -t ∑<i,j> ci^† cj + U ∑i ni↓ ni↑ + ∑i ωi n_i\n\n\n\n\n\n","category":"method"}]
}
