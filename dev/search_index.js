var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = FastFermionSampling","category":"page"},{"location":"#FastFermionSampling","page":"Home","title":"FastFermionSampling","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FastFermionSampling.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [FastFermionSampling]","category":"page"},{"location":"#FastFermionSampling.AHmodel","page":"Home","title":"FastFermionSampling.AHmodel","text":"Anderson-Hubbard Model\n\nlattice : LatticeRectangular{B} The lattice structure   \nt: Float64 Hopping parameter\nW : Float64 Disorder strength, on site energy is sampled from N(0, W/2)\nU : Float64 On-site interaction strength\nNup : Int Number of up spins\nNdown : Int Number of down spins\nomega : Vector{Float64} Random on-site energies\nU_up : Matrix{Float64} Unitary matrix for up spins\nU_down : Matrix{Float64} Unitary matrix for down spins\n\n\n\n\n\n","category":"type"},{"location":"#FastFermionSampling.AHmodel-Union{Tuple{B}, Tuple{LatticeRectangular{B}, Float64, Float64, Float64, Int64, Int64}} where B","page":"Home","title":"FastFermionSampling.AHmodel","text":"AHmodel(lattice::LatticeRectangular{B}, t::Float64, W::Float64, U::Float64, N_up::Int, N_down::Int)\n\nGenerate Anderson-Hubbard model and get the sampling ensemble\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.Gutzwiller","page":"Home","title":"FastFermionSampling.Gutzwiller","text":"Gutzwiller Ansatz\n\ng : Float64 Gutzwiller factor\nOg : Float64 Gutzwiller observable\nOL : Float64 Observable\n\n\n\n\n\n","category":"type"},{"location":"#FastFermionSampling.Gutzwiller-Union{Tuple{B}, Tuple{AHmodel{B}, BitVector, BitVector, Float64}} where B","page":"Home","title":"FastFermionSampling.Gutzwiller","text":"Gutzwiller(orbitals::AHmodel{B}, conf_up::BitVector, conf_down::BitVector, g::Float64)\n\nadd Gutzwiller Ansatz where G  = exp(-g2 sum_i (n_i - n_mean)^2), psi_G = G psi_0\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.LatticeRectangular-Tuple{Int64, Int64, Periodic}","page":"Home","title":"FastFermionSampling.LatticeRectangular","text":"LatticeRectangular(nx::Int, ny::Int, B::Periodic)\nLatticeRectangular(nx::Int, ny::Int, B::Open)\n\nGenerate a rectangular lattice with periodic or open boundary conditions\n\nnx : number of sites in x-direction\nny : number of sites in y-direction\nB : boundary condition, either Periodic or Open\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.FFS-Tuple{Random.AbstractRNG, AbstractMatrix}","page":"Home","title":"FastFermionSampling.FFS","text":"FFS([rng=default_rng()], U::AbstractMatrix)\n\nEmploying Fast Fermion Sampling Algorithm to sample free Fermions\n\nU : the sampling ensemble, a matrix of size L x N, where L is the number of energy states and N is the number of Fermions \n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.fast_G_update-Union{Tuple{T}, Tuple{N}, Tuple{BitBasis.BitStr{N, T}, BitBasis.BitStr{N, T}, Float64, Float64}} where {N, T}","page":"Home","title":"FastFermionSampling.fast_G_update","text":"fast_G_update(newwholeconf::BitStr{N,T}, oldwholeconf::BitStr{N,T}, g::Float64, n_mean::Float64) where {N,T}\n\nFast Gutzwiller Factor update technique from Becca and Sorella 2017\n\nShould input whole configuration \n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.fast_update-Union{Tuple{T}, Tuple{N}, Tuple{AbstractMatrix, AbstractMatrix, BitBasis.BitStr{N, T}, BitBasis.BitStr{N, T}}} where {N, T}","page":"Home","title":"FastFermionSampling.fast_update","text":"fast_update(U::AbstractMatrix, Uinvs::AbstractMatrix, newconf::BitStr{N,T}, oldconf::BitStr{N,T}) where {N,T}\n\nFast computing technique from Becca and Sorella 2017\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.getHmat-Union{Tuple{B}, Tuple{LatticeRectangular{B}, Float64, Vector{Float64}, Int64, Int64}} where B","page":"Home","title":"FastFermionSampling.getHmat","text":"getHmat(lattice::LatticeRectangular{B}, t::Float64, omega::Vector{Float64}, N_up::Int, N_down::Int)'\n\nGet the non-interacting Anderson model Hamiltonian Matrix to construct Slater Determinants \n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.getOL-Tuple{AHmodel, BitVector, BitVector, Float64}","page":"Home","title":"FastFermionSampling.getOL","text":"getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector, g::Float64)\n\nThe observable O_L = fracxHpsi_Gxpsi_G\n\n\n\n\n\n","category":"method"},{"location":"#FastFermionSampling.getxprime-Union{Tuple{T}, Tuple{N}, Tuple{B}, Tuple{AHmodel{B}, BitBasis.BitStr{N, T}}} where {B, N, T}","page":"Home","title":"FastFermionSampling.getxprime","text":"getxprime(orb::AHmodel{B}, x::BitStr{N,T}) where {B,N,T}\n\nreturn x = Hx  where H = -t _ij c_i^ c_j + U _i n_i n_i + _i ω_i n_i\n\n\n\n\n\n","category":"method"}]
}
