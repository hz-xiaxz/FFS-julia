var documenterSearchIndex = {"docs":
[{"location":"91-developer/#dev_docs","page":"Developer documentation","title":"Developer documentation","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"note: Contributing guidelines\nIf you haven't, please read the Contributing guidelines first.","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"If you want to make contributions to this package that involves code, then this guide is for you.","category":"page"},{"location":"91-developer/#First-time-clone","page":"Developer documentation","title":"First time clone","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"tip: If you have writing rights\nIf you have writing rights, you don't have to fork. Instead, simply clone and skip ahead. Whenever upstream is mentioned, use origin instead.","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"If this is the first time you work with this repository, follow the instructions below to clone the repository.","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Fork this repo\nClone your repo (this will create a git remote called origin)\nAdd this repo as a remote:\ngit remote add upstream https://github.com/hz-xiaxz/FFS-julia","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"This will ensure that you have two remotes in your git: origin and upstream. You will create branches and push to origin, and you will fetch and update your local main branch from upstream.","category":"page"},{"location":"91-developer/#Linting-and-formatting","page":"Developer documentation","title":"Linting and formatting","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Install a plugin on your editor to use EditorConfig. This will ensure that your editor is configured with important formatting settings.","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"We use https://pre-commit.com to run the linters and formatters. In particular, the Julia code is formatted using JuliaFormatter.jl, so please install it globally first:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"julia> # Press ]\npkg> activate\npkg> add JuliaFormatter","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"To install pre-commit, we recommend using pipx as follows:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"# Install pipx following the link\npipx install pre-commit","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"With pre-commit installed, activate it as a pre-commit hook:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"pre-commit install","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"To run the linting and formatting manually, enter the command below:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"pre-commit run -a","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Now, you can only commit if all the pre-commit tests pass.","category":"page"},{"location":"91-developer/#Testing","page":"Developer documentation","title":"Testing","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"As with most Julia packages, you can just open Julia in the repository folder, activate the environment, and run test:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"julia> # press ]\npkg> activate .\npkg> test","category":"page"},{"location":"91-developer/#Working-on-a-new-issue","page":"Developer documentation","title":"Working on a new issue","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"We try to keep a linear history in this repo, so it is important to keep your branches up-to-date.","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Fetch from the remote and fast-forward your local main\ngit fetch upstream\ngit switch main\ngit merge --ff-only upstream/main\nBranch from main to address the issue (see below for naming)\ngit switch -c 42-add-answer-universe\nPush the new local branch to your personal remote repository\ngit push -u origin 42-add-answer-universe\nCreate a pull request to merge your remote branch into the org main.","category":"page"},{"location":"91-developer/#Branch-naming","page":"Developer documentation","title":"Branch naming","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"If there is an associated issue, add the issue number.\nIf there is no associated issue, and the changes are small, add a prefix such as \"typo\", \"hotfix\", \"small-refactor\", according to the type of update.\nIf the changes are not small and there is no associated issue, then create the issue first, so we can properly discuss the changes.\nUse dash separated imperative wording related to the issue (e.g., 14-add-tests, 15-fix-model, 16-remove-obsolete-files).","category":"page"},{"location":"91-developer/#Commit-message","page":"Developer documentation","title":"Commit message","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Use imperative or present tense, for instance: Add feature or Fix bug.\nHave informative titles.\nWhen necessary, add a body with details.\nIf there are breaking changes, add the information to the commit message.","category":"page"},{"location":"91-developer/#Before-creating-a-pull-request","page":"Developer documentation","title":"Before creating a pull request","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"tip: Atomic git commits\nTry to create \"atomic git commits\" (recommended reading: The Utopic Git History).","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Make sure the tests pass.\nMake sure the pre-commit tests pass.\nFetch any main updates from upstream and rebase your branch, if necessary:\ngit fetch upstream\ngit rebase upstream/main BRANCH_NAME\nThen you can open a pull request and work with the reviewer to address any issues.","category":"page"},{"location":"91-developer/#Building-and-viewing-the-documentation-locally","page":"Developer documentation","title":"Building and viewing the documentation locally","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Following the latest suggestions, we recommend using LiveServer to build the documentation. Here is how you do it:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Run julia --project=docs to open Julia in the environment of the docs.\nIf this is the first time building the docs\nPress ] to enter pkg mode\nRun pkg> dev . to use the development version of your package\nPress backspace to leave pkg mode\nRun julia> using LiveServer\nRun julia> servedocs()","category":"page"},{"location":"91-developer/#Making-a-new-release","page":"Developer documentation","title":"Making a new release","text":"","category":"section"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"To create a new release, you can follow these simple steps:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Create a branch release-x.y.z\nUpdate version in Project.toml\nUpdate the CHANGELOG.md:\nRename the section \"Unreleased\" to \"[x.y.z] - yyyy-mm-dd\" (i.e., version under brackets, dash, and date in ISO format)\nAdd a new section on top of it named \"Unreleased\"\nAdd a new link in the bottom for version \"x.y.z\"\nChange the \"[unreleased]\" link to use the latest version - end of line, vx.y.z ... HEAD.\nCreate a commit \"Release vx.y.z\", push, create a PR, wait for it to pass, merge the PR.\nGo back to main screen and click on the latest commit (link: https://github.com/hz-xiaxz/FFS-julia/commit/main)\nAt the bottom, write @JuliaRegistrator register","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"After that, you only need to wait and verify:","category":"page"},{"location":"91-developer/","page":"Developer documentation","title":"Developer documentation","text":"Wait for the bot to comment (should take < 1m) with a link to a RP to the registry\nFollow the link and wait for a comment on the auto-merge\nThe comment should said all is well and auto-merge should occur shortly\nAfter the merge happens, TagBot will trigger and create a new GitHub tag. Check on https://github.com/hz-xiaxz/FFS-julia/releases\nAfter the release is create, a \"docs\" GitHub action will start for the tag.\nAfter it passes, a deploy action will run.\nAfter that runs, the stable docs should be updated. Check them and look for the version number.","category":"page"},{"location":"95-reference/#reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"95-reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"95-reference/","page":"Reference","title":"Reference","text":"Pages = [\"95-reference.md\"]","category":"page"},{"location":"95-reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"95-reference/","page":"Reference","title":"Reference","text":"Pages = [\"95-reference.md\"]","category":"page"},{"location":"95-reference/","page":"Reference","title":"Reference","text":"Modules = [FastFermionSampling]","category":"page"},{"location":"95-reference/#FastFermionSampling.AHmodel","page":"Reference","title":"FastFermionSampling.AHmodel","text":"Anderson-Hubbard Model\n\nlattice : LatticeRectangular{B} The lattice structure\nt: Float64 Hopping parameter\nW : Float64 Disorder strength, on site energy is sampled from N(0, W/2)\nU : Float64 On-site interaction strength\nNup : Int Number of up spins\nNdown : Int Number of down spins\nomega : Vector{Float64} Random on-site energies\nU_up : Matrix{Float64} Unitary matrix for up spins\nU_down : Matrix{Float64} Unitary matrix for down spins\n\n\n\n\n\n","category":"type"},{"location":"95-reference/#FastFermionSampling.AHmodel-Union{Tuple{B}, Tuple{LatticeRectangular{B}, Float64, Float64, Float64, Int64, Int64}} where B","page":"Reference","title":"FastFermionSampling.AHmodel","text":"AHmodel(lattice::LatticeRectangular{B}, t::Float64, W::Float64, U::Float64, N_up::Int, N_down::Int)\n\nGenerate Anderson-Hubbard model and get the sampling ensemble\n\nRaise warning if the shell of degenerate eigenstates are not whole-filled\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.Gutzwiller","page":"Reference","title":"FastFermionSampling.Gutzwiller","text":"Gutzwiller Ansatz\n\ng : Float64 Gutzwiller factor\n\nStore the Gutzwiller factor parameter g.\n\nThe Gutzwiller factor is defined as G = exp(-g2 sum_i (n_i - n_mean)^2).\n\n\n\n\n\n","category":"type"},{"location":"95-reference/#FastFermionSampling.LatticeRectangular-Tuple{Int64, Int64, Periodic}","page":"Reference","title":"FastFermionSampling.LatticeRectangular","text":"LatticeRectangular(nx::Int, ny::Int, B::Periodic)\nLatticeRectangular(nx::Int, ny::Int, B::Open)\n\nGenerate a rectangular lattice with periodic or open boundary conditions\n\nnx : number of sites in x-direction\nny : number of sites in y-direction\nB : boundary condition, either Periodic or Open\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.MC-Tuple{AbstractDict}","page":"Reference","title":"FastFermionSampling.MC","text":"MC(params::AbstractDict)\n\n\n\nCreate a Monte Carlo object\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#Carlo.init!-Union{Tuple{B}, Tuple{MC{B}, Carlo.MCContext, AbstractDict}} where B","page":"Reference","title":"Carlo.init!","text":"Carlo.init!(mc::MC, ctx::MCContext, params::AbstractDict)\n\n\n\nInitialize the Monte Carlo object params\n\nnx : Int number of sites in x direction\nny : Int number of sites in y direction\nB : AbstractBoundary boundary condition, Periodic or Open\nt : Float64 hopping parameter\nW : Float64 disorder strength\nU : Float64 on-site interaction strength\nN_up : Int number of up spins\nN_down : Int number of down spins\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#Carlo.register_evaluables-Tuple{Type{MC}, Carlo.Evaluator, AbstractDict}","page":"Reference","title":"Carlo.register_evaluables","text":"fg the gradient of ⟨E_g⟩\n\nGet the gradient of the observable, f_g = -  E_g  g.\n\n``math \\begin{align} fk &= -2 ℜ[⟨OL(x)^* × (Og(x)- ⟨Og⟩) ⟩ ]\\\n &= -2 ℜ[⟨OL(x)^* × Og(x) ⟩ - ⟨OL(x)^* ⟩ × ⟨Og⟩  ]\\\n\\end{align} ``\n\nFisher Scalar\n\nGet the Fisher Matrix of the observable, S_kk  = ℜO_k O_k = ℜ( O_k O_k -O_k O_k ) where k and k are labels of the parameters of the model.\n\nWhen ansatz has only one parameter, the Fisher Matrix is a scalar, and the Fisher Information is the inverse of the Fisher Matrix.\n\nStructure Factor in low momentum\n\nN_q = n_qn_-q_disorder- n_qn_-q_disorder\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.FFS-Tuple{Random.AbstractRNG, AbstractMatrix}","page":"Reference","title":"FastFermionSampling.FFS","text":"FFS([rng=default_rng()], U::AbstractMatrix)\n\nEmploying Fast Fermion Sampling Algorithm to sample free Fermions\n\nU : the sampling ensemble, a matrix of size L x N, where L is the number of energy states and N is the number of Fermions\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.check_shell-Tuple{AbstractArray, Int64, Int64}","page":"Reference","title":"FastFermionSampling.check_shell","text":"check_shell(E::AbstractArray, Nup::Int, ns::Int)\n\nCheck if the whole degenerate shell of single particle eigenstates are filled.\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.fast_G_update-Union{Tuple{T}, Tuple{N}, Tuple{BitBasis.BitStr{N, T}, BitBasis.BitStr{N, T}, Float64, Float64}} where {N, T}","page":"Reference","title":"FastFermionSampling.fast_G_update","text":"fast_G_update(newwholeconf::BitStr{N,T}, oldwholeconf::BitStr{N,T}, g::Float64, n_mean::Float64) where {N,T}\n\nFast Gutzwiller Factor update technique from Becca and Sorella 2017\n\nShould input whole configuration\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.fast_update-Union{Tuple{T1}, Tuple{T}, Tuple{N}, Tuple{N1}, Tuple{D}, Tuple{AbstractMatrix, AbstractMatrix, Union{BitBasis.DitStr{D, N1, T1}, BitBasis.SubDitStr{D, N1, T1}}, BitBasis.BitStr{N, T}}} where {D, N1, N, T, T1}","page":"Reference","title":"FastFermionSampling.fast_update","text":"fast_update(U::AbstractMatrix, Uinvs::AbstractMatrix, newconf::BitStr{N,T}, oldconf::BitStr{N,T}) where {N,T}\n\nFast computing technique from Becca and Sorella 2017\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.getHmat-Union{Tuple{B}, Tuple{LatticeRectangular{B}, Float64, Vector{Float64}, Int64, Int64}} where B","page":"Reference","title":"FastFermionSampling.getHmat","text":"getHmat(lattice::LatticeRectangular{B}, t::Float64, omega::Vector{Float64}, N_up::Int, N_down::Int)'\n\nGet the non-interacting Anderson model Hamiltonian Matrix to construct Slater Determinants\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.getOL-Tuple{AHmodel, BitVector, BitVector, Float64}","page":"Reference","title":"FastFermionSampling.getOL","text":"getOL(orb::AHmodel, conf_up::BitVector, conf_down::BitVector, g::Float64)\n\nThe observable O_L = fracxHpsi_Gxpsi_G\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.getOg-Union{Tuple{B}, Tuple{AHmodel{B}, BitVector, BitVector}} where B","page":"Reference","title":"FastFermionSampling.getOg","text":"getOg(orbitals::AHmodel{B}, conf_up::BitVector, conf_down::BitVector)\n\nThe local operator to update the variational parameter g mathcalO_k(x)=fracpartial ln Psi_alpha(x)partial alpha_k\n\n\n\n\n\n","category":"method"},{"location":"95-reference/#FastFermionSampling.getxprime-Union{Tuple{T}, Tuple{N}, Tuple{B}, Tuple{AHmodel{B}, BitBasis.BitStr{N, T}}} where {B, N, T}","page":"Reference","title":"FastFermionSampling.getxprime","text":"getxprime(orb::AHmodel{B}, x::BitStr{N,T}) where {B,N,T}\n\nreturn x = Hx  where H = -t _ij c_i^ c_j + U _i n_i n_i + _i ω_i n_i\n\n\n\n\n\n","category":"method"},{"location":"90-contributing/#contributing","page":"Contributing guidelines","title":"Contributing guidelines","text":"","category":"section"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"First of all, thanks for the interest!","category":"page"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"We welcome all kinds of contribution, including, but not limited to code, documentation, examples, configuration, issue creating, etc.","category":"page"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"Be polite and respectful, and follow the code of conduct.","category":"page"},{"location":"90-contributing/#Bug-reports-and-discussions","page":"Contributing guidelines","title":"Bug reports and discussions","text":"","category":"section"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"90-contributing/#Working-on-an-issue","page":"Contributing guidelines","title":"Working on an issue","text":"","category":"section"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"If you found an issue that interests you, comment on that issue what your plans are. If the solution to the issue is clear, you can immediately create a pull request (see below). Otherwise, say what your proposed solution is and wait for a discussion around it.","category":"page"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"tip: Tip\nFeel free to ping us after a few days if there are no responses.","category":"page"},{"location":"90-contributing/","page":"Contributing guidelines","title":"Contributing guidelines","text":"If your solution involves code (or something that requires running the package locally), check the developer documentation. Otherwise, you can use the GitHub interface directly to create your pull request.","category":"page"},{"location":"","page":"FastFermionSampling","title":"FastFermionSampling","text":"CurrentModule = FastFermionSampling","category":"page"},{"location":"#FastFermionSampling","page":"FastFermionSampling","title":"FastFermionSampling","text":"","category":"section"},{"location":"","page":"FastFermionSampling","title":"FastFermionSampling","text":"Documentation for FastFermionSampling.","category":"page"}]
}
