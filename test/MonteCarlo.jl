using Test, FastFermionSampling

@testset "MC" begin
    params = Dict(
        :nx => 4,
        :ny => 4,
        :t => 1.0,
        :W => 1.0,
        :U => 1.0,
        :N_up => 8,
        :N_down => 8,
        :B => "Periodic",
        :g => 1.0
    )
    mc = FastFermionSampling.MC(params)
    @test mc.model.lattice.nx == params[:nx]
    @test mc.model.lattice.ny == params[:ny]
end