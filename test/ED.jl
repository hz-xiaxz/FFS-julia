using FastFermionSampling, Test

@testset "ED" begin
    energy = FastFermionSampling.doED(3, 4, 1.0, 8.0, zeros(12), 'O')
    @test length(energy) == 1
    @test energy[1]≈-4.913259209075605 atol=1e-10
end
