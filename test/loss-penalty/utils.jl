@testset "elastic net" begin
    a = 0.5 * L2Penalty() + 0.3 * L1Penalty()
    b = 0.3 * L1Penalty() + 0.5 * L2Penalty()
    c = 0.3 * L1Penalty() + 0.2 * L2Penalty() + 0.3 * L1Penalty()
    @test R.is_elnet(a)
    @test R.is_elnet(b)
    @test R.getscale_l1(a) == 0.3
    @test R.getscale_l2(c) == 0.2
    @test R.getscale_l1(c) == 0.6

    @test R.getscale_l2(2*L2Penalty()) == 2
    @test R.getscale_l1(3.4*L1Penalty()) == 3.4
end
