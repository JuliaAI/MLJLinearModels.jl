@testset "scratchspace" begin
    n = 50
    p = 10
    R.allocate(n, p)
    x1 = R.TEMP_N[]
    x2 = R.TEMP_N2[]
    x3 = R.TEMP_N3[]
    w  = R.TEMP_P[]
    for v in (x1, x2, x3)
        @test v isa Vector{Float64}
        @test length(v) == n
    end
    @test w isa Vector{Float64}
    @test length(w) == p
    R.deallocate()
    x1 = R.TEMP_N[]
    x2 = R.TEMP_N2[]
    x3 = R.TEMP_N3[]
    w  = R.TEMP_P[]
    for v in (x1, x2, x3, w)
        @test length(v) == 0
    end
    n, p, c = 50, 10, 3
    R.allocate(n, p, c)
    m1 = R.TEMP_NC[]
    m2 = R.TEMP_NC2[]
    for m in (m1, m2)
        @test m isa Matrix{Float64}
        @test size(m) == (n, c)
    end
    R.deallocate()
    @test size(R.TEMP_NC[]) == size(R.TEMP_NC2[]) == (0,0)
end
