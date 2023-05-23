using MLJLinearModels
using BenchmarkTools, Random, LinearAlgebra
include("../testutils.jl")

n, p = 50_000, 500
((X, y, θ), (X1, y1, θ1)) = generate_continuous(n, p;  seed=512, sparse=0.5)

# ======================== #
# ELEMENTARIES
#
# > apply_X!        ✅ Sept 13, 2019
# > apply_Xt!       ✅ Sept 13, 2019
# > get_residuals!  ✅ Sept 13, 2019
# ======================== #

# ------------
# No intercept

Xθ = similar(y)
Xtθ = similar(θ)
r = zeros(n)

# Sept 13, 2019 :: 7.83 ms (0 allocations: 0 bytes)
# May 12, 2020 :: 8.091 ms (8 allocations: 128 bytes)
#              :: 7.612 ms (8 allocations: 128 bytes)
@btime R.apply_X!($Xθ, $X, $θ);

# Sept 13, 2019 :: 7.94 ms (0 allocations: 0 bytes) [pretty much only apply_X!]
# May 12, 2020 :: 8.305 ms (12 allocations: 256 bytes)
#              :: 7.826 ms (12 allocations: 256 bytes)
@btime R.get_residuals!($r, $X, $θ, $y);

# Sept 13, 2019 :: 7.9 ms (0 allocations: 0 bytes)
# May 12, 2020 :: 8.047 ms (8 allocations: 128 bytes)
#              :: 7.616 ms (8 allocations: 128 bytes)
@btime R.apply_Xt!($Xtθ, $X, $y);

# --------------
# With intercept

Xθ = similar(y)
Xtθ = similar(θ1)

# Sept 13, 2019 :: 7.949 ms (1 allocation: 48 bytes) -- alloc for the view
# May 12, 2020 :: 8.190 ms (12 allocations: 272 bytes)
#              :: 7.810 ms (12 allocations: 272 bytes)
@btime R.apply_X!($Xθ, $X, $θ1);

# Sept 13, 2019 :: 7.955 ms (1 allocation: 48 bytes) -- mostly apply_X!
# May 12, 2020 :: 8.375 ms (16 allocations: 400 bytes)
#              :: 7.945 ms (16 allocations: 400 bytes)
@btime R.get_residuals!($r, $X, $θ1, $y);

# Sept 13, 2019 :: 7.883 ms (1 allocation: 48 bytes)
# May 12, 2020 :: 8.320 ms (9 allocations: 176 bytes)
#              :: 7.884 ms (9 allocations: 176 bytes)
@btime R.apply_Xt!($Xtθ, $X, $y);
