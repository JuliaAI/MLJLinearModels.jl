using MLJLinearModels
using BenchmarkTools, Random, LinearAlgebra
DO_COMPARISONS = false; include("../testutils.jl")

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

R.allocate(n, p)

Xθ = similar(y)
Xtθ = similar(θ)
r = R.SCRATCH_N[]

# Sept 13, 2019 :: 7.83 ms (0 allocations: 0 bytes)
@btime R.apply_X!($Xθ, $X, $θ);

# Sept 13, 2019 :: 7.94 ms (0 allocations: 0 bytes) [pretty much only apply_X!]
@btime R.get_residuals!($r, $X, $θ, $y);

# Sept 13, 2019 :: 7.9 ms (0 allocations: 0 bytes)
@btime R.apply_Xt!($Xtθ, $X, $y);

# --------------
# With intercept

R.allocate(n, p+1)

Xθ = similar(y)
Xtθ = similar(θ1)
r = R.SCRATCH_N[]

# Sept 13, 2019 :: 7.949 ms (1 allocation: 48 bytes) -- alloc for the view
@btime R.apply_X!($Xθ, $X, $θ1);

# Sept 13, 2019 :: 7.955 ms (1 allocation: 48 bytes) -- mostly apply_X!
@btime R.get_residuals!($r, $X, $θ1, $y);

# Sept 13, 2019 :: 7.883 ms (1 allocation: 48 bytes)
@btime R.apply_Xt!($Xtθ, $X, $y);
