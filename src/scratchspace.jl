# Objects here are defined as scratchspace for recurring computations when
# fitting models. The naming indicates the dimensions, e.g. _N, _N2 etc are
# vectors of size N (several as sometimes we need non-overlapping vectors)
# _NC are N*C matrices used in multiclass settings
#
# For now all of these are Float64 scratch spaces.

const SCRATCH_N   = Ref(zeros(0))
const SCRATCH_N2  = Ref(zeros(0))
const SCRATCH_N3  = Ref(zeros(0))
const SCRATCH_P   = Ref(zeros(0))
const SCRATCH_NC  = Ref(zeros(0,0))
const SCRATCH_NC2 = Ref(zeros(0,0))
const SCRATCH_NC3 = Ref(zeros(0,0))
const SCRATCH_NC4 = Ref(zeros(0,0))
const SCRATCH_PC  = Ref(zeros(0,0))

allocate(n, p, c=0) = begin
    SCRATCH_N[]  = zeros(n)
    SCRATCH_N2[] = zeros(n)
    SCRATCH_N3[] = zeros(n)
    SCRATCH_P[]  = zeros(p)
    if !iszero(c)
        SCRATCH_NC[]  = zeros(n, c)
        SCRATCH_NC2[] = zeros(n, c)
        SCRATCH_NC3[] = zeros(n, c)
        SCRATCH_NC4[] = zeros(n, c)
        SCRATCH_PC[]  = zeros(p, c)
    end
end

deallocate() = begin
    SCRATCH_N[]   = zeros(0)
    SCRATCH_N2[]  = zeros(0)
    SCRATCH_N3[]  = zeros(0)
    SCRATCH_P[]   = zeros(0)
    SCRATCH_NC[]  = zeros(0,0)
    SCRATCH_NC2[] = zeros(0,0)
    SCRATCH_NC3[] = zeros(0,0)
    SCRATCH_NC4[] = zeros(0,0)
    SCRATCH_PC[]  = zeros(0,0)
end
