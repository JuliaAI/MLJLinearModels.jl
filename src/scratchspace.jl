# Objects here are defined as scratchspace for recurring computations when
# fitting models. The naming indicates the dimensions, e.g. _N, _N2 etc are
# vectors of size N (several as sometimes we need non-overlapping vectors)
# _NC are N*C matrices used in multiclass settings
#
# For now all of these are Float64 scratch spaces.

const TEMP_N   = Ref(zeros(0))
const TEMP_N2  = Ref(zeros(0))
const TEMP_N3  = Ref(zeros(0))
const TEMP_P   = Ref(zeros(0))
const TEMP_NC  = Ref(zeros(0,0))
const TEMP_NC2 = Ref(zeros(0,0))

allocate(n, p, c=0) = begin
    TEMP_N[]  = zeros(n)
    TEMP_N2[] = zeros(n)
    TEMP_N3[] = zeros(n)
    TEMP_P[]  = zeros(p)
    if !iszero(c)
        TEMP_NC[]  = zeros(n, c)
        TEMP_NC2[] = zeros(n, c)
    end
end

deallocate() = begin
    TEMP_N[]   = zeros(0)
    TEMP_N2[]  = zeros(0)
    TEMP_N3[]  = zeros(0)
    TEMP_P[]   = zeros(0)
    TEMP_NC[]  = zeros(0,0)
    TEMP_NC2[] = zeros(0,0)
end
