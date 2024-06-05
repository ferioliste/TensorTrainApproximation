module TensorTrainApproximation

include("sketching.jl")
include("tensor_operations.jl")
include("tt_factorization.jl")
include("tt_rounding.jl")

export tt_svd
export tt_rsvd
export tt_STTA

export tt_svd_rounding
export tt_rsvd_rounding
export tt_rand_orth_rounding
export tt_STTA_rounding

end
