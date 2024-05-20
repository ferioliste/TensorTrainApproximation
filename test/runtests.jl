using LinearAlgebra
using StatsBase
using Random
using Hadamard
using FFTW
using CSV
using DataFrames
using Dates
using TensorOperations
using Plots
using TimerOutputs

include("../src/utils.jl")
include("../src/sketching.jl")
include("../src/tensor_operations.jl")
include("../src/test_tensors.jl")
include("../src/tt_factorization.jl")
include("../src/tt_rounding.jl")


#include("../test/test_temp.jl")
#include("../test/test_hilbert.jl")
#include("../test/test_sqrt_sum.jl")

#include("../test/test_pseudonorm.jl")

#include("../test/test_A_hilbert_sqrt_sum.jl")
#include("../test/test_B_tt_hilbert.jl")

include("../test/test_B_tt_random_sum.jl")