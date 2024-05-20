function test_tensor(tensor_type, n, d; a=-0.2, b=2, r=20, pert=1e-10, seed=nothing)
    if tensor_type == "hilbert"
        return hilbert_tensor(n, d)
    elseif tensor_type == "sqrt_sum"
        return square_root_sum_tensor(n, d, a, b)
    elseif tensor_type == "random"
        return random_tensor(n, d, seed)
    elseif tensor_type == "tt_hilbert"
        return tt_hilbert_tensor(n, d, r)
    elseif tensor_type == "tt_random"
        return tt_random_tensor(n, d, r, seed)
    elseif tensor_type == "tt_random_sum"
        return tt_random_sum_tensor(n, d, r, pert, seed)
    else
        error("Unsupported tensor type: $tensor_type")
    end
end

function hilbert_tensor(n, d)
    H = zeros(fill(n, d)...)

    for indices in CartesianIndices(H)
        H[indices] = 1 / (sum(indices.I) + d - 1)
    end
    
    return H
end

function square_root_sum_tensor(n, d, a, b)
    S = zeros(fill(n, d)...)

    for indices in CartesianIndices(S)
        idxs = indices.I
        S[indices] = sqrt(abs(sum([a*(n-idxs[j])/(n-1) + b*(idxs[j]-1)/(n-1) for j in 1:d])))
    end
    return S
end

function random_tensor(n, d, seed)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    return randn(rng, fill(n, d)...)
end

function tt_hilbert_tensor(n, d, r)
    return tt_svd(hilbert_tensor(n, d), fix_ranks(r, fill(n, d)))
end

function tt_random_tensor(n, d, r, seed)
    tt_ranks = [1; fix_ranks(r, fill(n, d)); 1]
    TT = Array{AbstractArray{Float64},1}(undef, n_dims)
    
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    for μ in 1:d
        TT[μ] = randn(rng, tt_ranks[μ], n, tt_ranks[μ+1])
    end

    return TT
end

function tt_random_sum_tensor(n, d, r, pert, seed)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    
    TT1 = tt_random_tensor(n, d, r, rand(rng, UInt32))
    TT2 = tt_mult(pert, tt_random_tensor(n, d, r, rand(rng, UInt32)))

    return tt_sum(TT1, TT2)
end