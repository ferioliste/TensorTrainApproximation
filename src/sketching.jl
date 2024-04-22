function sketch(A, k, sketch_type = "gaussian"; s = 1, seed = nothing)
    if sketch_type == "gaussian"
        return sketch_gaussian(A, k, seed)
    elseif sketch_type == "srht"
        return sketch_srht(A, k, seed)
    elseif sketch_type == "srht_hash"
        return sketch_srht_hash(A, k, s, seed)
    else
        error("Unsupported sketch type: $sketch_type")
    end
end

function sketch_gaussian(A, k, seed = nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    return randn(rng, k, size(A, 1)) * A
end

function sketch_srht(A, k, seed = nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    
    rows, cols = size(A)
    rows_pow2 = 2^ceil(Int, log2(rows))

    rows_idxs = sample(rng, 1:rows_pow2, k, replace=true)

    return (rows_pow2/sqrt(k)) * fwht_natural(vcat(sample(rng, [-1, 1], rows) .* A, zeros(Float64, rows_pow2 - rows, cols)),1)[rows_idxs,:]
end


function sketch_srht_hash(A, k, s, seed = nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    
    rows, cols = size(A)
    rows_pow2 = 2^ceil(Int, log2(rows))

    A_transformed = (rows_pow2/sqrt(k)) * fwht_natural(vcat(sample(rng, [-1, 1], rows) .* A, zeros(Float64, rows_pow2 - rows, cols)),1)
    A_sketched = zeros(k, cols)

    for i in 1:k
        if rows_pow2 < s
            hashed_idxs = sample(rng, 1:rows_pow2, s, replace=true)
        else
            hashed_idxs = sample(rng, 1:rows_pow2, s, replace=false)
        end
        
        A_sketched[i,:] = sum(A_transformed[hashed_idxs,:], dims=1)/s
    end
    return A_sketched
end