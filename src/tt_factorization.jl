function tt_svd(tensor, tt_ranks; to::TimerOutput=TimerOutput())
    dims = size(tensor)
    n_dims = length(dims)
    tt_ranks = [1; fix_ranks(tt_ranks, dims); 1]

    TTTensor = Array{AbstractArray{Float64},1}(undef, n_dims)

    C = deepcopy(tensor)
    for i in 1:n_dims-1
        C = reshape(C, (tt_ranks[i]*dims[i], :))
        @timeit to "t1" U, S, V = LinearAlgebra.svd(C)
        TTTensor[i] = reshape(U[:, 1:tt_ranks[i+1]], (tt_ranks[i], dims[i], tt_ranks[i+1]))
        @timeit to "t2" C = S[1:tt_ranks[i+1]] .* V[:, 1:tt_ranks[i+1]]'
    end
    TTTensor[n_dims] = reshape(C, (size(C)..., 1))

    return TTTensor
end

function tt_rsvd(tensor, tt_ranks; sketch_type = "gaussian", s = 1, seed = nothing, to::TimerOutput=TimerOutput())
    dims = size(tensor)
    n_dims = length(dims)
    tt_ranks = [1; fix_ranks(tt_ranks, dims); 1]

    TTTensor = Array{AbstractArray{Float64},1}(undef, n_dims)

    C = deepcopy(tensor)
    for i in 1:n_dims-1
        C = reshape(C, (tt_ranks[i]*dims[i], :))
        @timeit to "t1" begin
            temp = Matrix(sketch(C', tt_ranks[i+1], sketch_type, s=s, seed=seed)')
            @timeit to "t2" F = qr(temp)
        end
        TTTensor[i] = reshape(Matrix(F.Q), (tt_ranks[i], dims[i], tt_ranks[i+1]))
        @timeit to "t3" C = Matrix(F.Q)' * C
    end
    TTTensor[n_dims] = reshape(C, (size(C)..., 1))

    return TTTensor
end

function tt_sketch_STTA(tensor, left_ranks, right_ranks; sketch_type = "gaussian", s = 1, seed = nothing)
    dims = size(tensor)
    n_dims = length(dims)

    Psi = Array{AbstractArray{Float64},1}(undef, n_dims)
    Omega = Array{AbstractArray{Float64},1}(undef, n_dims - 1)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    X_seeds = rand(rng, UInt32, n_dims-1)
    Y_seeds = rand(rng, UInt32, n_dims-1)
    
    C = nothing
    for μ in 1:n_dims-1
        C = reshape(tensor, prod(dims[1:μ]), prod(dims[μ+1:n_dims]))
        if μ == 1
            Psi[1] = reshape(sketch(C', right_ranks[1], sketch_type, s=s, seed=X_seeds[1])', 1, dims[1], right_ranks[1])
        elseif μ < n_dims
            Psi[μ] = reshape(sketch(reshape(sketch(C', right_ranks[μ], sketch_type, s=s, seed=X_seeds[μ])', prod(dims[1:μ-1]), dims[μ]*right_ranks[μ]), left_ranks[μ-1], sketch_type, s=s, seed=Y_seeds[μ-1]), left_ranks[μ-1], dims[μ], right_ranks[μ])
        end
        Omega[μ] = sketch(sketch(C, left_ranks[μ], sketch_type, s=s, seed=Y_seeds[μ])', right_ranks[μ], sketch_type, s=s, seed=X_seeds[μ])'
    end
    Psi[n_dims] = reshape(sketch(C, left_ranks[n_dims-1], sketch_type, s=s, seed=Y_seeds[n_dims-1]), left_ranks[n_dims-1], dims[n_dims], 1)

    return Psi, Omega
end

function assemble_sketch_kres_right(Psi, Omega; tol = 1e-12)
    n_dims = length(Psi)

    TTTensor = Array{AbstractArray{Float64},1}(undef, n_dims)

    TTTensor[1] = copy(Psi[1])
    for μ = 2:1:n_dims
        U, S, V = svd(Omega[μ-1])
        
        local_tol = tol * maximum(S)
        significant_idxs = S .> local_tol
        S_inv = zeros(size(S))
        S_inv[significant_idxs] = 1.0 ./ S[significant_idxs]

        TTTensor[μ] = reshape(V * (S_inv .* (U'*unfold_tensor(Psi[μ], 1))), size(Omega[μ-1], 2), size(Psi[μ], 2), size(Psi[μ], 3))
    end

    return TTTensor
end

function assemble_sketch_kres_left(Psi, Omega; tol = 1e-12)
    n_dims = length(Psi)

    TTTensor = Array{AbstractArray{Float64},1}(undef, n_dims)

    TTTensor[n_dims] = copy(Psi[n_dims])
    for μ = n_dims-1:-1:1
        U, S, V = svd(Omega[μ])
        
        local_tol = tol * maximum(S)
        significant_idxs = S .> local_tol
        S_inv = zeros(size(S))
        S_inv[significant_idxs] = 1.0 ./ S[significant_idxs]

        TTTensor[μ] = reshape(((unfold_tensor(Psi[μ], 1:2) * V) .* S_inv') * U', size(Psi[μ], 1), size(Psi[μ], 2), size(Omega[μ], 1))
    end

    return TTTensor
end

function tt_STTA(tensor, left_ranks, right_ranks; sketch_type = "gaussian", s = 1, seed = nothing)
    if !(all(left_ranks .< right_ranks) || all(right_ranks .< left_ranks))
        error("The ranks provided are not valid")
    end

    Psi, Omega = tt_sketch_STTA(tensor, left_ranks, right_ranks, sketch_type=sketch_type, s=s, seed=seed)
    if left_ranks[1] < right_ranks[1]
        return assemble_sketch_kres_left(Psi, Omega)
    else
        return assemble_sketch_kres_right(Psi, Omega)
    end
end