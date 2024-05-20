function tt_svd_rounding(oldTT, tt_ranks; to::TimerOutput=TimerOutput())
    dims = get_tt_dims(oldTT)
    n_dims = length(dims)
    old_tt_ranks = [1; get_tt_ranks(oldTT); 1]
    
    TT = Array{AbstractArray{Float64},1}(undef, n_dims)
    tt_ranks = [1; fix_ranks(tt_ranks, dims); 1]

    @timeit to "t1" begin
        TT[n_dims] = deepcopy(oldTT[n_dims])
        for μ = n_dims:-1:2
            F = qr(reshape(TT[μ], (old_tt_ranks[μ], dims[μ]*old_tt_ranks[μ+1]))')
            TT[μ] = Matrix(F.Q)'
            TT[μ-1] = reshape(oldTT[μ-1], (old_tt_ranks[μ-1]*dims[μ-1], old_tt_ranks[μ])) * Matrix(F.R)'
        end
    end

    @timeit to "t2" begin
        for μ = 1:1:n_dims-1
            U, S, V = LinearAlgebra.svd(reshape(TT[μ], (tt_ranks[μ]*dims[μ], old_tt_ranks[μ+1])))
            TT[μ] = reshape(U[:, 1:tt_ranks[μ+1]], (tt_ranks[μ], dims[μ], tt_ranks[μ+1]))
            TT[μ+1] = (S[1:tt_ranks[μ+1]] .* V[:, 1:tt_ranks[μ+1]]') * reshape(TT[μ+1], (old_tt_ranks[μ+1], dims[μ+1]*old_tt_ranks[μ+2]))
        end
        TT[n_dims] = reshape(TT[n_dims], (tt_ranks[n_dims], dims[n_dims], 1))
    end

    return TT
end

function tt_rsvd_rounding(oldTT, tt_ranks; sketch_type = "gaussian", s = 1, seed = nothing, to::TimerOutput=TimerOutput())
    dims = get_tt_dims(oldTT)
    n_dims = length(dims)
    old_tt_ranks = [1; get_tt_ranks(oldTT); 1]
    
    TT = Array{AbstractArray{Float64},1}(undef, n_dims)
    tt_ranks = [1; fix_ranks(tt_ranks, dims); 1]
    
    @timeit to "t1" begin
        TT[n_dims] = deepcopy(oldTT[n_dims])
        for μ = n_dims:-1:2
            F = qr(reshape(TT[μ], (old_tt_ranks[μ], dims[μ]*old_tt_ranks[μ+1]))')
            TT[μ] = Matrix(F.Q)'
            TT[μ-1] = reshape(oldTT[μ-1], (old_tt_ranks[μ-1]*dims[μ-1], old_tt_ranks[μ])) * Matrix(F.R)'
        end
    end

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    seeds = rand(rng, UInt32, n_dims-1)

    @timeit to "t2" begin
        for μ = 1:1:n_dims-1
            temp = reshape(TT[μ], (tt_ranks[μ]*dims[μ], old_tt_ranks[μ+1]))
            
            F = qr(sketch(temp', tt_ranks[μ+1], sketch_type, s=s, seed=seeds[μ])')
            TT[μ] = reshape(Matrix(F.Q), (tt_ranks[μ], dims[μ], tt_ranks[μ+1]))
            TT[μ+1] = (Matrix(F.Q)' * temp) * reshape(TT[μ+1], (old_tt_ranks[μ+1], dims[μ+1]*old_tt_ranks[μ+2]))
        end
        TT[n_dims] = reshape(TT[n_dims], (tt_ranks[n_dims], dims[n_dims], 1))
    end
    
    return TT
end

function tt_rand_orth_rounding(oldTT, tt_ranks;  sketch_type = "gaussian", s = 1, seed = nothing, to::TimerOutput=TimerOutput())
    dims = get_tt_dims(oldTT)
    n_dims = length(dims)
    old_tt_ranks = [1; get_tt_ranks(oldTT); 1]
    
    TT = Array{AbstractArray{Float64},1}(undef, n_dims)
    tt_ranks = [1; fix_ranks(tt_ranks, dims); 1]

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    seeds = rand(rng, UInt32, n_dims-1)

    @timeit to "t1" begin
        W = Array{Array{Float64, 2},1}(undef, n_dims - 1)
        W[n_dims - 1] = sketch(reshape(oldTT[n_dims], (old_tt_ranks[n_dims], dims[n_dims]*old_tt_ranks[n_dims+1]))', tt_ranks[n_dims], sketch_type, s=s, seed=seeds[n_dims-1])'
        for μ = n_dims-1:-1:2
            W[μ - 1] = sketch(reshape(reshape(oldTT[μ], (old_tt_ranks[μ]*dims[μ], old_tt_ranks[μ+1]))*W[μ], (old_tt_ranks[μ], dims[μ]*tt_ranks[μ+1]))', tt_ranks[μ], sketch_type, s=s, seed=seeds[μ-1])'
        end
    end

    @timeit to "t2" begin
        TT[1] = deepcopy(oldTT[1])
        for μ = 1:n_dims-1
            Z = reshape(TT[μ], (tt_ranks[μ]*dims[μ], old_tt_ranks[μ+1]))
            F = qr(Z*W[μ])
            TT[μ] = reshape(Matrix(F.Q), (tt_ranks[μ], dims[μ], tt_ranks[μ+1]))
            TT[μ+1] = Matrix(F.Q)' * Z * reshape(oldTT[μ+1], (old_tt_ranks[μ+1], dims[μ+1]*old_tt_ranks[μ+2]))
        end
        TT[n_dims] = reshape(TT[n_dims], (tt_ranks[n_dims], dims[n_dims], 1))
    end

    return TT
end

function tt_sketch_STTA_rounding(C, left_ranks, right_ranks; sketch_type = "gaussian", s = 1, seed = nothing)
    dims = get_tt_dims(C)
    n_dims = length(dims)
    L = Array{AbstractArray{Float64},1}(undef, n_dims-1)
    R = Array{AbstractArray{Float64},1}(undef, n_dims-1)
    Psi = Array{AbstractArray{Float64},1}(undef, n_dims)
    Omega = Array{AbstractArray{Float64},1}(undef, n_dims-1)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    X_seeds = rand(rng, UInt32, n_dims-1)
    Y_seeds = rand(rng, UInt32, n_dims-1)

    # Compute L
    L[1] = sketch(reshape(C[1], size(C[1],2), size(C[1],3)), left_ranks[1], sketch_type, s=s, seed=X_seeds[1])
    for μ = 2:1:n_dims-1
        temp = reshape(sketch(unfold_tensor(C[μ],2), left_ranks[μ-1]*left_ranks[μ], sketch_type, s=s, seed=X_seeds[μ]), left_ranks[μ-1], left_ranks[μ], size(C[μ],1), size(C[μ],3))
        @tensor L[μ][:] := L[μ-1][1,2] * temp[1,-1,2,-2]
    end

    # Compute R
    R[n_dims - 1] = sketch(reshape(C[n_dims], size(C[n_dims],1), size(C[n_dims],2))', right_ranks[n_dims-1], sketch_type, s=s, seed=Y_seeds[n_dims-1])'
    for μ = n_dims-2:-1:1
        temp = reshape(sketch(unfold_tensor(C[μ+1],2), right_ranks[μ]*right_ranks[μ+1], sketch_type, s=s, seed=Y_seeds[μ])', size(C[μ+1],1), size(C[μ+1],3), right_ranks[μ], right_ranks[μ+1])
        @tensor R[μ][:] := R[μ+1][1,2] * temp[-1, 1, -2, 2]
    end

    # Compute Omega
    for μ = 1:n_dims-1
        Omega[μ] = L[μ] * R[μ]
    end

    # Compute Psi
    Psi[1] = reshape(reshape(C[1], size(C[1],2), size(C[1],3)) * R[1], 1, size(C[1],2), size(R[1],2))
    for μ = 2:n_dims-1
        @tensor Psi[μ][:] := L[μ-1][-1,1] * C[μ][1,-2,2] * R[μ][2,-3]
    end
    Psi[n_dims] = reshape(L[n_dims-1] * reshape(C[n_dims], size(C[n_dims],1), size(C[n_dims],2)), size(L[n_dims-1], 1), size(C[n_dims],2), 1)

    return Psi, Omega
end

function tt_STTA_rounding(TT, left_ranks, right_ranks; sketch_type = "gaussian", s = 1, seed = nothing, to::TimerOutput=TimerOutput())
    if !(all(left_ranks .< right_ranks) || all(right_ranks .< left_ranks))
        error("The ranks provided are not valid")
    end

    @timeit to "t1" Psi, Omega = tt_sketch_STTA_rounding(TT, left_ranks, right_ranks, sketch_type=sketch_type, s=s, seed=seed)
    @timeit to "t2" begin
        if left_ranks[1] < right_ranks[1]
            return assemble_sketch_kres_left(Psi, Omega)
        else
            return assemble_sketch_kres_right(Psi, Omega)
        end
    end
end