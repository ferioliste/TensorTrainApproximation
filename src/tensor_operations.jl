function unfold_tensor(tensor, k)
    dims = ndims(tensor)
    
    k = k isa Integer ? (k,) : k
    
    for index in k
        if index < 1 || index > dims
            throw(ArgumentError("All elements of k must be within the range of tensor dimensions"))
        end
    end
    
    new_order = [k...]
    append!(new_order, setdiff(1:dims, k))

    s = size(tensor)
    new_size = (prod([s[i] for i in k]), :)

    return reshape(permutedims(tensor, new_order), new_size)
end

function fold_tensor(unfolded_matrix, row_dims, col_dims)
    if size(unfolded_matrix, 1) != prod(row_dims)
        throw(ArgumentError("The rows of the unfolded matrix must match prod(row_dims)"))
    end
    if size(unfolded_matrix, 2) != prod(col_dims)
        throw(ArgumentError("The columns of the unfolded matrix must match prod(col_dims)"))
    end

    return reshape(unfolded_matrix, row_dims..., col_dims...)
end

function mode_kj_mult(A,B,k,j)
    n_dims_A = ndims(A)
    n_dims_B = ndims(B)
    dims_A = size(A)
    dims_B = size(B)

    k = k isa Integer ? (k,) : k
    j = j isa Integer ? (j,) : j

    if length(k) != length(j)
        throw(ArgumentError("k and j must have the same length"))
    end
    for i in 1:length(k)
        if k[i] < 1 || k[i] > n_dims_A
            throw(ArgumentError("All elements of k must be within the range of A dimensions"))
        end
        if j[i] < 1 || j[i] > n_dims_B
            throw(ArgumentError("All elements of j must be within the range of B dimensions"))
        end
        if dims_A[k[i]] != dims_B[j[i]]
            throw(ArgumentError("The dimesions relative to the $i-th index of k and j do not agree"))
        end
    end

    return fold_tensor(unfold_tensor(A,k)' * unfold_tensor(B,j), [dims_A[i] for i in setdiff(1:n_dims_A, k)], [dims_B[i] for i in setdiff(1:n_dims_B, j)])
end

function tt2full(mps)
    if length(mps) == 1
        return reshape(mps[1], size(mps[1],2), size(mps[1],3))
    end
    res = copy(mps[1])
    for i in 2:length(mps)
        res = mode_kj_mult(res,mps[i],1+i,1)
    end
    return reshape(res, size(res)[2:end-1]...)
end

function get_tt_ranks(mps)
    tt_rankss=fill(1,length(mps)-1)
    for i in 1:length(mps)-1
        tt_rankss[i] = size(mps[i],3)
    end
    return tt_rankss
end

function get_tt_dims(mps)
    tt_dims=fill(1,length(mps))
    for i in 1:length(mps)
        tt_dims[i] = size(mps[i],2)
    end
    return tt_dims
end

function fix_ranks(tt_ranks, dims)
    n_dims = length(dims)
    if isa(tt_ranks, Integer)
        tt_ranks = fill(tt_ranks, n_dims-1)
    end
    @assert length(tt_ranks) == n_dims-1 "tt_ranks has the wrong length"
    
    rows_unfolding = 1
    cols_unfolding = prod(dims)
    for i in 1:n_dims-1
        rows_unfolding *= dims[i]
        cols_unfolding ÷= dims[i]
        tt_ranks[i] = min(tt_ranks[i], rows_unfolding, cols_unfolding)
    end

    return tt_ranks
end

function tt_sum(TT1, TT2)
    dims = get_tt_dims(TT1)
    n_dims = length(dims)
    tt_ranks1 = get_tt_ranks(TT1)
    tt_ranks2 = get_tt_ranks(TT2)
    tt_ranks = tt_ranks1 + tt_ranks2
    
    TT = Array{AbstractArray{Float64},1}(undef, n_dims)

    TT[1] = zeros(1, dims[1], tt_ranks[1])
    TT[1][:,:,1:tt_ranks1[1]] = TT1[1]
    TT[1][:,:,tt_ranks2[1]+1:tt_ranks[1]] = TT2[1]
    for μ in 2:n_dims-1
        TT[μ] = zeros(tt_ranks[μ-1], dims[μ], tt_ranks[μ])
        TT[μ][1:tt_ranks1[μ-1],:,1:tt_ranks1[μ]] = TT1[μ]
        TT[μ][tt_ranks2[μ-1]+1:tt_ranks[μ-1],:,tt_ranks2[μ]+1:tt_ranks[μ]] = TT2[μ]
    end
    TT[n_dims] = zeros(tt_ranks[n_dims-1], dims[1], 1)
    TT[n_dims][1:tt_ranks1[n_dims-1],:,:] = TT1[n_dims]
    TT[n_dims][tt_ranks2[n_dims-1]+1:tt_ranks[n_dims-1],:,:] = TT2[n_dims]

    return TT
end

function tt_mult(k, oldTT)
    n_dims = length(TT)
    k_root = abs(k)^(1/n_dims)

    TT = Array{AbstractArray{Float64},1}(undef, n_dims)
    for μ in 1:n_dims
        TT[μ] = (μ == 1 ? sign(k) : 1)*k_root*deepcopy(oldTT[μ])
    end

    return TT
end