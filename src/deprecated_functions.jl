function tt_sketch_STTA_gauss(tensor, left_ranks, right_ranks)
    dims = size(tensor)
    n_dims = length(dims)

    Psi = Array{AbstractArray{Float64},1}(undef, n_dims)
    Omega = Array{AbstractArray{Float64},1}(undef, n_dims - 1)

    X = Array{AbstractArray{Float64},1}(undef, n_dims-1)
    Y = Array{AbstractArray{Float64},1}(undef, n_dims-1)
    for μ in 1:n_dims-1
        X[μ] = randn(prod(dims[μ+1:n_dims]), right_ranks[μ])
        Y[μ] = randn(left_ranks[μ], prod(dims[1:μ]))
    end

    C = nothing
    for μ in 1:n_dims-1
        C = reshape(tensor, prod(dims[1:μ]), prod(dims[μ+1:n_dims]))
        if μ == 1
            Psi[1] = reshape(C * X[1], 1, dims[1], right_ranks[1])
        elseif μ < n_dims
            Psi[μ] = reshape(Y[μ-1] * reshape(C * X[μ], prod(dims[1:μ-1]), dims[μ] * right_ranks[μ]), left_ranks[μ-1], dims[μ], right_ranks[μ])
        end
        Omega[μ] = Y[μ] * C * X[μ]
    end
    Psi[n_dims] = reshape(Y[n_dims-1] * C, left_ranks[n_dims-1], dims[n_dims], 1)

    return Psi, Omega
end

function fft_cosine!(A)
    n_col = size(A, 2)
    for i in 1:n_col
        A[:,i] = reinterpret(Float64, fft(A[:,i]))[1:2:end-1]
    end
    return A
end

function generate_gaussian_tt(ranks, dims)
    gtens::Array{AbstractArray{Float64},1} = []
    ranks = [1, ranks..., 1]
    for i in 1:length(dims)
        push!(gtens, randn(ranks[i],dims[i],ranks[i+1])/sqrt(ranks[i]*dims[i]*ranks[i+1]))
    end
    return gtens
end

function set_test_results(file_path::String, row::Int, new_values::Dict)
    lines = readlines(file_path)
    
    cols = Dict(letter => index for (index, letter) in enumerate(split(lines[1],',')))
    
    line_elements = split(lines[row+1], ',')
    for (col_name, new_value) in new_values
        line_elements[cols[col_name]] = string(new_value)
    end
    line_elements[1] = get_current_datetime_string()
    lines[row+1] = join(line_elements, ',')
    
    write(file_path, join(lines, "\n"))
end