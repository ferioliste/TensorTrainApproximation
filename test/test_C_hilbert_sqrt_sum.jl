function run_test_batch(file_path::String, row_index::Int, num_test::Int; save::Bool = true)
    parameters_names = ["rank", "algorithm"]
    parameters = get_test_parameters(file_path, row_index, parameters_names)

    errors = zeros(num_test)
    times = zeros(num_test)
    seeds = zeros(Int64, num_test)

    for i in 1:num_test
        errors[i], times[i], seeds[i] = run_test(parameters)
    end

    if save
        set_test_results(file_path, row_index, Dict(:"date_time" => fill(get_current_datetime_string(),num_test), :"error" => errors, :"time_taken" => times, :"seed" => seeds))
    end
end

function run_test(parameters)
    dims = size(T)
    d = length(dims)

    rank = parameters["rank"]
    algorithm = parameters["algorithm"]

    seed::Int64 = rand(UInt32)

    if algorithm == "tt_svd"
        mps, time_taken = @timed tt_svd(T, fix_ranks(fill(rank, d-1), dims))
    elseif algorithm == "tt_rsvd"
        mps, time_taken = @timed tt_rsvd(T, fix_ranks(fill(rank, d-1), dims), sketch_type="gaussian", seed=seed)
    elseif algorithm == "STTA_gauss"
        mps, time_taken = @timed tt_STTA(T, fix_ranks(fill(rank, d-1), dims), fix_ranks(fill(rank, d-1), dims)*2, sketch_type="gaussian", seed=seed)
    elseif algorithm == "STTA_srht"
        mps, time_taken = @timed tt_STTA(T, fix_ranks(fill(rank, d-1), dims), fix_ranks(fill(rank, d-1), dims)*2, sketch_type="srht", seed=seed)
    elseif algorithm == "STTA_srht_hash"
        mps, time_taken = @timed tt_STTA(T, fix_ranks(fill(rank, d-1), dims), fix_ranks(fill(rank, d-1), dims)*2, sketch_type="srht_hash", s=10, seed=seed)
    else
        error("Algorithm '$algorithm' does not exist.")
    end

    T_ = tt2full(mps)
    err = norm(T-T_)/norm_T

    println("Algorithm $algorithm, rank $rank, error $err, time_taken $time_taken")
    return err, time_taken, seed
end

mid = "sqrt_sum"

n = 5
d = 7
global T = test_tensor(mid, n, d)
global norm_T = norm(T)

file_path = "../test_results/C_" * mid * "_r.csv"
num_test = 100
start_row = 1


for i::Int64 in start_row:num_test:1e16
    println("Starting row $i")
    run_test_batch(file_path, i, num_test, save=true)
    GC.gc()
end