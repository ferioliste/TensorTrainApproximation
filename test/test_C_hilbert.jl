function run_test(file_path::String, row_index::Int; save::Bool = true)
    parameters_names = ["tensor", "d", "n", "rank", "algorithm"];
    parameters = get_test_parameters(file_path, row_index, parameters_names);

    d = parameters["d"]
    n = parameters["n"]
    T = test_tensor(parameters["tensor"], n, d)

    to = TimerOutput()

    seed::Int64 = rand(UInt32)
    if algorithm == "tt_svd"
        @timeit to "time_taken" mps = tt_svd(T, fix_ranks(fill(rank, d-1), dims))
    elseif algorithm == "tt_rsvd"
        @timeit to "time_taken" mps = tt_rsvd(T, fix_ranks(fill(rank, d-1), dims), sketch_type="gaussian", seed=seed)
    elseif algorithm == "STTA_gauss"
        @timeit to "time_taken" mps = tt_STTA(T, fix_ranks(fill(rank, d-1), dims), fix_ranks(fill(rank, d-1), dims)*2, sketch_type="gaussian", seed=seed)
    elseif algorithm == "STTA_srht"
        @timeit to "time_taken" mps = tt_STTA(T, fix_ranks(fill(rank, d-1), dims), fix_ranks(fill(rank, d-1), dims)*2, sketch_type="srht", seed=seed)
    elseif algorithm == "STTA_srht_hash"
        @timeit to "time_taken" mps = tt_STTA(T, fix_ranks(fill(rank, d-1), dims), fix_ranks(fill(rank, d-1), dims)*2, sketch_type="srht_hash", s=10, seed=seed)
    else
        error("Algorithm '$algorithm' does not exist.")
    end

    T_ = tt2full(mps)
    err = norm(T-T_)/norm(T)
    time_taken = TimerOutputs.time(to["time_taken"])/1e9

    println("$row_index: tensor $(parameters["tensor"]) n $n d $d algorithm $(parameters["algorithm"]), rank $(parameters["rank"]), error $err time $time_taken")
    if save
        set_test_results(file_path, row_index, Dict(:"date_time" => get_current_datetime_string(), :"d" => d, :"n" => n, :"error" => err, :"time_taken" => time_taken, :"seed" => seed))
    end
end


mid = "hilbert"
ending = "n"

file_path = "../test_results/C_" * mid * "_" * ending * ".csv"

q=5
for i in 1:10
    for j in 1:q
        run_test(file_path, j, save = false)
    end
end

for i in (q+1):100000
    run_test(file_path, i, save = true)
end