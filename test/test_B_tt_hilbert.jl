function run_test(file_path::String, row_index::Int; save::Bool = true)
    parameters_names = ["tensor", "d", "n", "rank", "algorithm"];
    parameters = get_test_parameters(file_path, row_index, parameters_names);

    TT = test_tensor(parameters["tensor"], parameters["n"], parameters["d"])
    Tfull = tt2full(TT)

    d = length(TT)
    n = max(get_tt_dims(TT)...)
    to = TimerOutput()

    seed::Int64 = rand(UInt32)
    if parameters["algorithm"] == "tt_svd_rounding"
        @timeit to "time_taken" mps = tt_svd_rounding(TT, parameters["rank"], to=to)
    elseif parameters["algorithm"] == "tt_rsvd_rounding"
        @timeit to "time_taken" mps = tt_rsvd_rounding(TT, parameters["rank"], sketch_type="gaussian", seed=seed, to=to)
    elseif parameters["algorithm"] == "tt_rand_orth_rounding"
        @timeit to "time_taken" mps = tt_rand_orth_rounding(TT, parameters["rank"], sketch_type="gaussian", seed=seed, to=to)
    else
        error("Algorithm '$(parameters["algorithm"])' does not exist.")
    end
    
    Tfull_ = tt2full(mps)
    err = norm(Tfull-Tfull_)/norm(Tfull)

    t1 = TimerOutputs.time(to["time_taken"]["t1"])/1e9
    t2 = TimerOutputs.time(to["time_taken"]["t2"])/1e9
    time_taken = TimerOutputs.time(to["time_taken"])/1e9

    println("tensor $(parameters["tensor"]) n $n d $d algorithm $(parameters["algorithm"]), rank $(parameters["rank"]), error $err, time_taken $time_taken")
    if save
        set_test_results(file_path, row_index, Dict(:"date_time" => get_current_datetime_string(), :"d" => d, :"n" => n, :"error" => err, :"t1" => t1, :"t2" => t2, :"time_taken" => time_taken, :"seed" => seed))
    end
end



ending = "d"

file_path = "../test_results/B_tt_hilbert_" * ending * ".csv"

for i in 1:10
    run_test(file_path, 1, save = false)
    run_test(file_path, 2, save = false)
    run_test(file_path, 3, save = false)
end

for i in 4:100000
    run_test(file_path, i, save = true)
end

run_python_script("../plots_code/plot_B_tt_hilbert_" * ending * ".py")