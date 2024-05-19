function run_test(file_path::String, row_index::Int; save::Bool = true, T = nothing, norm_T = nothing)
    parameters_names = ["tensor", "d", "n", "rank", "algorithm"];
    parameters = get_test_parameters(file_path, row_index, parameters_names);

    if isnothing(T)
        T = test_tensor(parameters["tensor"], parameters["n"], parameters["d"])
    end
    if isnothing(norm_T)
        norm_T = norm(T)
    end
    d = ndims(T)
    n = max(size(T)...)
    to = TimerOutput()

    seed::Int64 = rand(UInt32)

    if parameters["algorithm"] == "tt_svd"
        @timeit to "time_taken" mps = tt_svd(T, parameters["rank"], to=to)
    elseif parameters["algorithm"] == "tt_rsvd"
        @timeit to "time_taken" mps = tt_rsvd(T, parameters["rank"], sketch_type="gaussian", seed=seed, to=to)
    else
        error("Algorithm '$(parameters["algorithm"])' does not exist.")
    end
    
    T_ = tt2full(mps)
    err = norm(T-T_)/norm_T
    if parameters["algorithm"] == "tt_svd"
        t1 = TimerOutputs.time(to["time_taken"]["t1"])/1e9
        t2 = TimerOutputs.time(to["time_taken"]["t2"])/1e9
        t3 = 0
    else
        t2 = TimerOutputs.time(to["time_taken"]["t1"]["t2"])/1e9
        t1 = TimerOutputs.time(to["time_taken"]["t1"])/1e9 - t2
        t3 = TimerOutputs.time(to["time_taken"]["t3"])/1e9
    end
    time_taken = TimerOutputs.time(to["time_taken"])/1e9

    println("tensor $(parameters["tensor"]) n $n d $d algorithm $(parameters["algorithm"]), rank $(parameters["rank"]), error $err, time_taken $time_taken")
    if save
        set_test_results(file_path, row_index, Dict(:"date_time" => get_current_datetime_string(), :"d" => d, :"n" => n, :"error" => err, :"t1" => t1, :"t2" => t2, :"t3" => t3, :"time_taken" => time_taken, :"seed" => seed))
    end
end

mid = "sqrt_sum"
ending = "d"

file_path = "../test_results/A_" * mid * "_" * ending * ".csv"

for i in 1:10
    run_test(file_path, 1, save = false, T = nothing, norm_T = nothing)
    run_test(file_path, 2, save = false, T = nothing, norm_T = nothing)
end

for i in 3:382
    run_test(file_path, i, save = true, T = nothing, norm_T = nothing)
end

run_python_script("../plots_code/plot_A_" * mid * "_" * ending * ".py")