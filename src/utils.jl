function get_test_parameters(filepath::String, row_index::Int, column_names::Vector{String})
    df = CSV.File(filepath) |> DataFrame
    
    if row_index < 1 || row_index > nrow(df)
        error("Row index is out of bounds.")
    end
    
    missing_columns = setdiff(column_names, names(df))
    if !isempty(missing_columns)
        error("The following columns are not found in the DataFrame: ", join(missing_columns, ", "))
    end
    
    row_data = df[row_index, column_names]
    
    return Dict(zip(column_names, row_data))
end

function set_test_results(file_path::String, row::Int, new_values::Dict)
    lines = readlines(file_path)
    
    num_duplications = length(ensure_vector(first(values(new_values))))

    tab = [split(lines[row+1], ',')]
    for i in 1:num_duplications-1
        insert!(tab, 2, copy(tab[1]))
    end
    
    cols = Dict(letter => index for (index, letter) in enumerate(split(lines[1],',')))

    for (col_name, values) in new_values
        values = ensure_vector(values)
        for i in 1:num_duplications
            tab[i][cols[col_name]] = string(values[i])
        end
    end
    
    lines[row+1] = join([join(tab[i],",") for i in 1:length(tab)], "\n")
    
    write(file_path, join(lines, "\n"))
end

function get_current_datetime_string()
    return Dates.format(now(), "yyyymmdd_HHMMSS")
end

function ensure_vector(x)
    if x isa AbstractVector
        return x
    end
    return [x]
end

function run_python_script(script_path)
    run(`python $(script_path)`)
end