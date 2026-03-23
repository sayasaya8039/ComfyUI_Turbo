#!/usr/bin/env julia
# verify_deps.jl — Python の pythonImportVerifier.ts を Julia で置き換え
# 使用: julia verify_deps.jl [--modules mod1,mod2,...] [--check-gpu] [--check-system]

using JSON3, Dates

struct VerificationResult
    success::Bool
    failed_imports::Vector{String}
    gpu_info::Dict{String,Any}
    system_info::Dict{String,Any}
    timestamp::String
end

"""GPU 情報を検出（nvidia-smi / CUDA.jl フォールバック）"""
function detect_gpu()::Dict{String,Any}
    info = Dict{String,Any}(
        "available" => false,
        "name" => "",
        "driver" => "",
        "memory_mb" => 0,
        "architecture" => "",
        "cuda_version" => "",
        "is_blackwell" => false,
    )

    # nvidia-smi で検出
    try
        output = read(`nvidia-smi --query-gpu=name,driver_version,memory.total,gpu_uuid --format=csv,noheader,nounits`, String)
        lines = filter(!isempty, strip.(split(output, '\n')))
        if !isempty(lines)
            parts = strip.(split(lines[1], ','))
            info["available"] = true
            info["name"] = length(parts) >= 1 ? parts[1] : ""
            info["driver"] = length(parts) >= 2 ? parts[2] : ""
            info["memory_mb"] = length(parts) >= 3 ? parse(Int, strip(parts[3])) : 0
        end
    catch
        # nvidia-smi not available
    end

    # CUDA バージョン検出
    try
        output = read(`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`, String)
        info["cuda_version"] = strip(output)
    catch end

    # Blackwell アーキテクチャ検出
    try
        output = read(`nvidia-smi -q`, String)
        info["is_blackwell"] = occursin(r"Product Architecture\s*:\s*Blackwell", output)
        m = match(r"Product Architecture\s*:\s*(\w+)", output)
        if m !== nothing
            info["architecture"] = m.captures[1]
        end
    catch end

    return info
end

"""システム情報を収集"""
function collect_system_info()::Dict{String,Any}
    return Dict{String,Any}(
        "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "cpu_threads" => Sys.CPU_THREADS,
        "total_memory_gb" => round(Sys.total_memory() / 1024^3, digits=1),
        "free_memory_gb" => round(Sys.free_memory() / 1024^3, digits=1),
        "julia_version" => string(VERSION),
        "word_size" => Sys.WORD_SIZE,
    )
end

"""Julia パッケージのインポートを検証"""
function verify_julia_imports(module_names::Vector{String})::Vector{String}
    failed = String[]
    for mod in module_names
        try
            # Julia パッケージとして読み込み試行
            sym = Symbol(mod)
            Base.require(Main, sym)
        catch
            push!(failed, mod)
        end
    end
    return failed
end

"""Python モジュールの存在確認（pip list 経由）"""
function verify_python_modules(modules::Vector{String}, venv_path::String="")::Vector{String}
    failed = String[]

    python_cmd = if !isempty(venv_path)
        if Sys.iswindows()
            joinpath(venv_path, "Scripts", "python.exe")
        else
            joinpath(venv_path, "bin", "python")
        end
    else
        "python"
    end

    for mod in modules
        try
            script = "import $(mod); print('OK')"
            output = read(`$python_cmd -c $script`, String)
            if !occursin("OK", output)
                push!(failed, mod)
            end
        catch
            push!(failed, mod)
        end
    end

    return failed
end

function main()
    modules = String[]
    check_gpu = false
    check_system = false
    venv_path = ""

    # 引数パース
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--modules" && i < length(ARGS)
            i += 1
            modules = filter(!isempty, split(ARGS[i], ','))
        elseif ARGS[i] == "--check-gpu"
            check_gpu = true
        elseif ARGS[i] == "--check-system"
            check_system = true
        elseif ARGS[i] == "--venv" && i < length(ARGS)
            i += 1
            venv_path = ARGS[i]
        end
        i += 1
    end

    # 検証実行
    failed = if !isempty(modules)
        verify_python_modules(modules, venv_path)
    else
        String[]
    end

    gpu_info = check_gpu ? detect_gpu() : Dict{String,Any}()
    sys_info = check_system ? collect_system_info() : Dict{String,Any}()

    result = Dict{String,Any}(
        "success" => isempty(failed),
        "failed_imports" => failed,
        "gpu_info" => gpu_info,
        "system_info" => sys_info,
        "timestamp" => string(Dates.now()),
    )

    # JSON 出力
    println(JSON3.write(result))
    exit(isempty(failed) ? 0 : 1)
end

main()
