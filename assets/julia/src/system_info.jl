#!/usr/bin/env julia
# system_info.jl — GPU検出・システム情報をJuliaで実行
# gpuHandlers.ts の nvidia-smi シェルアウトを置き換え

using JSON3

"""詳細なGPU情報を取得"""
function get_gpu_info()::Dict{String,Any}
    gpus = Dict{String,Any}[]

    try
        output = read(`nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,compute_cap --format=csv,noheader,nounits`, String)

        for line in filter(!isempty, strip.(split(output, '\n')))
            parts = strip.(split(line, ','))
            length(parts) >= 9 || continue

            gpu = Dict{String,Any}(
                "index" => parse(Int, parts[1]),
                "name" => parts[2],
                "driver_version" => parts[3],
                "memory_total_mb" => parse(Int, parts[4]),
                "memory_used_mb" => parse(Int, parts[5]),
                "memory_free_mb" => parse(Int, parts[6]),
                "temperature_c" => tryparse(Int, parts[7]),
                "utilization_pct" => tryparse(Int, parts[8]),
                "compute_capability" => parts[9],
            )
            push!(gpus, gpu)
        end
    catch e
        # nvidia-smi unavailable
    end

    # Blackwell 検出
    is_blackwell = false
    architecture = ""
    try
        output = read(`nvidia-smi -q`, String)
        is_blackwell = occursin(r"Product Architecture\s*:\s*Blackwell", output)
        m = match(r"Product Architecture\s*:\s*(\w+)", output)
        if m !== nothing
            architecture = m.captures[1]
        end
    catch end

    # CUDA バージョン
    cuda_version = ""
    try
        output = read(`nvidia-smi`, String)
        m = match(r"CUDA Version:\s*([\d.]+)", output)
        if m !== nothing
            cuda_version = m.captures[1]
        end
    catch end

    return Dict{String,Any}(
        "gpus" => gpus,
        "gpu_count" => length(gpus),
        "is_blackwell" => is_blackwell,
        "architecture" => architecture,
        "cuda_version" => cuda_version,
        "nvidia_available" => !isempty(gpus),
    )
end

"""CPU・メモリ情報を取得"""
function get_system_info()::Dict{String,Any}
    return Dict{String,Any}(
        "os" => string(Sys.KERNEL),
        "os_version" => if Sys.iswindows()
            try strip(read(`cmd /c ver`, String)) catch; "unknown" end
        else
            try strip(read(`uname -r`, String)) catch; "unknown" end
        end,
        "arch" => string(Sys.ARCH),
        "cpu_name" => Sys.cpu_info()[1].model,
        "cpu_threads" => Sys.CPU_THREADS,
        "cpu_cores" => length(Sys.cpu_info()),
        "total_memory_gb" => round(Sys.total_memory() / 1024^3, digits=2),
        "free_memory_gb" => round(Sys.free_memory() / 1024^3, digits=2),
        "julia_version" => string(VERSION),
        "hostname" => gethostname(),
    )
end

"""ディスク使用量を取得"""
function get_disk_info(check_path::String=".")::Dict{String,Any}
    try
        if Sys.iswindows()
            # Windows: wmic で取得
            drive = uppercase(first(abspath(check_path)))
            output = read(`wmic logicaldisk where "DeviceID='$(drive):'" get FreeSpace,Size /format:csv`, String)
            lines = filter(l -> !isempty(strip(l)) && !startswith(l, "Node"), split(output, '\n'))
            if !isempty(lines)
                parts = strip.(split(strip(lines[end]), ','))
                if length(parts) >= 3
                    free = parse(Float64, parts[2])
                    total = parse(Float64, parts[3])
                    return Dict{String,Any}(
                        "drive" => "$(drive):",
                        "total_gb" => round(total / 1024^3, digits=2),
                        "free_gb" => round(free / 1024^3, digits=2),
                        "used_pct" => round((1 - free/total) * 100, digits=1),
                    )
                end
            end
        end
    catch end
    return Dict{String,Any}("error" => "unable to detect")
end

function main()
    check_path = length(ARGS) >= 1 ? ARGS[1] : "."

    result = Dict{String,Any}(
        "gpu" => get_gpu_info(),
        "system" => get_system_info(),
        "disk" => get_disk_info(check_path),
    )

    println(JSON3.write(result))
end

main()
