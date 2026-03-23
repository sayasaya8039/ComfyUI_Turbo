#!/usr/bin/env julia
# image_utils.jl — 画像前処理/後処理を Julia で実行
# Python PIL/OpenCV の代替

using JSON3, Images, FileIO, SHA

"""画像のメタデータを取得"""
function get_image_info(filepath::String)::Dict{String,Any}
    if !isfile(filepath)
        return Dict{String,Any}("error" => "file_not_found", "path" => filepath)
    end

    try
        img = load(filepath)
        h, w = size(img)
        return Dict{String,Any}(
            "path" => filepath,
            "width" => w,
            "height" => h,
            "channels" => ndims(img) >= 3 ? size(img, 3) : (eltype(img) <: Gray ? 1 : 3),
            "eltype" => string(eltype(img)),
            "file_size_bytes" => filesize(filepath),
            "sha256" => bytes2hex(sha256(read(filepath))),
        )
    catch e
        return Dict{String,Any}("error" => string(e), "path" => filepath)
    end
end

"""画像をリサイズ"""
function resize_image(input::String, output::String, width::Int, height::Int)::Dict{String,Any}
    try
        img = load(input)
        resized = imresize(img, (height, width))
        save(output, resized)
        return Dict{String,Any}("status" => "success", "output" => output, "width" => width, "height" => height)
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""サムネイル生成（アスペクト比保持）"""
function create_thumbnail(input::String, output::String, max_size::Int=256)::Dict{String,Any}
    try
        img = load(input)
        h, w = size(img)
        scale = min(max_size / w, max_size / h, 1.0)
        new_w = round(Int, w * scale)
        new_h = round(Int, h * scale)
        thumb = imresize(img, (new_h, new_w))
        save(output, thumb)
        return Dict{String,Any}("status" => "success", "output" => output, "width" => new_w, "height" => new_h)
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""画像フォーマット変換"""
function convert_image(input::String, output::String)::Dict{String,Any}
    try
        img = load(input)
        save(output, img)
        return Dict{String,Any}("status" => "success", "input" => input, "output" => output)
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""バッチ画像情報取得"""
function batch_image_info(dir::String; extensions=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])::Dict{String,Any}
    if !isdir(dir)
        return Dict{String,Any}("error" => "directory_not_found", "path" => dir)
    end

    images = Dict{String,Any}[]
    total_size = 0

    for f in readdir(dir, join=true)
        ext = lowercase(splitext(f)[2])
        if ext in extensions && isfile(f)
            info = get_image_info(f)
            push!(images, info)
            total_size += get(info, "file_size_bytes", 0)
        end
    end

    return Dict{String,Any}(
        "directory" => dir,
        "count" => length(images),
        "total_size_mb" => round(total_size / 1024^2, digits=2),
        "images" => images,
    )
end

"""画像のハッシュ比較（重複検出）"""
function find_duplicates(dir::String)::Dict{String,Any}
    hashes = Dict{String,Vector{String}}()

    for f in readdir(dir, join=true)
        ext = lowercase(splitext(f)[2])
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"] && isfile(f)
            h = bytes2hex(sha256(read(f)))
            if !haskey(hashes, h)
                hashes[h] = String[]
            end
            push!(hashes[h], f)
        end
    end

    duplicates = filter(p -> length(p.second) > 1, hashes)
    return Dict{String,Any}(
        "directory" => dir,
        "duplicate_groups" => length(duplicates),
        "duplicates" => Dict(k => v for (k, v) in duplicates),
    )
end

function main()
    if length(ARGS) < 1
        println(JSON3.write(Dict("error" => "usage: image_utils.jl <command> [args...]")))
        exit(1)
    end

    command = ARGS[1]
    result = if command == "info" && length(ARGS) >= 2
        get_image_info(ARGS[2])
    elseif command == "resize" && length(ARGS) >= 4
        resize_image(ARGS[2], ARGS[3], parse(Int, ARGS[4]), parse(Int, length(ARGS) >= 5 ? ARGS[5] : ARGS[4]))
    elseif command == "thumbnail" && length(ARGS) >= 3
        max_sz = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 256
        create_thumbnail(ARGS[2], ARGS[3], max_sz)
    elseif command == "convert" && length(ARGS) >= 3
        convert_image(ARGS[2], ARGS[3])
    elseif command == "batch-info" && length(ARGS) >= 2
        batch_image_info(ARGS[2])
    elseif command == "find-duplicates" && length(ARGS) >= 2
        find_duplicates(ARGS[2])
    else
        Dict{String,Any}("error" => "unknown command: $command")
    end

    println(JSON3.write(result))
end

main()
