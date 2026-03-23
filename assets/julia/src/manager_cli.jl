#!/usr/bin/env julia
# manager_cli.jl — cm-cli.py (ComfyUI-Manager CLI) を Julia で置き換え
# スナップショット保存/復元、カスタムノード管理

using JSON3, Dates, SHA

const SNAPSHOT_VERSION = 1

"""カスタムノードディレクトリをスキャン"""
function scan_custom_nodes(custom_nodes_dir::String)::Vector{Dict{String,Any}}
    nodes = Dict{String,Any}[]

    if !isdir(custom_nodes_dir)
        return nodes
    end

    for entry in readdir(custom_nodes_dir, join=true)
        if isdir(entry)
            name = basename(entry)
            # .git ディレクトリの存在でGitリポジトリ判定
            git_dir = joinpath(entry, ".git")
            is_git = isdir(git_dir) || isfile(git_dir)

            node_info = Dict{String,Any}(
                "name" => name,
                "path" => entry,
                "is_git_repo" => is_git,
                "has_requirements" => isfile(joinpath(entry, "requirements.txt")),
                "has_init" => isfile(joinpath(entry, "__init__.py")),
                "disabled" => isfile(joinpath(entry, ".disabled")),
            )

            # Gitリポ情報
            if is_git
                try
                    url = strip(read(`git -C $entry remote get-url origin`, String))
                    node_info["git_url"] = url
                catch end

                try
                    hash = strip(read(`git -C $entry rev-parse HEAD`, String))
                    node_info["git_hash"] = hash
                catch end

                try
                    branch = strip(read(`git -C $entry rev-parse --abbrev-ref HEAD`, String))
                    node_info["git_branch"] = branch
                catch end
            end

            push!(nodes, node_info)
        end
    end

    return nodes
end

"""スナップショットを保存"""
function save_snapshot(comfyui_path::String, output_file::String; full_snapshot::Bool=false)::Dict{String,Any}
    custom_nodes_dir = joinpath(comfyui_path, "custom_nodes")
    nodes = scan_custom_nodes(custom_nodes_dir)

    snapshot = Dict{String,Any}(
        "version" => SNAPSHOT_VERSION,
        "timestamp" => string(Dates.now()),
        "comfyui_path" => comfyui_path,
        "custom_nodes" => nodes,
        "node_count" => length(nodes),
        "git_nodes" => count(n -> get(n, "is_git_repo", false), nodes),
    )

    # フルスナップショットの場合、各ノードのファイルリストも含む
    if full_snapshot
        for node in nodes
            node_path = node["path"]
            try
                files = String[]
                for (root, dirs, fs) in walkdir(node_path)
                    # .git は除外
                    filter!(d -> d != ".git", dirs)
                    for f in fs
                        rel = relpath(joinpath(root, f), node_path)
                        push!(files, rel)
                    end
                end
                node["files"] = files
                node["file_count"] = length(files)
            catch end
        end
    end

    try
        open(output_file, "w") do io
            write(io, JSON3.write(snapshot))
        end
        return Dict{String,Any}("status" => "success", "output" => output_file, "node_count" => length(nodes))
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""スナップショットから復元"""
function restore_snapshot(snapshot_file::String, target_dir::String)::Dict{String,Any}
    if !isfile(snapshot_file)
        return Dict{String,Any}("status" => "error", "error" => "snapshot file not found: $snapshot_file")
    end

    try
        data = JSON3.read(read(snapshot_file, String), Dict{String,Any})
        nodes = get(data, "custom_nodes", Dict{String,Any}[])

        restored = String[]
        failed = Dict{String,String}[]

        for node in nodes
            url = get(node, "git_url", nothing)
            name = get(node, "name", "unknown")
            hash = get(node, "git_hash", nothing)

            if url === nothing
                push!(failed, Dict("name" => name, "reason" => "no git URL"))
                continue
            end

            target_path = joinpath(target_dir, name)

            # 既に存在する場合はスキップ
            if isdir(target_path)
                push!(restored, "$name (already exists)")
                continue
            end

            try
                # git clone
                run(`git clone $url $target_path`)

                # 特定コミットにチェックアウト
                if hash !== nothing
                    run(`git -C $target_path checkout $hash`)
                end

                push!(restored, name)
            catch e
                push!(failed, Dict("name" => name, "reason" => string(e)))
            end
        end

        return Dict{String,Any}(
            "status" => "success",
            "restored" => restored,
            "restored_count" => length(restored),
            "failed" => failed,
            "failed_count" => length(failed),
        )
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""カスタムノードの有効/無効を切り替え"""
function toggle_node(custom_nodes_dir::String, node_name::String, enable::Bool)::Dict{String,Any}
    node_path = joinpath(custom_nodes_dir, node_name)
    if !isdir(node_path)
        return Dict{String,Any}("status" => "error", "error" => "node not found: $node_name")
    end

    disabled_marker = joinpath(node_path, ".disabled")

    try
        if enable && isfile(disabled_marker)
            rm(disabled_marker)
        elseif !enable && !isfile(disabled_marker)
            touch(disabled_marker)
        end

        return Dict{String,Any}("status" => "success", "node" => node_name, "enabled" => enable)
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""カスタムノードを更新（git pull）"""
function update_node(custom_nodes_dir::String, node_name::String)::Dict{String,Any}
    node_path = joinpath(custom_nodes_dir, node_name)
    if !isdir(node_path)
        return Dict{String,Any}("status" => "error", "error" => "node not found: $node_name")
    end

    try
        before = strip(read(`git -C $node_path rev-parse HEAD`, String))
        run(`git -C $node_path pull --ff-only`)
        after = strip(read(`git -C $node_path rev-parse HEAD`, String))

        return Dict{String,Any}(
            "status" => "success",
            "node" => node_name,
            "updated" => before != after,
            "before" => before,
            "after" => after,
        )
    catch e
        return Dict{String,Any}("status" => "error", "error" => string(e))
    end
end

"""全カスタムノードの更新チェック"""
function check_updates(custom_nodes_dir::String)::Dict{String,Any}
    nodes = scan_custom_nodes(custom_nodes_dir)
    updates_available = Dict{String,Any}[]

    for node in nodes
        if !get(node, "is_git_repo", false)
            continue
        end

        name = node["name"]
        node_path = node["path"]

        try
            run(`git -C $node_path fetch --quiet`)
            local_head = strip(read(`git -C $node_path rev-parse HEAD`, String))
            remote_head = strip(read(`git -C $node_path rev-parse @{u}`, String))

            if local_head != remote_head
                push!(updates_available, Dict{String,Any}(
                    "name" => name,
                    "local" => local_head[1:8],
                    "remote" => remote_head[1:8],
                ))
            end
        catch
            # リモート追跡ブランチがない場合など
        end
    end

    return Dict{String,Any}(
        "total_nodes" => length(nodes),
        "updates_available" => length(updates_available),
        "nodes_with_updates" => updates_available,
    )
end

function main()
    if length(ARGS) < 1
        println(JSON3.write(Dict("error" => "usage: manager_cli.jl <command> [args...]")))
        exit(1)
    end

    command = ARGS[1]
    result = if command == "scan" && length(ARGS) >= 2
        Dict{String,Any}("nodes" => scan_custom_nodes(ARGS[2]))
    elseif command == "save-snapshot" && length(ARGS) >= 3
        full = "--full" in ARGS
        save_snapshot(ARGS[2], ARGS[3]; full_snapshot=full)
    elseif command == "restore-snapshot" && length(ARGS) >= 3
        restore_snapshot(ARGS[2], ARGS[3])
    elseif command == "enable" && length(ARGS) >= 3
        toggle_node(ARGS[2], ARGS[3], true)
    elseif command == "disable" && length(ARGS) >= 3
        toggle_node(ARGS[2], ARGS[3], false)
    elseif command == "update" && length(ARGS) >= 3
        update_node(ARGS[2], ARGS[3])
    elseif command == "check-updates" && length(ARGS) >= 2
        check_updates(ARGS[2])
    else
        Dict{String,Any}("error" => "unknown command: $command. Available: scan, save-snapshot, restore-snapshot, enable, disable, update, check-updates")
    end

    println(JSON3.write(result))
end

main()
