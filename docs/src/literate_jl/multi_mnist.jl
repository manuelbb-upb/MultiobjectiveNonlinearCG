# This file is meant to be parsed by Literate.jl #src
if !(joinpath(@__DIR__, "..", "..") in LOAD_PATH) #src
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) #src
end #src
using Pkg #src
Pkg.activate(@__DIR__) #src

#=
Besides `MultiobjectiveNonlinearCG`, the following packages are required to run this example:
```julia
using Pkg
Pkg.add(;url="https://github.com/manuelbb-upb/MultiTaskLearning.jl.git")
Pkg.add(;url="https://github.com/manuelbb-upb/MultiMLDatasets.jl.git")
Pkg.add(
    ["MLUtils", "OneHotArrays", "ComponentArrays", 
    "Lux", "Zygote", "ChainRulesCore"]
)
```
=#

using MultiMLDatasets
mmnist = SenerKoltunMNIST(; force_recreate=true)

let
    img, labels = mmnist[1]
    image(img; 
        aspect=DataAspect(), 
        axis=(title=string(Int.(labels)), yreversed=true,), 
        interpolate=false
    )
end

import MLUtils: DataLoader
import OneHotArrays: onehotbatch, onecold

function load_multi_mnist(mode=:train; batchsize=-1, shuffle=true, data_percentage=0.1)
    if mode != :train mode = :test end

    mmnist = MultiMNIST(Float32, mode)

    last_index = min(length(mmnist), floor(Int, length(mmnist)*data_percentage))
    imgs, _labels = mmnist[1:last_index];# _labels[:,i] is a vector with two labels for sample i
    labels = onehotbatch(_labels, 0:9)   # labels[:,:,i] is a matrix, labels[:, 1, i] is the one hot vector of the first label, labels[:,2,i] is for the second label
    sx, sy, num_dat = size(imgs)

    # reshape for convolutional layer, it wants an additional dimension:
    X = reshape(imgs, sx, sy, 1, num_dat)
    return DataLoader(
        (features=X, labels=(llabels=labels[:,1,:], rlabels=labels[:,2,:]));
        shuffle, batchsize=batchsize > 0 ? min(batchsize,num_dat) : num_dat
    )
end

dat = load_multi_mnist(;batchsize=64);
X, Y = first(dat); # extract first batch

import Lux
import Random
import MultiTaskLearning: LRModel, multidir
import ComponentArrays: ComponentArray, getaxes, Axis
import Zygote: withgradient, withjacobian, pullback, jacobian
## optional: skip certain parts of code in gradient computation:
import ChainRulesCore: @ignore_derivatives, ignore_derivatives

rng = Random.seed!(31415)    # reproducible pseudo-random numbers
## Initialize ann with shared parameters
nn = LRModel();
ps, st = Lux.setup(rng, nn); # parameters and states

logitcrossentropy(y_pred, y) = Lux.mean(-sum(y .* Lux.logsoftmax(y_pred); dims=1))
function compute_losses(nn, ps, st, X, Y)
    Y_pred, _st = Lux.apply(nn, X, ps, st)
    losses = [
        logitcrossentropy(Y_pred[1], Y.llabels);
        logitcrossentropy(Y_pred[2], Y.rlabels)
    ]
    return losses, _st
end

import LinearAlgebra as LA
function losses_states_jac(nn, ps_c, st, X, Y, mode=Val(:standard)::Val{:standard}; norm_grads=false)
    local new_st
    losses, jac_t = withjacobian(ps_c) do params
        losses, new_st = compute_losses(nn, params, st, X, Y)
        losses
    end
    jac = only(jac_t)
    if norm_grads
        jac ./= LA.norm.(eachrow(jac))
    end
    return losses, new_st, jac
end

function apply_multidir(
    nn, ps_c, st, X, Y, mode=Val(:full)::Val{:standard};
    lr=Float32(1e-3), jacmode=:standard, norm_grads=false
)
    losses, new_st, jac = losses_states_jac(nn, ps_c, st, X, Y, Val(jacmode); norm_grads)
    dir = multidir(jac)
    return losses, ps_c .+ lr .* dir, new_st
end

import Printf: @sprintf
function train(
    nn, ps_c, st, dat;
    norm_grads=false,
    dirmode=:full, jacmode=:standard, num_epochs=1, lr=Float32(1e-3)
)
    # printing offsets:
    epad = ndigits(num_epochs)
    num_batches = length(dat)
    bpad = ndigits(num_batches)
    # safeguard learning type
    lr = eltype(ps_c)(lr)
    # training loop
    for e_ind in 1:num_epochs
        @info "----------- Epoch $(lpad(e_ind, epad)) ------------"
        epoch_stats = @timed for (b_ind, (X, Y)) in enumerate(dat)
            batch_stats = @timed begin
                losses, ps_c, st = apply_multidir(nn, ps_c, st, X, Y, Val(dirmode); lr, jacmode, norm_grads)
                # excuse this ugly info string, please...
                @info "\tE/B/Prog $(lpad(e_ind, epad)) / $(lpad(b_ind, bpad)) / $(@sprintf "%3.2f" b_ind/num_batches) %; l1 $(@sprintf "%3.4f" losses[1]); l2 $(@sprintf "%3.4f" losses[2])"
            end
            @info "\t\tBatch time: $(@sprintf "%8.2f msecs" batch_stats.time*1000)."
            if b_ind >= 2
                break
            end
        end
        @info "Epoch time: $(@sprintf "%.2f secs" epoch_stats.time)"
    end
    return ps_c, st
end

ps_fin, st_fin = train(nn, ps_c, st, dat);