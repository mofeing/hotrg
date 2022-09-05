# Higher-order Tensor Renormalization Group for 2D Ising model
using LinearAlgebra
using TensorOperations
import Base: size
using Profile

function contract_tensors(top::Array{T,4}, bottom::Array{T,4}) where {T<:AbstractFloat}
    @tensor m[lt, lb, rt, rb, u, d] := top[lt, rt, u, i] * bottom[lb, rb, i, d]
    m = reshape(m, (size(m, 1:2), size(m, 3:4), size(m, 5), size(m, 6)))
end

function update_tensor(U::Matrix{T}, M::Array{T,4}) where {T<:Number}
    @tensor _T[l, r, u, d] := U[i, l] * M[i, j, u, d] * U[j, r]
end

square(x) = ^(x, 2)
ϵₗ(S::Array{T,4}, χ::Integer) where {T<:Number} = sum((square ∘ abs).(S[χ:end, :, :, :]))
ϵᵣ(S::Array{T,4}, χ::Integer) where {T<:Number} = sum((square ∘ abs).(S[:, χ:end, :, :]))

size(A::AbstractArray{T,N}, r::AbstractRange) where {T,N} = prod(size(A, i) for i in r)

function hosvd_sides(A::Array{T,4}) where {T<:Number}
    shape = size(A)

    # left SVD
    A = reshape(A, (size(A, 1), size(A, 2:4)))
    Uₗ, Σ, V = svd(A; full=false)

    A = Σ .* V'

    A = reshape(A, shape)

    # right SVD
    A = permutedims(A, (2, 1, 3, 4))
    A = reshape(A, (size(A, 1), size(A, 2:4)))
    Uᵣ, Σ, V = svd(A; full=false)

    S = reshape(Σ .* V', shape)

    S, Uₗ, Uᵣ
end

function trace(A::Array{T,4}) where {T<:Number}
    scalar(@tensor a[] := T[i, i, j, j])
end

W(t) = [sqrt(cosh(1 / t)) sqrt(sinh(1 / t)); sqrt(cosh(1 / t)) -sqrt(sinh(1 / t))]

partition_tensor(t) = begin
    w = W(t)
    _T = zeros(2, 2, 2, 2)
    for (l, r, u, d) in Iterators.product(1:2, 1:2, 1:2, 1:2)
        _T[l, r, u, d] = sum(w[:, l] .* w[:, r] .* w[:, u] .* w[:, d])
    end
    _T
end

magnetization(t) = begin
    w = W(t)
    m = [1.0, -1.0]
    _T = zeros(2, 2, 2, 2)
    for (l, r, u, d) in Iterators.product(1:2, 1:2, 1:2, 1:2)
        _T[l, r, u, d] = sum(w[:, l] .* w[:, r] .* w[:, u] .* w[:, d] .* m)
    end
    _T
end

###
temperature = 2.0
repeats = 10
χ = 10

function one_iteration(H, T, χ)
    # contract with environment
    Mh = contract_tensors(H, T)
    Mt = contract_tensors(T, T)

    # decompose tensors
    S, Uₗ, Uᵣ = hosvd_sides(Mt)

    # compute projector
    U = if ϵₗ(S, χ) < ϵᵣ(S, χ)
        Uₗ
    else
        Uᵣ
    end

    if size(U, 1) > χ
        U = U[:, 1:χ]
    end
    U /= sqrt(maximum(S))

    # update tensors
    H = update_tensor(U, Mh)
    T = update_tensor(U, Mt)

    H, T
end


function benchmark(temperature, repeats, χ)
    T = partition_tensor(temperature)
    H = magnetization(temperature)

    for _ in 1:repeats
        for perm in Iterators.take(Iterators.cycle([(1, 2, 4, 3) (3, 4, 1, 2)]), 4)
            H, T = one_iteration(H, T, χ)
            H = permutedims(H, perm)
            T = permutedims(T, perm)
        end
        m = (@tensor H[i, i, j, j]) / (@tensor T[i, i, j, j])
        println("magnetization=$m")
    end
end

@timev benchmark(2.0, 1, 8)

@timev benchmark(2.0, 10, 8)

@profview benchmark(2.0, 10, 8)

@timev benchmark(2.0, 5, 12)

@profview benchmark(2.0, 5, 12)

@timev benchmark(2.0, 5, 14)

@profview benchmark(2.0, 2, 14)