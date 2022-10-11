module StaticBitArrays

import Base: IndexStyle, size, getindex, setindex!, similar, all, any

export StaticBitVector, StaticBitMatrix, StaticBitArray

const S = sizeof(UInt) * 8

mutable struct StaticBitArray{D, N, L} <: AbstractArray{Bool, N}
    data::NTuple{L, UInt}
end

const StaticBitVector = StaticBitArray{Bool, 1}
const StaticBitMatrix = StaticBitArray{Bool, 2}

function StaticBitArray(::UndefInitializer, D::Vararg{Int, N}) where N
    d, r = divrem(N, S)
    L = d + !iszero(r)

    StaticBitArray{D, N, L}(ntuple(_ -> zero(UInt), L))
end

StaticBitArray(::UndefInitializer, D::NTuple{Int, N}) where N = StaticBitArray(undef, D...)

Base.IndexStyle(::StaticBitArray) = IndexLinear()

@inline Base.size(::StaticBitArray{D}) where D = D

Base.@propagate_inbounds function Base.getindex(arr::StaticBitArray{D, N, L}, i::Int) where {D, N, L}
    @boundscheck checkbounds(arr, i)

    @inbounds begin
        x = unsafe_trunc(UInt, i - 1)
        d, r = divrem(x, S)
        k = arr.data[d + 1]
        m = (one(UInt) << r)
    end

    return !iszero(k & m)
end

Base.@propagate_inbounds function Base.getindex(arr::StaticBitArray{D, N, L}, I::Vararg{Int, N}) where {D, N, L}
    @boundscheck checkbounds(arr, I...)
    @inbounds arr[LinearIndices(D)[I...]]
end

Base.@propagate_inbounds function Base.setindex!(arr::StaticBitArray{D, N, L}, b::Bool, i::Int) where {D, N, L}
    @boundscheck checkbounds(arr, i)

    @inbounds begin
        x = unsafe_trunc(UInt, i - 1)
        d, r = divrem(x, S)
        o = arr.data
        k = o[d + 1]
        m = ~(one(UInt) << r)
        v = UInt(b) << r
        t = Base.setindex(o, k & m | v, d + 1)

        arr.data = t
    end

    return b
end

Base.@propagate_inbounds function Base.setindex!(arr::StaticBitArray{D, N, L}, b::Bool, I::Vararg{Int, N}) where {D, N, L}
    @boundscheck checkbounds(arr, I...)
    @inbounds arr[LinearIndices(D)[I...]] = b
end

Base.similar(::StaticBitArray{D}) where {D} = StaticBitArray(undef, D)

function _bmap(f::Function, A::StaticBitArray{D, N, L}) where {D, N, L}
    data = ntuple(Val(L)) do i
        @inbounds f(A.data[i])
    end

    return StaticBitArray{D, N, L}(data)
end

function _bmap(f::Function, A::StaticBitArray{D, N, L}, B::StaticBitArray{D, N, L}) where {D, N, L}
    data = ntuple(Val(L)) do i
        @inbounds f(A.data[i], B.data[i])
    end

    return StaticBitArray{D, N, L}(data)
end

function _bmap(f::Function, A::Vararg{StaticBitArray{D, N, L}, K}) where {D, N, L, K}

    data = ntuple(Val(L)) do i
        f(map(A) do arr
            @inbounds(arr.data[i])
        end...)
    end

    return StaticBitArray{D, N, L}(data)
end

Base.map(::typeof(~), A::StaticBitArray) = _bmap((~), A)
Base.map(::typeof(&), A::StaticBitArray, B::StaticBitArray) = _bmap((&), A, B)
Base.map(::typeof(⊻), A::StaticBitArray, B::StaticBitArray) = _bmap((⊻), A, B)
Base.map(::typeof(|), A::StaticBitArray, B::StaticBitArray) = _bmap((|), A, B)

function Base.any(arr::StaticBitArray{D, N, L}) where {D, N, L}
    for i = 1:L-1
        if !iszero(arr.data[i])
            return true
        end
    end

    mask = unsafe_trunc(UInt, -1) >> (S - (N % S))

    return !iszero(mask & arr.data[L])
end

Base.any(::StaticBitArray{D, N, 0}) where {D, N} = false

function Base.any(arr::StaticBitArray{D, N, L}) where {D, N, L}
    for i = 1:L-1
        if !iszero(arr.data[i])
            return true
        end
    end

    mask = ~zero(UInt) >> (S - (N % S))

    return !iszero(mask & arr.data[L])
end

Base.all(::StaticBitArray{D, N, 0}) where {D, N} = true

function Base.all(arr::StaticBitArray{D, N, L}) where {D, N, L}
    for i = 1:L-1
        if arr.data[i] !== ~zero(UInt)
            return false
        end
    end

    mask = unsafe_trunc(UInt, -1) >> (S - (N % S))

    return iszero(mask & ~arr.data[L])
end

end