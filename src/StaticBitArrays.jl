module StaticBitArrays

import Base: IndexStyle, size, getindex, setindex!, similar, all, any, map

export StaticBitVector, StaticBitMatrix, StaticBitArray

const S = sizeof(UInt) * 8

"""
    StaticBitArray(::UndefInitializer, dims)
    StaticBitArray(::UndefInitializer, dims...)

Constructs a zero-initialized N-dimensional BitArray. 
This Array doesn't support push! or insert! but works with most other AbstractArray algorithms.

# Examples
```julia-repl
julia> StaticBitArray(undef, 5, 5)
5×5 StaticBitMatrix{(5, 5), 2, 1}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
```
"""
mutable struct StaticBitArray{D, N, L} <: AbstractArray{Bool, N}
    data::NTuple{L, UInt}
end

const StaticBitVector = StaticBitArray{D, 1, L} where {D, L}
const StaticBitMatrix = StaticBitArray{D, 2, L} where {D, L}

function StaticBitArray{D, N}(value::Bool) where {D, N}
    D === () && return StaticBitArray{(), 0, 0}(())

    d, r = divrem(prod(D), S)
    L = d + !iszero(r)

    c = (zero(UInt), ~zero(UInt))[value + 1]

    StaticBitArray{D, length(D), L}(ntuple(_ -> c, L))
end

StaticBitArray{D}(v::Bool) where {D} = StaticBitArray{D, length(D)}(v)
StaticBitArray{D}(::UndefInitializer) where {D} = StaticBitArray{D, length(D)}(false)

StaticBitArray(v::Bool, D::Vararg{Int, N}) where {N} = StaticBitArray{D, N}(v)
StaticBitArray(v::Bool, D::NTuple{N, Int}) where {N} = StaticBitArray{D, N}(v)

StaticBitArray(::UndefInitializer, D::Vararg{Int, N}) where {N} = StaticBitArray{D, N}(false)
StaticBitArray(::UndefInitializer, D::NTuple{N, Int}) where {N} = StaticBitArray{D, N}(false)


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

function _bmap(f::Function, A::StaticBitArray{D, N, L}...) where {D, N, L}

    data = ntuple(Val(L)) do i
        f(map(A) do arr
            @inbounds(arr.data[i])
        end...)
    end

    return StaticBitArray{D, N, L}(data)
end

"""
    map(::typeof(~), ::StaticBitArray)
    map(::typeof(&), ::StaticBitArray, ::StaticBitArray)
    map(::typeof(⊻), ::StaticBitArray, ::StaticBitArray)
    map(::typeof(|), ::StaticBitArray, ::StaticBitArray)

Specialized methods to handle bitoperations like ~, &, | and ⊻ efficiently.

# Examples
```julia-repl
julia> A = StaticBitArray(undef, 3, 2)
2×3 StaticBitMatrix{(2, 3), 2, 1}:
 0  0  0
 0  0  0

julia> B = map((~), A)
2×3 StaticBitMatrix{(2, 3), 2, 1}:
 1  1  1
 1  1  1
```
"""
Base.map(::typeof(~), A::StaticBitArray) = _bmap((~), A)
Base.map(::typeof(&), A::StaticBitArray, B::StaticBitArray) = _bmap((&), A, B)
Base.map(::typeof(⊻), A::StaticBitArray, B::StaticBitArray) = _bmap((⊻), A, B)
Base.map(::typeof(|), A::StaticBitArray, B::StaticBitArray) = _bmap((|), A, B)


"""
    all(::StaticBitArray)

Test whether all bits of the array are set.    

# Examples

```julia-repl
julia> arr = StaticBitArray(false, 3, 3)
3×3 StaticBitMatrix{(3, 3), 1}:
 0  0  0
 0  0  0
 0  0  0

julia> any(arr)
false

julia> arr[2, 2] = true ;

julia> any(arr)
true

```
"""
function Base.any(arr::StaticBitArray{D, N, L}) where {D, N, L}
    for i = 1:L-1
        if !iszero(arr.data[i])
            return true
        end
    end

    mask = unsafe_trunc(UInt, -1) >> (S - (prod(D) % S))

    return !iszero(mask & arr.data[L])
end
Base.any(::StaticBitArray{D, N, 0}) where {D, N} = false


"""
    all(::StaticBitArray)

Test whether all bits of the array are set.    

# Examples

```julia-repl
julia> arr = StaticBitArray(true, 3, 3)
3×3 StaticBitMatrix{(3, 3), 1}:
 1  1  1
 1  1  1
 1  1  1

julia> all(arr)
true

julia> arr[2, 2] = false ;

julia> all(arr)
false

```
"""
function Base.all(arr::StaticBitArray{D, N, L}) where {D, N, L}
    for i = 1:L-1
        if arr.data[i] !== ~zero(UInt)
            return false
        end
    end

    mask = unsafe_trunc(UInt, -1) >> (S - (prod(D) % S))

    return iszero(mask & ~arr.data[L])
end
Base.all(::StaticBitArray{D, N, 0}) where {D, N} = true

end