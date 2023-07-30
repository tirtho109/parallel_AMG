using PartitionedArrays

np = 4
n = 50
#const ranks = distribute_with_mpi(LinearIndices((np,)))
const ranks = LinearIndices((np,))

neigs_snd = map(ranks) do rank
    # if rank != np
    #     [rank + 1]
    # else
    #     [1]
    # end
    if rank == 1
        [2,4]
    elseif  rank == 2
        [3,1]
    elseif rank == 3
        [4,2]
    else
        [1,3]
    end
end

data_snd = map(ranks) do rank
    [[10*rank],[2*rank]]
end

graph = ExchangeGraph(neigs_snd)

data_rcv = exchange(data_snd,graph) |> fetch

map(data_snd) do snd
    @show snd
end
map(neigs_snd) do neigh
    @show neigh
end
map(data_rcv) do rcv
    @show rcv
end