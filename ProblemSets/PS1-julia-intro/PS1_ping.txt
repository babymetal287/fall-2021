using JLD2
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

#(l)
function q1()
    #(a)
    rng = Random.seed!(1234)
    A = rand(rng,-5:10,(10,7))
    d = Normal(-2,15)
    B = rand(rng,d,(10,7))
    C = hcat(A[1:5,1:5],B[1:5,6:end])
    D = copy(A)
    D[D .> 0] .= 0
    #(b)
    println(length(A))
    #(c)
    println(length(unique(D)))
    #(d)
    E = reshape(B,(1,:))
    #(e)
    F =cat(A, B, dims=3)
    #(f)
    F = permutedims(F, (3,1,2))
    #(g)
    G = kron(B, C)
    #if I use kron(C, F), there is a dimentional mismatch
    #(h)
    @save "matrixpractice.jld"  A B C D E F G
    #(i)
    @save "firstmatrix.jld"  A B C D
    #(j)
    C = DataFrame(C,:auto)
    CSV.write( "Cmatrix.csv",  C)
    #(k)
    D = DataFrame(D,:auto)
    CSV.write( "Dmatrix.dat",  D)
    return A,B,C,D
end

A,B,C,D = q1()


#(2)
#(f)
function q2(A,B,C)
    #(a)
    AB = zeros(size(A))
    for j = 1:size(AB,2)
        for i = 1:size(AB,1)
            AB[i,j]= A[i,j]*B[i,j]
        end
    end
    AB2 = A.*B
    #{b}
    Cprime = vec(Array(C))
    Cprime = [k for k in Cprime if k <5 && k>=-5]
    Cprime2 = vec(Array(C))
    Cprime2 = filter(t ->  - 5 <= t <= 5, Cprime2)
    #(c)
    X = zeros(15,6,5)
    for t in 1:size(X,3)
        for k in 1:size(X,2)
            if k == 1
                X[:,k,t] = ones(size(X,1))
            elseif k ==2
                p =0.75*(6-t)/5
                d = Binomial(1,p)
                N = size(X,1)
                X[:,k,t] = rand(d,N)
            elseif k ==3
                m = 15 +t -1
                dev = 5*(t-1)
                N = size(X,1)
                d = Normal(m,dev)
                X[:,k,t] = rand(d,N)
            elseif k==4
                m = pi*(6-t)/3
                dev = 1/ℯ
                N = size(X,1)
                d = Normal(m,dev)
                X[:,k,t] = rand(d,N)
            elseif k ==5
                d = Binomial(20,0.6)
                N = size(X,1)
                X[:,k,t] = rand(d,N)
            else
                d = Binomial(20,0.5)
                N = size(X,1)
                X[:,k,t] = rand(d,N)
            end
        end
    end                
    #(d)
    β = zeros(6,5)
    for k in 1:size(β,1)
        for t in 1:size(β,2)
            if k == 1
                β[k,t] = 0.75 + 0.25 *t
            elseif k == 2
                β[k,t] = log(t)
            elseif k == 3
                β[k,t] = - sqrt(t)
            elseif k == 4
                β[k,t] = ℯ^t - ℯ^(t+1)
            elseif k == 5
                β[k,t] = t
            else
                β[k,t] = t/3
            end
        end
    end                
    #(e)
    Y = zeros(15,5)
    for t in 1:size(X,3)
        d = Normal(0,0.36)
        N =size(X,1)
        ϵ =  rand(d,N)
        Y[:,t]= X[:,:,t]*β[:,t] + ϵ
    end
    return
end
q2(A,B,C)

#(3)
#(g)
function q3()
    df = CSV.read("nlsw88.csv",DataFrame)
    @save "nlsw88.jld"  df
    #(b)
    println("Never married rate: ",size(df[(df.never_married.==1),:],1)/size(df,1))
    println("college graduates rate: ",size(df[(df.collgrad.==1),:],1)/size(df,1))
    #(c)
    freqtable(df.race)
    #(d)
    summarystats = describe(df,:mean, :std, :min, :q25, :median, :q75, :max, :nunique, :nmissing)
    println("Number of missing values of grade is ",summarystats[summarystats.variable.==:grade,:].nmissing)
    summarystats = Array(summarystats)
    #(e)
    cv = freqtable(df.industry,df.occupation)
    #(f)
    df1 = select(df, [:industry,:occupation, :wage])
    gdf = groupby(df1, [:industry,:occupation])
    df_mean = combine(gdf,:wage => mean)
    return
end

q3()

#(4)
#(h)
function q4()
    #(a)
    d = load("firstmatrix.jld")
    A = d["A"]
    B = d["B"]
    C = d["C"]
    D = d["D"]
    #(b)
    function matrixops(A,B)
        #=
        The function does the following calculations:
        (i) AB2: the element-by-element product of the inputs, (ii) AB3: the product A0B, and (iii) AB4: the sum of all the elements of A+B.
        =#
        if size(A)!=size(B)
            throw(error("inputs must have the same size."))
            return
        end
        AB2 = A.*B
        AB3 = A'B
        AB4 = sum(A+B)
        return AB2,AB3, AB4
    end
    #(d)
    println(matrixops(A,B))
    #(f)
    #println(matrixops(C,D))
    #(g)
    df = CSV.read("nlsw88.csv",DataFrame)
    ttl_exp = convert(Array,df.ttl_exp)
    wage = convert(Array,df.wage)
    println(matrixops(ttl_exp,wage))
    return
end

q4()
