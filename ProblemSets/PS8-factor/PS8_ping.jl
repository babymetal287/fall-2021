using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
using MultivariateStats
using Distributions

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS8-factor/nlsy.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)
show(df, allcols=true)

ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + grad4yr + gradHS),df)
println(ols)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
println(cor(Matrix(@select(df, :asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK))))


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + grad4yr + gradHS + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK),df)
println(ols)
#it will be problematic to directly include these in the regression since the correlation among six asvabs is too high



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
asvabMat = Matrix(@select(df, :asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK))'
M = fit(PCA, asvabMat; maxoutdim=1)
asvabPCA = MultivariateStats.transform(M, asvabMat)
insertcols!(df, 16, :asvabFac => vec(asvabPCA'))
ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + grad4yr + gradHS + asvabFac),df)
println(ols)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
asvabMat = Matrix(@select(df, :asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK))'
M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
asvabFA= MultivariateStats.transform(M, asvabMat)
insertcols!(df, 17, :asvabFacFA => vec(asvabFA'))
ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + grad4yr + gradHS + asvabFacFA),df)
println(ols)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
include("E://oklohoma//ECON6343//fall-2021-master//ProblemSets//PS4-mixture//lgwt.jl") # make sure the function gets read in

function ms_mle(x,df,N_grid)
    N = size(df,1)
    J = 6
    alpha = reshape(x[1:4*J],(4,J))
    beta = reshape(x[4*J+1:4*J+7],(7,1))
    gamma = x[4*J+8:4*J+13]
    sigmaj = x[4*J+14:4*J+19]
    delta = x[end-1]
    sigmaw = x[end]
    y = Matrix(@select(df, :logwage))
    M = Matrix(@select(df, :asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK))
    Xm = hcat(ones(N),Matrix(@select(df, :black, :hispanic, :female)))
    X = hcat(ones(N),Matrix(@select(df, :black, :hispanic, :female,:schoolt, :grad4yr, :gradHS)))
    d = Normal(0,1)
    nodes, weights = lgwt(N_grid,-4,4)
    Random.seed!(1234)
    xi = randn(N_grid)
    Li = ones(N)
    lik = zeros(N)
    for i = 1:N_grid
        for j = 1:J
            Li .*= pdf.(d, (M[:,j].-Xm*alpha[:,j].-gamma[j]*xi[i])./sigmaj[j])./sigmaj[j]
        end
        Li = Li .*pdf.(d, (y.-X*beta.-delta*xi[i])./sigmaw)./sigmaw
        lik =  lik .+ Li .*weights[i].*pdf.(d,xi[i])
    end
    loglik = -sum(log.(lik))
    return loglik
end

x_start = rand(45)
lower = vcat(repeat([-Inf],37),zeros(6),[-Inf],[0])
upper = repeat([Inf],45)
N_grid = 1000
x_optim = optimize(x -> ms_mle(x, df, N_grid),lower, upper, x_start, Fminbox(NelderMead()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
x_mle = x_optim.minimizer
println(x_mle)