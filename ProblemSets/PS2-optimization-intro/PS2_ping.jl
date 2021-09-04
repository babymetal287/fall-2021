using LinearAlgebra
using Random
using Statistics
using DataFrames
using FreqTables

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Optim
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

using GLM
bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(alpha, X, d)
    loglike = sum([d[i]*log(exp(sum(alpha.*X[i,:]))/(1+exp(sum(alpha.*X[i,:]))))+(1-d[i])*log(1/(1+exp(sum(alpha.*X[i,:])))) for i=1:size(X,1)])
    return loglike
end

startval = rand(size(X,2))
obj = optimize(alpha->-logit(alpha, X, y), startval, LBFGS())
println(obj.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# see Lecture 3 slides for example
res = glm(X, y, Binomial(), LogitLink())
println(res)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation


function mlogit(alpha, X, d)
    loglike = 0
    for i = 1:size(X,1)
        term1 = sum([convert(Float64,d[i]==j)*sum(alpha[:,j].*X[i,:]) for j =1:6])
        term2 = log(1 + sum([exp(sum(alpha[:,j].*X[i,:])) for j =1:6]))
        loglike += term1 - term2
    end
    return loglike
end


#startval = zeros(size(X,2),6)
#startval = rand(size(X,2),6)
startval = 2*rand(size(X,2),6).-1
obj = optimize(alpha->-mlogit(alpha, X, y), startval, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100000))
println(obj.minimizer)
