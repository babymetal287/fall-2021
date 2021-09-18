using Distributions
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables

 #:::::::::::::::::::::::::::::::::::::::::::::::::::
 # question 1
 #:::::::::::::::::::::::::::::::::::::::::::::::::::
 
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code


function mlogit_with_Z(theta, X, Z, y)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
        
    P = num./repeat(dem,1,J)
        
    loglike = -sum( bigY.*log.(P) )
        
    return loglike
end
startvals = [2*rand(7*size(X,2)).-1; .1]
td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
# run the optimizer
theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
theta_hat_mle_ad = theta_hat_optim_ad.minimizer
# evaluate the Hessian at the estimates
H  = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([theta_hat_mle_ad theta_hat_mle_ad_se]) # these standard errors match Stata

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#Yes. Estimated gamma in PS3 is -0.09419383447879262. Yet gamma/100 is the change in utility with a 1% increase in expected wage.
#It should not be negative. Here I got a postitive number(1.3074804109794527) so it is more reasonable.

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#(a)
include("lgwt.jl") # make sure the function gets read in
# define distribution
d = Normal(0,1) # mean=0, standard deviation=1
# get quadrature nodes and weights for 7 grid points
nodes, weights = lgwt(7,-4,4)
# now compute the integral over the density and verify it
println(sum(weights.*pdf.(d,nodes)))
# now compute the expectation and verify it
println(sum(weights.*nodes.*pdf.(d,nodes)))



#(b)
#(1)
d = Normal(0,2) 
nodes, weights = lgwt(7,-5,5)
println(sum(weights.*(nodes.^2).*pdf.(d,nodes)))
#(2)
d = Normal(0,2) 
nodes, weights = lgwt(10,-5,5)
println(sum(weights.*(nodes.^2).*pdf.(d,nodes)))
#(3)
#As the number of quadrature points increase, the quadrature approximates closer to the true value.

#(c)
#(1)
d = Normal(0,2) 
a = -5*2
b = 5*2
z = Uniform(a,b)
x = rand(z,1000000)
println((b-a)*mean((x.^2).*pdf.(d,x)))
#(2)
println((b-a)*mean(x.*pdf.(d,x)))
#(3)
println((b-a)*mean(pdf.(d,x)))
#(4)
x = rand(z,1000)
println((b-a)*mean((x.^2).*pdf.(d,x)))
println((b-a)*mean(x.*pdf.(d,x)))
println((b-a)*mean(pdf.(d,x)))
#when D = 1000000, the simulated integral approximates the true value better than that when D = 1000




function mixed_logit_normal(theta, X, Z, y)
        
    alpha = theta[1:end-2]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    d = Normal(mu_gamma,sigma_gamma) 
    gammas, weights = lgwt(10,-4,4)
    M = size(gammas,1)
    F = ones(N,M)
    for i =1:M
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J]).*gammas[i])
            dem .+= num[:,j]
        end
        P = (num./repeat(dem,1,J))
        
        for j=1:J
            P[:,j]=P[:,j].^bigY[:,j]
            F[:,i].*=P[:,j]
        end
    end          
        
    loglike = -sum([log(sum(weights.*F[i,:].*pdf.(d,gammas))) for i = 1:N])
        
    return loglike
end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::


startvals = [2*rand(7*size(X,2)).-1; 0; 1]
obj = optimize(alpha->mixed_logit_normal(alpha, X, Z, y), startvals, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=50))
beta = obj.minimizer[1:end-2]
mu_gamma = obj.minimizer[end-1]
sigma_gamma = obj.minimizer[end]
println('beta is ',beta)
println('mu_gamma is ',mu_gamma)
println('sigma_gamma is ', sigma_gamma)



#I also tried this code yet MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 12})
# startvals = [2*rand(7*size(X,2)).-1; 0; 1]
# td = TwiceDifferentiable(theta -> mixed_logit_normal(theta, X, Z, y), startvals; autodiff = :forward)
# theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100000, show_trace=true, show_every=50))
# theta_hat_mle_ad = theta_hat_optim_ad.minimizer
# H  = Optim.hessian!(td, theta_hat_mle_ad)
# theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
# println([theta_hat_mle_ad theta_hat_mle_ad_se]) # these standard errors match Stata

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::


function mixed_logit_normal2(theta, X, Z, y)
        
    alpha = theta[1:end-2]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    d = Normal(mu_gamma,sigma_gamma) 
    gammas, weights = lgwt(10,-4,4)
    M = size(gammas,1)
    F = ones(N,M)
    for i =1:M
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J]).*gammas[i])
            dem .+= num[:,j]
        end
        P = (num./repeat(dem,1,J))
        
        for j=1:J
            P[:,j]=P[:,j].^bigY[:,j]
            F[:,i].*=P[:,j]
        end
    end     
    
    a = -5*sigma_gamma 
    b = 5*sigma_gamma 
    z = Uniform(a,b)
    x = rand(z,1000000)
    loglike = -sum([log((b-a)*mean(F[i,:].*pdf.(d,gammas))) for i = 1:N])
        
    return loglike
end

startvals = [2*rand(7*size(X,2)).-1; 0; 1]
obj = optimize(alpha->mixed_logit_normal2(alpha, X, Z, y), startvals, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=50))
beta = obj.minimizer[1:end-2]
mu_gamma = obj.minimizer[end-1]
sigma_gamma = obj.minimizer[end]
println('beta is ',beta)
println('mu_gamma is ',mu_gamma)
println('sigma_gamma is ', sigma_gamma)


