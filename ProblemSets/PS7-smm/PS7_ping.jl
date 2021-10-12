using SMM
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

#GMM
function ols_gmm(b, X, y)
    g = y .- X*b
    J = g'*I*g
    return J
end
res =  optimize(b -> ols_gmm(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(res.minimizer)


#closed form solution
res1 = inv(X'*X)*X'*y
println(res1)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

#MlE
function mlogit(alpha, X, y)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

alpha_zero = zeros(6*size(X,2))
alpha_rand = rand(6*size(X,2))
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
alpha_start = alpha_true.*rand(size(alpha_true))
#println(size(alpha_true))
alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)


#GMM
function mlogit_gmm(alpha, X, y)
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end
        
    P = num./repeat(dem,1,J)
        
    
    g = bigY .-P
    J = sum(g'*I*g)
    return J
end

alpha_rand = rand(6*size(X,2))
alpha_hat_optim1 = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_gmm = alpha_hat_optim1.minimizer
println(alpha_hat_gmm)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function mlogit_sim(alpha,X,K,J,N)
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end
        
    P = num./repeat(dem,1,J)
    epsilon = rand(N)

    Y = zeros(N)
    for i =1:N
        temp = zeros(J)
        for j = 1:J
            temp[j] = convert(Int,(sum(P[i,j:end])>epsilon[i]))
        end
        Y[i] = sum(temp)
    end
    return Y
end

K = 3
J = 7
N = 10000
X = randn((N,K))
alpha_true = rand((J-1)*K)
println("beta true:",alpha_true)
y_sim = mlogit_sim(alpha_true,X,K,J,N)
alpha_start = alpha_true.*rand(size(alpha_true))
alpha_hat_optim2 = optimize(a -> mlogit(a, X, y_sim), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_sim = alpha_hat_optim2.minimizer
println("beta estimated:",alpha_hat_sim)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
MA = SMM.parallelNormal()
dc = SMM.history(MA.chains[1])
dc = dc[dc[:accepted].==true,:]
println(describe(dc))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function mlogit_smm(alpha, X, y, D) 
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    
   # N+1 moments in both model and data
    gmodel = zeros(N+1,D)
    # data moments are just the y vector itself
    # and the variance of the y vector
    gdata  = vcat(y,var(y))
    
    Random.seed!(1234)                    
   
    # simulated model moments
    
    for d = 1:D
        y_sim = mlogit_sim(alpha,X,K,J,N)
        gmodel[1:end-1,d] = y_sim
        gmodel[  end  ,d] = var(y_sim)
    end
    # criterion function
    err = vec(gdata .- mean(gmodel; dims=2))
    # weighting matrix is the identity matrix
    # minimize weighted difference between data and moments
    J = err'*I*err
    
    return J
end


alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
alpha_start = alpha_true.*rand(size(alpha_true))
D = 2000
#println(size(alpha_true))
alpha_hat_optim = optimize(a -> mlogit_smm(a, X, y,D), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_smm = alpha_hat_optim.minimizer
println(alpha_hat_smm)