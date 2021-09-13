using DataFrames
using CSV
using HTTP
using Optim
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using FreqTables

 #:::::::::::::::::::::::::::::::::::::::::::::::::::
 # question 1
 #:::::::::::::::::::::::::::::::::::::::::::::::::::
 
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body,DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation


function mlogit(alpha, X, Z, y)
    J = size(unique(y),1)
    beta = reshape(alpha[1:end-1],size(X,2),J-1)
    gamma = alpha[end]
    Z_diff = Z[:,1:end-1].-Z[:,end]
    loglike = 0
    for i = 1:size(X,1)
        term1 = sum([convert(Float64,y[i]==j)*(sum(beta[:,j].*X[i,:])+sum(gamma.*Z_diff[i,j])) for j =1:J-1])
        term2 = log(1 + sum([exp((sum(beta[:,j].*X[i,:])+sum(gamma.*Z_diff[i,j]))) for j =1:J-1]))
        loglike += term1 - term2
    end
    return -loglike
end

J = size(unique(y),1)
start_beta = rand((J-1)*size(X,2))
start_gamma = rand(1)
alpha0 = vcat(start_beta,start_gamma)
obj = optimize(alpha->mlogit(alpha, X, Z, y), alpha0, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=50))
beta = obj.minimizer[1:end-1]
gamma = obj.minimizer[end]
println("beta is ",beta)
println("gamma is ",gamma)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#lamdba is the coefficient of marginal effect of wage difference. 


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::


function nestedlogit(alpha, X, Z, y)
    K1= size(X,2)
    K2= size(Z,2)
    #WC:1 Professional/Technical; 2 Managers/Administrators; 3 Sales
    J_WC = 3
    s0 = 1
    s1 = K1*J_WC
    beta_wc = reshape(alpha[s0:s1],K1,J_WC)
    #BC: 4 Clerical/Unskilled; 5 Craftsmen;6 Operatives; 7 Transport
    J_BC = 4
    s2 = s1 + 1
    s3 = s1 + K1 * J_BC
    beta_bc = reshape(alpha[s2:s3],K1,J_BC)
    
    lambda_wc = alpha[s3+1]
    lambda_bc = alpha[s3+2]
    gamma = alpha[s3+3]
    # Other occupations
    J_O = 1
    beta_O = zeros(K1)
    
    beta = [beta_wc beta_bc beta_O]
    
    N = length(y)
    WC = zeros(N,J_WC)
    BC = zeros(N,J_BC)
    O = zeros(N,J_O)
    J = length(unique(y))
    Z_diff = Z[:,1:end].-Z[:,end]
    
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    
    
    num = zeros(N,J)
    num1 = zeros(N,J)
    num2 = zeros(N,3)
    dem = zeros(N)
    for j=1:J
        if j <= 2
            num1[:,j] = exp.((X*beta[:,j]+gamma*Z_diff[:,j])/lambda_wc)
            num2[:,1] .+= num1[:,j]
        elseif j == 3
            num1[:,j] = exp.((X*beta[:,j]+gamma*Z_diff[:,j])/lambda_wc)
            num2[:,1] .+= num1[:,j]
            
            dem .+= num2[:,1].^lambda_wc
            for m =1:3
                num[:,m]=num1[:,m].*num2[:,1].^(lambda_wc-1)
            end
            
        elseif j >3 && j<7
            num1[:,j] = exp.((X*beta[:,j]+gamma*Z_diff[:,j])/lambda_bc)
            num2[:,2] .+= num1[:,j]
            
        elseif j == 7 
            num1[:,j] = exp.((X*beta[:,j]+gamma*Z_diff[:,j])/lambda_bc)
            num2[:,2] .+= num1[:,j]
            
            dem .+= num2[:,2].^lambda_bc
            for m =4:7
                num[:,m]=num1[:,m].*num2[:,2].^(lambda_bc-1)
            end
            
        else
            num1[:,j] = exp.(X*beta[:,j]+gamma*Z_diff[:,j])
            num2[:,3] .+= num1[:,j]
            
            num[:,8] = num2[:,3]
            dem .+= num2[:,3]
        end
        
        
       
    end
    
    P = num./repeat(dem,1,J)
    loglike = -sum( bigY.*log.(P))
    
    return loglike
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
K1= size(X,2)
J_WC = 3
J_BC = 4
J = length(unique(y))
alpha0 = rand(K1*J+3)
alpha_hat_optim = optimize(alpha -> nestedlogit(alpha, X, Z, y), alpha0, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=1000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
#println(alpha_hat_mle)
println("beta_wc is ",alpha_hat_mle[1:K1*J_WC])
println("beta_bc is ",alpha_hat_mle[K1*J_WC+1:K1*J_WC+K1*J_BC])
println("lambda_wc is ",alpha_hat_mle[K1*J_WC+K1*J_BC+1])
println("lambda_bc is ",alpha_hat_mle[K1*J_WC+K1*J_BC+2])
println("gamma is ",alpha_hat_mle[K1*J_WC+K1*J_BC+3])


