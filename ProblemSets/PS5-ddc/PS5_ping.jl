using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM


# read in function to create state transitions for dynamic model
#include("create_grids.jl")
include("create_grids.jl") 


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# load in the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)



# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

res = glm(@formula(Y~Branded+Odometer),df_long, Binomial(), LogitLink())
println(res)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# [ Load in the data ]
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
#show(df, allcols=true)

Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

# [ convert other data frame columns to matrices]
O = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
X = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
Z = Matrix(df[:,[:Zst]])
B = Matrix(df[:,[:Branded]])


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: generate state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#discretize the mileage x1 transitions into 0,125-mile bins (i.e. 0.125 units of x1t ). 
#specify x2 as a discrete uniform distribution ranging from 0.25 to 1.25 with 0.01 unit increments.
zval,zbin,xval,xbin,xtran = create_grids()


T = 20
FV = zeros((size(xtran,1),2,T+1))
beta = 0.9
theta = [0.1,0.2,0.3]

theta = ones(3)
for t = 1:T
    for b =0:1
        for z = 1:zbin
            for x = 1:xbin
                t = (T+1)-t
                row = x + (z-1)*xbin
                #not replace the engine
                v1t = theta[1] +theta[2]*xval[x]+theta[3]*b+ (xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]) 
                #replace the engine
                v0t = xtran[1+(z-1)*xbin,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
                FV[row,b+1,t] = beta*log(exp(v0t)+exp(v1t))
            end  
        end
    end
end


loglik = 0
for t = 1:T
    for i = 1:size(X,1)
        t = (T+1)-t
        z = Z[i]
        x = X[i,t]
        b = B[i]
        row1 = x + (z-1)*xbin
        row0 = 1+(z-1)*xbin
        v_diff  = theta[1] +theta[2]*O[i,t]+theta[3]*b + (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,b+1,t+1]
        if Y[i,t] == 1
            loglik += log(exp(v_diff)/(1+exp(v_diff)))*Y[i,t]
        elseif Y[i,t] == 0
            loglik += log(1/(1+exp(v_diff)))*Y[i,t]
        end
    end
end
loglik = -loglik



@views @inbounds function myfun()
    
    
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

    # [ convert other data frame columns to matrices]
    O = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    X = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Z = Matrix(df[:,[:Zst]])
    B = Matrix(df[:,[:Branded]])
    
    #discretize the mileage x1 transitions into 0,125-mile bins (i.e. 0.125 units of x1t ). 
    #specify x2 as a discrete uniform distribution ranging from 0.25 to 1.25 with 0.01 unit increments.
    zval,zbin,xval,xbin,xtran = create_grids()
    
    T = 20
    FV = zeros((size(xtran,1),2,T+1))
    beta = 0.9
    
    function ddc(theta,Y,O,X,Z,B)
        #(c)
        for t = 1:T
            for b =0:1
                for z = 1:zbin
                    for x = 1:xbin
                        t = (T+1)-t
                        row = x + (z-1)*xbin
                        #not replace the engine
                        v1t = theta[1] +theta[2]*xval[x]+theta[3]*b+ (xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]) 
                        #replace the engine
                        v0t = xtran[1+(z-1)*xbin,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
                        FV[row,b+1,t] = beta*log(exp(v0t)+exp(v1t))
                    end  
                end
            end
        end
        #(d)
        loglik = 0
        for t = 1:T
            for i = 1:size(X,1)
                t = (T+1)-t
                z = Z[i]
                x = X[i,t]
                b = B[i]
                row1 = x + (z-1)*xbin
                row0 = 1+(z-1)*xbin
                v_diff  = theta[1] +theta[2]*O[i,t]+theta[3]*b + (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,b+1,t+1]
                if Y[i,t] == 1
                    loglik += log(exp(v_diff)/(1+exp(v_diff)))*Y[i,t]
                elseif Y[i,t] == 0
                    loglik += log(1/(1+exp(v_diff)))*Y[i,t]
                end
            end
        end
        loglik = -loglik
        return loglik 
    end
    
    theta0 = rand(3)
    obj = optimize(theta->ddc(theta, Y,O,X,Z,B), theta0, LBFGS(), Optim.Options(g_tol=1e-5, iterations=10, show_trace=true, show_every=50))
    theta = obj.minimizer
    println("theta is ",theta)
    
    
end


myfun()
