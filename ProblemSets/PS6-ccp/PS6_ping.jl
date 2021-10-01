using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV



@time myfunc()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    #show(df, allcols=true)

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

    flogit = glm(@formula(Y~Odometer + Odometer^2 + RouteUsage + RouteUsage^2+ Branded + time + time^2 +
                Odometer*RouteUsage + Odometer*RouteUsage^2 + Odometer*Branded + Odometer*time +
            Odometer*time^2 + Odometer^2*RouteUsage + Odometer^2*RouteUsage^2 + Odometer^2*Branded + 
            Odometer^2*time + Odometer^2*time^2 + RouteUsage*Branded + RouteUsage*time + RouteUsage*time^2 +
            RouteUsage^2*Branded + RouteUsage^2*time + RouteUsage^2*time^2 + Branded*time + Branded*time^2 +
            Odometer*RouteUsage * Branded + Odometer*RouteUsage * time + Odometer*RouteUsage * time^2 + 
            Odometer*RouteUsage^2 * Branded + Odometer*RouteUsage^2 * time + Odometer*RouteUsage^2 * time^2 +
            Odometer*Branded * time + Odometer*Branded * time^2 + Odometer^2*RouteUsage * Branded +
            Odometer^2*RouteUsage * time + Odometer^2*RouteUsage * time^2 + Odometer^2*RouteUsage^2 * Branded + 
            Odometer^2*RouteUsage^2 * time + Odometer^2*RouteUsage^2 * time^2 + Odometer^2*Branded * time +
            Odometer^2*Branded * time^2 + RouteUsage*Branded * time + RouteUsage*Branded * time^2 + 
            RouteUsage^2*Branded * time + RouteUsage^2*Branded * time^2 +
            Odometer*RouteUsage * Branded * time + Odometer*RouteUsage * Branded * time^2 +
            Odometer*RouteUsage^2 * Branded * time + Odometer*RouteUsage^2 * Branded * time^2 +
            Odometer^2*RouteUsage * Branded * time + Odometer^2*RouteUsage * Branded * time^2 + 
            Odometer^2*RouteUsage^2 * Branded * time + Odometer^2*RouteUsage^2 * Branded * time^2),df_long, Binomial(), LogitLink())
    println(flogit)

    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    include("create_grids.jl") 
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

    # [ convert other data frame columns to matrices]
    O = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    X = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Z = Matrix(df[:,[:Zst]])
    B = Matrix(df[:,[:Branded]])

    zval,zbin,xval,xbin,xtran = create_grids()
    T = 20
    FV = zeros((size(xtran,1),2,T+1))
    beta = 0.9
    theta = [0.1,0.2,0.3]
    sdf = DataFrame(Odometer = kron(ones(zbin),xval), RouteUsage = kron(ones(xbin),zval),Branded = zeros(xbin*size(zval,1)),time = zeros(xbin*size(zval,1)))


    function future_val(sdf,flogit,X, Z, B, xtran)
        for t = 2:T
            for i =0:1
                sdf.time = t * ones(size(sdf,1))
                sdf.Branded = i * ones(size(sdf,1))
                p0 = predict(flogit,sdf)
                FV[:,i+1,t] = -beta*log.(p0)
            end
        end

        FVT1 = zeros(size(X,1),T)
        for t = 2:T
            for i = 1:size(X,1)
                t = (T+1)-t
                z = Z[i]
                x = X[i,t]
                b = B[i]
                row1 = x + (z-1)*xbin
                row0 = 1+(z-1)*xbin
                FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,b+1,t+1]
            end
        end 

        return FVT1'[:]

    end

    fvt1 = future_val(sdf,flogit,X, Z, B, xtran)
    df_long = @transform(df_long,fv = fvt1)
    theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded),df_long, Binomial(), LogitLink(),offset=df_long.fv)
    println(theta_hat_ccp_glm)

end