# simulates model performance and collects statistics


const ident_cs = [1e-1 0; 1e-2 0; 1e-3 0; 1e-4 0; 0 1e-4; 0 1e-3; 0 1e-2; 0 1e-1]
const categ_cs = [1.0 0.0; 0.8 0.2; 0.68 0.32; 0.56 0.44; 0.44 0.56; 0.32 0.68; 0.2 0.8; 0.0 1.0] * 0.1
const itlvd_cs = vcat(categ_cs, categ_cs * 0.1, categ_cs * 0.01, categ_cs * 0.001);
# loactions on x-axis (for plotting / psych curves)
const ident_xs = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
const categ_xs = [0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 1.0]
const itlvd_xs = repeat(categ_xs, outer=[4])


# constructs evidence sequences out of base blocks by concatenating shuffled
# versions of the base block until the desired number of trials is reached.
function constructevidence(base_cs, base_ids, trials)
    base_n = size(base_cs, 1)
    @assert length(base_ids) == base_n
    blocks = ceil(Integer, trials / base_n)
    # first fill larger blocks, then truncate
    cs = similar(base_cs, blocks * base_n, size(base_cs, 2))
    ids = similar(base_ids, blocks * base_n)
    i = collect(1:base_n)
    for block_i = 1:blocks
        shuffle!(i)
        base_i = (block_i - 1) * base_n + 1
        cs[base_i:(base_i + base_n - 1), :] = base_cs[i, :]
        ids[base_i:(base_i + base_n - 1), :] = base_ids[i]
    end
    cs[1:trials, :], ids[1:trials]
end


# create full evidence block by first repeating each base block blockreps number
# of times, constructing a base number of trials. These are then concatenated
# with different shuffle orders, until 'trials' number of trials are reached
# (using constructevidence()). This is done for the two conditions separately,
# after which they are concatenated. Shuffling subblocks ensures an evenly
# spread out distribution of trials even for small numbers of trials.
function catevidence(ident_cs, categ_cs, blockreps, trials)
    ident_evidence, ident_ids = constructevidence(
        repeat(ident_cs, outer=[blockreps, 1]), 
        repeat(collect(1:size(ident_cs, 1)), outer=[blockreps, 1]),
        trials)
    categ_evidence, categ_ids = constructevidence(
        repeat(categ_cs, outer=[blockreps, 1]),
        repeat(collect(1:size(categ_cs, 1)), outer=[blockreps, 1]),
        trials)
    return cat(ident_evidence, categ_evidence, dims=1), cat(ident_ids, categ_ids, dims=1)
end
# the same for the interleaved condition, without final concatenation
function catitlvdevidence(itlvd_cs, blockreps, trials)
    itlvd_evidence, itlvd_ids = constructevidence(
        repeat(itlvd_cs, outer=[blockreps, 1]),
        repeat(collect(1:size(itlvd_cs, 1)), outer=[blockreps, 1]),
        trials)
    return itlvd_evidence, itlvd_ids
end


# runs simulations with given evidence / stimulus id sequence, after
# which it splits results into two equally-sized blocks, each of which
# is returned as a DataFrame
function simmodel(m::ModelBase, ϕ::Vector{Float64}, evidence, ids)
    trials = length(ids)
    @assert size(evidence, 1) == trials
    @assert mod(trials, 2) == 0
    trials = div(trials, 2)

    resps, ts, gs = sim(m, ϕ, evidence)
    
    # determine correct replies by which-ever evidence is larger
    corrresps = convert(Vector{Int}, evidence[:,2] .> evidence[:,1])
    corr = resps .== corrresps

    # return results in two dataframes
    r1, r2 = 1:trials, (trials+1):(2*trials)
    stimmag = copy(ids)
    stimmag[stimmag .< 5] .= 9 .- stimmag[stimmag .< 5]
    previds = cat(ids[end], ids[1:end-1], dims=1)    # prev. before first is random
    prevcorr = cat(corr[end], corr[1:end-1], dims=1)
    return DataFrame(trial    = collect(r1), 
              stimid   = ids[r1],
              previd   = previds[r1],
              stimmag  = stimmag[r1],
              corrresp = corrresps[r1],
              resp     = resps[r1],
              t        = ts[r1],
              conf     = gs[r1],
              corr     = corr[r1],
              prevcorr = prevcorr[r1],
              prevrew  = prevcorr[r1]),
    DataFrame(trial    = collect(r1),
              stimid   = ids[r2],
              previd   = previds[r2],
              stimmag  = stimmag[r2],
              corrresp = corrresps[r2],
              resp     = resps[r2],
              t        = ts[r2],
              conf     = gs[r2],
              corr     = corr[r2],
              prevcorr = prevcorr[r2],
              prevrew  = prevcorr[r2])
end
# same as simmodel() but for interleaved trials, without the block splitting
function simitlvdmodel(m::ModelBase, ϕ::Vector{Float64}, evidence, ids)
    trials = length(ids)
    @assert size(evidence, 1) == trials

    resps, ts, gs = sim(m, ϕ, evidence)

    # determine correct replies by which-ever evidence is larger
    corrresps = convert(Vector{Int}, evidence[:,2] .> evidence[:,1])
    corr = resps .== corrresps

    # collect results in DataFrame
    stimmag = Vector{Int}(undef, trials)
    mixw = Vector{Int}(undef, trials)
    mixmag = Vector{Int}(undef, trials)
    concentr = Vector{Int}(undef, trials)
    for i = 1:4  # iterate over concentrations
        righttrials = (ids .>= 8i-3) .& (ids .<= 8i)
        lefttrials = (ids .>= 8i-7) .& (ids .<= 8i-4)
        alltrials = righttrials .| lefttrials
        stimmag[righttrials] .= ids[righttrials]
        stimmag[lefttrials] .= (16i - 7) .- ids[lefttrials]
        mixw[alltrials] .= ids[alltrials] .- 8(i-1)       # to 1-8 range
        mixmag[alltrials] .= stimmag[alltrials] .- (8i - 4)  # to 1-4 range
        concentr[alltrials] .= 5-i
    end

    return DataFrame(
        trial    = collect(1:trials),
        stimid   = ids,
        previd   = [ids[end]; ids[1:end-1]],
        stimmag  = stimmag,
        mixw     = mixw,
        mixmag   = mixmag,
        concentr = concentr,
        corrresp = corrresps,
        resp     = resps,
        t        = ts,
        conf     = gs,
        corr     = corr,
        prevcorr = [corr[end]; corr[1:end-1]],
        prevrew  = [corr[end]; corr[1:end-1]])
end
# same as simmodel(), but providing additional statistics
function simmodelwithstats(m::ModelBase, ϕ::Vector{Float64}, evidence, ids)
    trials = length(ids)
    @assert size(evidence, 1) == trials
    @assert mod(trials, 2) == 0
    trials = div(trials, 2)

    resps, ts, gs, dts, ws, biases, xs, inputs = simwithstats(m, ϕ, evidence)
    
    # determine correct replies by which-ever evidence is larger
    corrresps = convert(Vector{Int}, evidence[:,2] .> evidence[:,1])
    corr = resps .== corrresps

    # return results in two dataframes
    r1, r2 = 1:trials, (trials+1):(2*trials)
    stimmag = copy(ids)
    stimmag[stimmag .< 5] = 9 - stimmag[stimmag .< 5]
    previds = cat(ids[end], ids[1:end-1], dims=1)    # prev. before first is random
    prevcorr = cat(corr[end], corr[1:end-1], dims=1)
    mat2arrarr(A) = [A[i,:] for i in 1:size(A,1)]
    return DataFrame(trial    = collect(r1), 
              stimid   = ids[r1],
              previd   = previds[r1],
              stimmag  = stimmag[r1],
              corrresp = corrresps[r1],
              resp     = resps[r1],
              t        = ts[r1],
              conf     = gs[r1],
              corr     = corr[r1],
              prevcorr = prevcorr[r1],
              prevrew  = prevcorr[r1],
              dt       = dts[r1],
              w        = mat2arrarr(ws[r1,:]),
              bias     = biases[r1],
              x        = mat2arrarr(xs[r1,:]),
              input    = mat2arrarr(inputs[r1,:]) ),
    DataFrame(trial    = collect(r1),
              stimid   = ids[r2],
              previd   = previds[r2],
              stimmag  = stimmag[r2],
              corrresp = corrresps[r2],
              resp     = resps[r2],
              t        = ts[r2],
              conf     = gs[r2],
              corr     = corr[r2],
              prevcorr = prevcorr[r2],
              prevrew  = prevcorr[r2],
              dt       = dts[r2],
              w        = mat2arrarr(ws[r2,:]),
              bias     = biases[r2],
              x        = mat2arrarr(xs[r2,:]),
              input    = mat2arrarr(inputs[r2,:]) )
end


# returns two DataFrames. The first provides psychometric / chronometric
# statistics over all stimulus id's. The second does the same over stimulus
# magnitudes. If datasets > 1, then the SEM is computed as average over the
# given number of datasets, rather than over all trials.
function getpsychchron(result_df, datasets::Int=1)
    id_df = by(result_df, :stimid,
        df -> DataFrame(tμ = mean(df[:t]), tσ = √(var(df[:t])),
                        tsem = √(datasets * var(df[:t]) / length(df[:t])),
                        tcorrμ = mean(df[df[:corr], :t]), tcorrσ = √(var(df[df[:corr], :t])),
                        tincorrμ = mean(df[.!df[:corr], :t]), tincorrσ = √(var(df[.!df[:corr], :t])),
                        pcorr = mean(df[:corr]), pr = mean(df[:resp] .== 1),
                        confμ = mean(df[:conf]), confσ = √(var(df[:conf]))))
    mag_df = by(result_df, :stimmag,
        df -> DataFrame(tμ = mean(df[:t]), tσ = √(var(df[:t])), 
                        tsem = √(datasets * var(df[:t]) / length(df[:t])),
                        tcorrμ = mean(df[df[:corr], :t]), tcorrσ = √(var(df[df[:corr], :t])),
                        tincorrμ = mean(df[.!df[:corr], :t]), tincorrσ = √(var(df[.!df[:corr], :t])),
                        pcorr = mean(df[:corr]),
                        confμ = mean(df[:conf]), confσ = √(var(df[:conf]))))
    id_df, mag_df
end


# returns two DataFrames. The first groups trials by the current/prev. stimulus
# id tuple and provides p("right") and average times after correct and incorrect
# choices. The second fits a psychometric curve for reach prev. stimulus id, and
# returns mean and standard deviation of this fit for each prev. stimulus id.
# for the latter, idx[stimid] is used to map (current) stimulus ids onto
# the horizontal axis of the psychometric curve.
function getseqstats(result_df, idx)
    seq_df = by(result_df, [:stimid, :previd],
        df -> DataFrame(prcorr = mean(df[df[:prevcorr], :resp] .== 1),
                        tcorrμ = mean(df[df[:prevcorr], :t]), tcorrσ = √(var(df[df[:corr], :t])),
                        princorr = mean(df[.!df[:prevcorr], :resp] .== 1),
                        tincorrμ = mean(df[.!df[:prevcorr], :t]), tincorrσ = √(var(df[.!df[:corr], :t])),
                        x = idx[df[1, :stimid]]))
    # fit psychometric curves
    ids = unique(seq_df[:stimid])
    xcenter = (maximum(idx) + minimum(idx)) / 2
    seq_bias_df = DataFrame(previd = similar(ids, 0),
                            μcorr = similar(idx, 0), σcorr = similar(idx, 0), biascorr = similar(idx, 0),
                            μincorr = similar(idx, 0), σincorr = similar(idx, 0), biasincorr = similar(idx, 0))
    for i in ids
        μcorr, σcorr = psychprobitfit(
            Vector(seq_df[seq_df[:previd] .== i, :x]),
            Vector(seq_df[seq_df[:previd] .== i, :prcorr]))
        biascorr = Φ((xcenter - μcorr) / σcorr) - 0.5
        μincorr, σincorr = psychprobitfit(
            Vector(seq_df[seq_df[:previd] .== i, :x]),
            Vector(seq_df[seq_df[:previd] .== i, :princorr]))
        biasincorr = Φ((xcenter - μincorr) / σincorr) - 0.5
        push!(seq_bias_df, [i, μcorr, σcorr, biascorr,
                               μincorr, σincorr, biasincorr])
    end
    seq_df, seq_bias_df
end


# returns another dataframe of moving average p(correct) and RT, window size w
function getmovingavg(result_df, w)
    trials = size(result_df, 1)
    DataFrame(trial=collect((1:(trials-w+1))+div(w,2)), 
              pc=movingavg(result_df[:corr], w), t=movingavg(result_df[:t], w))
end

