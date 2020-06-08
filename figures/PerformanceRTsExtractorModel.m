function [t_model, p_model] = PerformanceRTsExtractorModel(TE_MODEL, allStim)

NumStim=length(allStim);

Perf_model=zeros(NumStim,1);
RTs=zeros(NumStim,1);

for iS=1:NumStim
    Perf_model(iS) = nanmean(TE_MODEL.ChoiceDir(TE_MODEL.Stimulus==iS));
    RTs(iS)=nanmean(TE_MODEL.OSD(TE_MODEL.Stimulus==iS));
end

%make contrast data for model
[Pc2, RTS2, MIXES]=eighTOfour(Perf_model, RTs, allStim);

%t_model -> model reaction times
%p_model -> model proportion predictions both collapsed
t_model=sort(RTS2)*1000;


p_model=sort(Pc2,'descend');