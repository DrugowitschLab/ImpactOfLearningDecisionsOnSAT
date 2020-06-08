function [Perf, r, n, RTs, sem_RTs] = PerformanceRTsExtractor(NumStim,TE_DATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - PerformanceRTsExtractor function
%This function is used to extract the relevant stats for plotting from a
%data structure
%INPUT VARIABLES
%NumStim - number of stimuli presented
%TE_DATA - data structure with the information for relevant stats
%OUTPUT VARIABLES
%Perf - Performance for that particular stimuli as fraction of left choices
%r - number of left responses
%n - number of trials for a given stimuli
%RTs - Reaction times for a given stimuli
%sem_RTs - standard error of the mean
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Perf=zeros(NumStim,1);
r=zeros(NumStim,1);
n=zeros(NumStim,1);
RTs=zeros(NumStim,1);
sem_RTs=zeros(NumStim,1);
for iS=1:NumStim
    Perf(iS) = nanmean(TE_DATA.ChoiceDir(TE_DATA.Stimulus==iS));
    r(iS) = nansum(TE_DATA.ChoiceDir(TE_DATA.Stimulus==iS));
    n(iS) = length(TE_DATA.ChoiceDir(TE_DATA.Stimulus==iS));
    RTs(iS)=nanmean(TE_DATA.OSD(TE_DATA.Stimulus==iS));
    sem_RTs(iS)=nanstd(TE_DATA.OSD(TE_DATA.Stimulus==iS))./sqrt(length(TE_DATA.OSD(TE_DATA.Stimulus==iS))./(4));
end