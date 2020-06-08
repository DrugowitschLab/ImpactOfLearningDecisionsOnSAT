%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - code for plotting figure 5
%task data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

%load categorization task data
%this command loads DATA_id a structure with
%DATA_id =
%
%         Stimulus: Stimulus being presented from 1 (easy right) to 8 (easy
%         left)
%              RAT: rat number from 1 to 4
%          Session: session number for particular rat and task
%            Trial: trials in that session
%        ChoiceDir: choice direction 0 for right 1 for left
%          Outcome: outcome for that trial - 0 error, 1 rewarded, NaN is
%          reward ommision (correct but not rewarded because rat didn't
%          wait long enought)
%              OSD: Odor sampling duration in seconds
%     MovementTime: Movement time from odor port to reward port, in seconds
%
%      RewardDelay: Reward delay, from reward poke in, in seconds. It's Nan
%      if error trial
load(['..' filesep 'data' filesep 'DATA ID.mat'])


%allStim -> stimuli that are presented to the rats
allStim=[0 0 0 0 0.001 0.01 0.1 1]*0.1;


load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'adfcolbiaslapse_psychchron1_ident.mat'])
    

    
%Extraction of performance and reaction times from model
NumStim=length(allStim);
Perf_model=zeros(NumStim,1);
RTs=zeros(NumStim,1);

TE_MODEL.Stimulus=stimid;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;
TE_MODEL.OSD=t;

for iS=1:NumStim
    Perf_model(iS) = nanmean(resp(stimid==iS));
    RTs(iS)=nanmean(t(stimid==iS));
end

%make contrast data for model
[Pc2, RTS2, MIXES]=eighTOfour(Perf_model, RTs, allStim);

%t_model -> model reaction times
%p_model -> model proportion predictions both collapsed
t_model=sort(RTS2)*1000;
p_model=sort(Pc2,'descend');
    
 
%number of stimuli
NumStim=length(allStim);
%extract Perf - Performance; r - > number of left responses; n-> number of
%trials; RTs - > Reaction times; sem_RTs -> standard error of the mean for
%reaction times


[Perf, r, n, RTs, sem_RTs] = PerformanceRTsExtractor(NumStim,DATA_id);
%make contrast data
[Pc2, RTS2, MIXES, sem_RTS2, n2]=eighTOfour2(Perf, RTs, allStim,sem_RTs, n);

%se_perf -> standard error for performance data, considering binomial
%distribution and re-weighting as 4 rats = 4 separate experiments
sigma_perf=sqrt(4.*Pc2.*(1-Pc2)./n2);

plotIdentification2withModelNoFitting(MIXES,Pc2, sigma_perf,RTS2.*1000, sem_RTS2.*1000, p_model, t_model, sum(n))



%load categorization task data
%this command loads DATA_cat a structure with
%DATA_cat =
%
%         Stimulus: Stimulus being presented from 1 (easy right) to 8 (easy
%         left)
%              RAT: rat number from 1 to 4
%          Session: session number for particular rat and task
%            Trial: trials in that session
%        ChoiceDir: choice direction 0 for right 1 for left
%          Outcome: outcome for that trial - 0 error, 1 rewarded, NaN is
%          reward ommision (correct but not rewarded because rat didn't
%          wait long enought)
%              OSD: Odor sampling duration in seconds
%     MovementTime: Movement time from odor port to reward port, in seconds
%
%      RewardDelay: Reward delay, from reward poke in, in seconds. It's Nan
%      if error trial
load(['..' filesep 'data' filesep 'DATA CAT.mat'])

%allStim -> stimuli that are presented to the rats
allStim=[0 0.2 0.32 0.44 0.56 0.68 0.8 1];


load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'adfcolbiaslapse_psychchron1_categ'])
    
    
TE_MODEL.Stimulus=stimid;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;
TE_MODEL.OSD=t;

%Extraction of performance and reaction times from model
NumStim=length(allStim);
Perf_model=zeros(NumStim,1);
RTs=zeros(NumStim,1);

for iS=1:NumStim
    Perf_model(iS) = nanmean(resp(stimid==iS));
    RTs(iS)=nanmean(t(stimid==iS));
end

%make contrast data for model
[Pc2, RTS2, MIXES]=eighTOfour(Perf_model, RTs, allStim);

%t_model -> model reaction times
%p_model -> model proportion predictions both collapsed
t_model=sort(RTS2)*1000;
p_model=sort(Pc2,'descend');
    
    
%number of stimuli
NumStim=length(allStim);
%extract Perf - Performance; r - > number of left responses; n-> number of
%trials; RTs - > Reaction times; sem_RTs -> standard error of the mean for


[Perf, r, n, RTs, sem_RTs] = PerformanceRTsExtractor(NumStim,DATA_cat);
%make contrast data
[Pc2, RTS2, MIXES, sem_RTS2, n2]=eighTOfour2(Perf, RTs, allStim,sem_RTs, n);

%se_perf -> standard error for performance data, considering binomial
%distribution and re-weighting as 4 rats = 4 separate experiments
sigma_perf=sqrt(4.*Pc2.*(1-Pc2)./n2);


plotCategorization2withModelNoFitting(MIXES,Pc2, sigma_perf,RTS2.*1000, sem_RTS2.*1000, p_model, t_model, sum(n))

    
    %load categorization task data
%this command loads DATA_int a structure with
%DATA_int =
%
%         Stimulus (1-32): Stimulus being presented from 1,9,17,25 (easy
%         rights, decreasing concentration) to 8,16,24,32 (hard
%         left, decreasing concentrations)
%              RAT: rat number from 1 to 4
%          Session: session number for particular rat and task
%            Trial: trials in that session
%        ChoiceDir: choice direction 0 for right 1 for left
%          Outcome: outcome for that trial - 0 error, 1 rewarded, NaN is
%          reward ommision (correct but not rewarded because rat didn't
%          wait long enought)
%              OSD: Odor sampling duration in seconds
%     MovementTime: Movement time from odor port to reward port, in seconds
%
%      RewardDelay: Reward delay, from reward poke in, in seconds. It's Nan
%      if error trial
load(['..' filesep 'data' filesep 'DATA INT.mat'])


%function to extract the data, but now notice that there are 32 data points
[Perf, r, n, RTs, sem_RTs] = PerformanceRTsExtractor(32,DATA_int);
%make contrast data - note that this is kinda tricky, look at function
[r2, n2, RTs2, sem_RTs2]=collapsedInterleaved(r, n, RTs,sem_RTs);


Pc2=r2./n2;
%se_perf -> standard error for performance data, considering binomial
%distribution and re-weighting as 4 rats = 4 separate experiments
sigma_perf=sqrt(4.*Pc2.*(1-Pc2)./n2);

%function to plot interleaved data
load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'adfcolbiaslapse_psychchron1itlvd_itlvd.mat']);



TE_MODEL.Stimulus=stimid;
TE_MODEL.OSD=t;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;

TE_MODEL.Stimulus(find(stimid==5))=8;
TE_MODEL.Stimulus(find(stimid==6))=7;
TE_MODEL.Stimulus(find(stimid==7))=6;
TE_MODEL.Stimulus(find(stimid==8))=5;
TE_MODEL.Stimulus(find(stimid==13))=16;
TE_MODEL.Stimulus(find(stimid==14))=15;
TE_MODEL.Stimulus(find(stimid==15))=14;
TE_MODEL.Stimulus(find(stimid==16))=13;
TE_MODEL.Stimulus(find(stimid==21))=24;
TE_MODEL.Stimulus(find(stimid==22))=23;
TE_MODEL.Stimulus(find(stimid==23))=22;
TE_MODEL.Stimulus(find(stimid==24))=21;
TE_MODEL.Stimulus(find(stimid==29))=32;
TE_MODEL.Stimulus(find(stimid==30))=31;
TE_MODEL.Stimulus(find(stimid==31))=30;
TE_MODEL.Stimulus(find(stimid==32))=29;


%function to extract the data, but now notice that there are 32 data points
[Perf_model, r_model, n_model, RTs_model, sem_RTs_model] = PerformanceRTsExtractor(32,TE_MODEL);
%make contrast data - note that this is kinda tricky, look at function
[r2_model, n2_model, RTs2_model, sem_RTs2_model]=collapsedInterleaved(r_model, n_model, RTs_model,sem_RTs_model);


plotInterleavedWithModel(RTs2, sem_RTs2, r2, n2, sigma_perf, RTs2_model, r2_model./n2_model);