%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - code for plotting figure 6
%task data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

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

%load Bayes model data

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

plotUpdatingPsychIdifferencePointWithModel2(DATA_cat,1, TE_MODEL, 1);


%%
%run this part separately for identification

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

%load Bayes model data

load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'adfcolbiaslapse_psychchron1_ident'])

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

plotUpdatingPsychIdifferencePointWithModel2(DATA_cat,1, TE_MODEL, 2);

