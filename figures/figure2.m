%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - code for plotting Fig. 2
%categorization and identification task data
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

%number of stimuli
NumStim=length(allStim);

%Extract quantities to plot
[Perf, r, n, RTs, sem_RTs] = PerformanceRTsExtractor(NumStim,DATA_cat);
%make contrast data
[Pc2, RTS2, MIXES, sem_RTS2, n2]=eighTOfour2(Perf, RTs, allStim,sem_RTs, n);

%se_perf -> standard error for performance data, considering binomial
%distribution and re-weighting as 4 rats = 4 separate experiments
sigma_perf=sqrt(4.*Pc2.*(1-Pc2)./n2);

%no model, just data
plotCategorization2(MIXES,Pc2, sigma_perf,RTS2.*1000, sem_RTS2.*1000)


%load identification task data
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

%no model, just data
plotIdentification2(MIXES,Pc2, sigma_perf,RTS2.*1000, sem_RTS2.*1000)
