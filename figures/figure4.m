%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - figure 4
%interleaved task data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

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
%make contrast data
[r2, n2, RTs2, sem_RTs2]=collapsedInterleaved(r, n, RTs,sem_RTs);


Pc2=r2./n2;
%se_perf -> standard error for performance data, considering binomial
%distribution and re-weighting as 4 rats = 4 separate experiments
sigma_perf=sqrt(4.*Pc2.*(1-Pc2)./n2);

%function to plot interleaved data
plotInterleaved(RTs2, sem_RTs2, r2, n2, sigma_perf);
