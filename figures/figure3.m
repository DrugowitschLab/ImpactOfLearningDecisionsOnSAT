%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - figure 3
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

%load Bayes model data for categorization
load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'optcollapse_psychchron1categ_categ']);

TE_MODEL.Stimulus=stimid;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;
TE_MODEL.OSD=t;

%Extraction of performance and reaction times from model
[t_model_cat_cat, p_model_cat_cat]=PerformanceRTsExtractorModel(TE_MODEL, allStim);

%number of stimuli
NumStim=length(allStim);
%extract Perf - Performance; r - > number of left responses; n-> number of
%trials; RTs - > Reaction times; sem_RTs -> standard error of the mean for
%individualrats
%DATA_cat=individualData(DATA_cat, 4);

[Perf, r, n, RTs, sem_RTs] = PerformanceRTsExtractor(NumStim,DATA_cat);
%make contrast data
[Pc2, RTS2, MIXES, sem_RTS2, n2]=eighTOfour2(Perf, RTs, allStim,sem_RTs, n);

%se_perf -> standard error for performance data, considering binomial
%distribution and re-weighting as 4 rats = 4 separate experiments
sigma_perf=sqrt(4.*Pc2.*(1-Pc2)./n2);

load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'optcollapse_psychchron1ident_categ']);

TE_MODEL.Stimulus=stimid;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;
TE_MODEL.OSD=t;

%Extraction of performance and reaction times from model
[t_model_id_cat, p_model_id_cat]=PerformanceRTsExtractorModel(TE_MODEL, allStim);


%Plotting with model
plotCategorization2withModelFig3(MIXES,Pc2, sigma_perf,RTS2.*1000, sem_RTS2.*1000, p_model_cat_cat, t_model_cat_cat, sum(n),p_model_id_cat, t_model_id_cat)


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
load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'optcollapse_psychchron1categ_ident.mat'])
    
    

%Extraction of performance and reaction times from model
NumStim=length(allStim);
Perf_model=zeros(NumStim,1);
RTs=zeros(NumStim,1);
    
TE_MODEL.Stimulus=stimid;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;
TE_MODEL.OSD=t;

%Extraction of performance and reaction times from model
[t_model_cat_id, p_model_cat_id]=PerformanceRTsExtractorModel(TE_MODEL, allStim);
    
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

%load Bayes model data
load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'optcollapse_psychchron1ident_ident.mat'])
TE_MODEL.Stimulus=stimid;
TE_MODEL.ChoiceDir=resp;
TE_MODEL.Outcome=corr;
TE_MODEL.OSD=t;

%Extraction of performance and reaction times from model
[t_model_id_id, p_model_id_id]=PerformanceRTsExtractorModel(TE_MODEL, allStim);

plotIdentification2withModelFig3(MIXES,Pc2, sigma_perf,RTS2.*1000, sem_RTS2.*1000, p_model_id_id, t_model_id_id, sum(n), p_model_cat_id, t_model_cat_id)







