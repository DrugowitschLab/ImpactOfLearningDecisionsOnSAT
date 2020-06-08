function p=plotUpdatingPsychIdifferencePointWithModel2(TE,condition, TE_MODEL, task)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%Change in choice bias regarding indifference percentage
%point with model
%TE - The structure with the data
%condition - the condition to calculate conditional curves 1= previous
%rewarded, 2 = previous non-rewarded
%TE_MODEL - The structure with the model data
%task to be plotted - task=1 (categorization) anything else
%(identification)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%start by plotting the data
plotUpdatingPsychIdifferencePoint2(TE, condition, task)

%from now on the code is exactly the same as plotUpdatingPsychBias.m
%function but adapted to only take the model data
TE=TE_MODEL;

%extract stimuli, number of trials and max trials from data
STIMULI=unique(TE.Stimulus);
NumStim=length(STIMULI);
MaxTrials=length(TE.Stimulus);

%iP condition=1 Outcome, 2 error
iP=condition;

%Mean phsyc function
for iS=1:NumStim
    NumberChoiceLeft(iS)= sum(TE.ChoiceDir(TE.Stimulus==iS));
    NumberTrials(iS)= length(TE.ChoiceDir(TE.Stimulus==iS));
    choiceLeft(iS) = nanmean(TE.ChoiceDir(TE.Stimulus==iS));
end


for iCOND=1:2 %1-previous Outcome; 2-Previous error
    for iPREV=1:NumStim
        stimulus= STIMULI(iPREV);
        switch iCOND
            case {1}
                posPREVSTIM=intersect(find(TE.Stimulus==stimulus),find(TE.Outcome==1)); %all rewarded prev stimuli
            case {2}
                posPREVSTIM=intersect(find(TE.Stimulus==stimulus),find(TE.Outcome~=1)); %all non rewarded prev stimuli
        end
        
        posSTIM=unique(min(posPREVSTIM+1,MaxTrials));
        
        for iCURR=1:NumStim
            stimulus = STIMULI(iCURR);      %current stimulus type
            posSTIM_current            = posSTIM(TE.Stimulus(posSTIM)==stimulus);   %curr
            left(iCOND,iPREV,iCURR)    = nansum(TE.ChoiceDir(posSTIM_current));
            total(iCOND,iPREV,iCURR)   = length(posSTIM_current);
            fracA(iCOND,iPREV,iCURR)   = left(iCOND,iPREV,iCURR)/total(iCOND,iPREV,iCURR);
            fracAse(iCOND,iPREV,iCURR) = binostat(1,fracA(iCOND,iPREV,iCURR))/sqrt(total(iCOND,iPREV,iCURR));
        end %iCURR
    end
end


%if task is categorization
if task==1
    MIXTURES=[0 0.2 0.32 0.44 0.56 0.68 0.8 1];
else
    %else
    MIXTURES=[-4 -3 -2 -1 1 2 3 4];
end

%here we fit the mean psychometric curve
x=1:8;
psych = @(y,x)(y(1)+(y(4)).*normcdf(x,y(2),y(3)));
[y,R,~,CovB]=nlinfit(x,NumberChoiceLeft(1,:)./(NumberTrials(1,:)),psych,[0, 4.5, 0.5, 1]);

%mean psych bias
bias_mean=y(2);

%indifference_percentage=psych(y, y(2));
%confidence interval for prediction 
[indifference_percentage,indifference_percentage_ci]=nlpredci(psych,bias_mean,y,R,'covar',CovB);


%initicialization for biases and confidence intervals (important for later
%plotting)
biases=zeros(length(MIXTURES),1);
biases_ci=zeros(length(MIXTURES),2);
biases_deltas=zeros(length(MIXTURES),1);
perc_at_mean_bias=zeros(length(MIXTURES),1);
perc_at_mean_bias_ci=zeros(length(MIXTURES),1);

for iS=1:length(MIXTURES)
    
    
    
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% this is one way of doing it: by forcing lapses and slope to be
    %%% always the same, but next version works better.
    %
    %      psych2 = @(b,x)(y(1)+(y(4)).*normcdf(x,b,y(3)));
    %      [z,R,J,CovB]=nlinfit(x,squeeze(fracA(iP,iS,:))',psych2,4.5);
    %      % ci_z=nlparci(z,R,'jacobian',J);
    %      ci_z = nlparci(z,R,'covar',CovB)
    %
    %      biases_ci(iS,:)=ci_z;
    %      z=[y(1) z(1) y(3) y(4)];
    %
    %      biases_deltas(iS,1)=abs(ci_z(1,1)-z(2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %fitting of conditional pscyh - lapses and bias can change, but slope
    %is shared with mean psych curve
    psych3 = @(b,x)(b(1)+(b(3)).*normcdf(x,b(2),y(3)));
    [z,R,~,CovB]=nlinfit(x,squeeze(fracA(iP,iS,:))',psych3,[0, 4.5, 1]);
    ci_z = nlparci(z,R,'covar',CovB);
    [perc_at_mean_bias(iS,1), perc_at_mean_bias_ci(iS,1)]=nlpredci(psych3,bias_mean,z',R,'covar',CovB);

    
    %these are the fitted parameters for the conditional psych curve:
    z=[z(1) z(2) y(3) z(3)];
    biases_ci(iS,:)=ci_z(2,:);
    biases_deltas(iS,1)=abs(ci_z(2,1)-z(2));
    
    %keep traick of the biases
    biases(iS,1)=z(2);
    %perc_at_mean_bias(iS,1)=psych_func(z,bias_mean);

    
end

%now starts the full change is choice bias updating plots
figure(task+2)
%change is choice bias as the difference between the new and mean
%normalized by the original mean bias
change_choice_bias=perc_at_mean_bias-indifference_percentage;
    
%uncertainty with the measurement - through partial derivatives
change_choice_bias_deltas=sqrt((perc_at_mean_bias_ci*5).^2+(indifference_percentage_ci*5).^2);

% 95% confidece interval of measured change
change_choice_bias_low=change_choice_bias-change_choice_bias_deltas./sqrt(4);
change_choice_bias_up=change_choice_bias+change_choice_bias_deltas./sqrt(4);

hold on
%plot change in choice bias and model estimated error
plot(MIXTURES, change_choice_bias, '-r', 'LineWidth', 2)
plot(MIXTURES, change_choice_bias_low, '--r', 'LineWidth', 1)
plot(MIXTURES, change_choice_bias_up, '--r', 'LineWidth', 1)
hold off


%now starts the full change is choice bias updating plots but collapsed
figure(task+4)
%collapsed change in choice bias:
collapsed_ccb=(flipud(change_choice_bias(5:8))-change_choice_bias(1:4))./2;
%uncertainty related to that change in choice bias - through derivatives
collapsed_change_choice_bias_deltas=sqrt((flipud(change_choice_bias_deltas(5:8)).^2+change_choice_bias_deltas(1:4).^2))/2;
% 95% confidece interval of measured change
collapsed_change_choice_bias__low_ci=collapsed_ccb-collapsed_change_choice_bias_deltas./sqrt(4);
collapsed_change_choice_bias__high_ci=collapsed_ccb+collapsed_change_choice_bias_deltas./sqrt(4);

%plotting:
if task==1
    STIM = ([1 0.6 0.68-0.32 0.56-0.44]);
else
    STIM=[4 3 2 1];
end

hold on
x=STIM;

plot(STIM, collapsed_ccb, '-r', 'LineWidth', 2)


%error bars
plot(STIM, collapsed_change_choice_bias__low_ci, '--r', 'LineWidth', 1)
plot(STIM, collapsed_change_choice_bias__high_ci, '--r', 'LineWidth', 1)

hold off
