function p=plotUpdatingPsychIdifferencePoint2(TE,condition, task)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%function to plot Change in choice bias regarding indifference percentage
%point
%TE - The structure with the data
%condition - the condition to calculate conditional curves 1= previous
%rewarded, 2 = previous non-rewarded
%task to be plotted - task=1 (categorization) anything else
%(identification)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%auxiliary variable for plotting
xaux=1:8;

%this for loop plots the datapoints for conditional psych curves
for iS=1:length(MIXTURES)
    figure(task)
    hold on
    if iS<5
        subplot(2,2,iS)
        plot((xaux-1)./7, squeeze(fracA(iP,iS,:)),'o', 'Color',[0 (1-0.125*iS) 0.125*iS]);
        axis square
    else
        subplot(2,2,9-iS)
        plot((xaux-1)./7, squeeze(fracA(iP,iS,:)),'o', 'Color',[0 (1-0.125*iS) 0.125*iS]);
    end
end
hold off

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
    
    %a new auxiliary variable to fit the curves
    xaux=1:0.01:8;
    
    
    %plotting to figure the conditional fitted curves
    figure(task)
    hold on
    if iS<5
        subplot(2,2,iS)
        plot((xaux-1)./7, psych_func(y,xaux), 'k')
        plot((xaux-1)./7, psych_func(z,xaux),'Color',[0 (1-0.125*iS) 0.125*iS]);
        axis square
        ylim([0 1])
        xlim([0 1])
    else
       
        subplot(2,2,9-iS)
        plot((xaux-1)./7, psych_func(y,xaux), 'k')
        plot((xaux-1)./7, psych_func(z,xaux),'Color',[0 (1-0.125*iS) 0.125*iS]);
        ylim([0 1])
        xlim([0 1])
    end
    
    %keep traick of the biases
    biases(iS,1)=z(2);
 
end
hold off

%now starts the full change is choice bias updating plots
figure(task+2)
%change is choice bias as the difference between the new and mean
%normalized by the original mean bias
change_choice_bias=perc_at_mean_bias-indifference_percentage;


%uncertainty with the measurement - through partial derivatives
change_choice_bias_deltas=sqrt(perc_at_mean_bias_ci.^2+indifference_percentage_ci.^2);

% 95% confidece interval of measured change
change_choice_bias_low=change_choice_bias-change_choice_bias_deltas./sqrt(4);
change_choice_bias_up=change_choice_bias+change_choice_bias_deltas./sqrt(4);

%plot change in choice bias and error bars
plot(MIXTURES, change_choice_bias, 'o','MarkerEdgeColor','r', 'MarkerFaceColor','r', 'MarkerSize',8)
hold on
for i=1:8
    plot([MIXTURES(i) MIXTURES(i)], [change_choice_bias_low(i,1) change_choice_bias_up(i,1)], 'r');
end
hold off

axis square
if iP==1
    ylim([-0.35 0.35])
else
    ylim([-0.6 0.6])
end
xlabel('Previous odor A (fraction)','FontName','Arial','FontSize',12);
ylabel('Change in choice bias (fraction)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);


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
plot(STIM,collapsed_ccb, 'o','MarkerEdgeColor','r', 'MarkerFaceColor','r', 'MarkerSize',8);
%error bars
hold on
for i=1:4
    plot([STIM(i) STIM(i)], [collapsed_change_choice_bias__low_ci(i,1) collapsed_change_choice_bias__high_ci(i,1)], 'r');
end
hold off
axis square
if iP==1
    ylim([-0.05 0.30])
else
    ylim([-0.55 0.2])
end
if task==1
    xlim([0 1.2])
else
    xlim([0.5 4.5])
end
xlabel('Previous odor contrast (fraction)','FontName','Arial','FontSize',12);
ylabel('Collapsed change in choice bias (fraction)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);

