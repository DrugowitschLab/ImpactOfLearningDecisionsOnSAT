function p=plotIdentification2withModelNoFitting(MIX2,Pc2, semPerf,t_data, sigma_data, p_model, t_model, Number_trials)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%function to plot Performance and Reaction times for categorization task
%but also with model
%MIX2 - Contrasts
%Pc2 - Proportions for each contrast
%semPerf - Standard error for performance
%t_data - Reaction times
%sigma_data - Standard error for reaction times
%p_model - proportion predictions for model
%t_model - reaction tims for model
%Number_trials - number of trials in whole data. Important to estimate the
%confidence intervals for model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Performance
figure
xlabel('Concentration contrast','FontName','Arial','FontSize',12);
ylabel('Accuracy','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12,'XScale','log');
hold on

%plotting model
plot(MIX2, p_model, 'r', 'LineWidth', 1.5)
%plotting data
errorbar(MIX2, Pc2, semPerf, 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)
hold off
axis square
ylim([0.5 1]);
xlim([0 0.1]);


%Reaction times
figure
axis square
ylim([260 420]);
xlim([0 0.1]);
hold on

plot(MIX2, t_model, 'r', 'LineWidth', 1.5)
%plotting data
errorbar(MIX2, t_data, sigma_data, 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)
hold off
xlabel('Concentration contrast','FontName','Arial','FontSize',12);
ylabel('Reaction times (ms)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12,'XScale','log');