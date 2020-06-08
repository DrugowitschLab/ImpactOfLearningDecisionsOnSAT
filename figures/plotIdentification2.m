function p=plotIdentification2(MIX2,Pc2, semPerf,t_data, sigma_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%function to plot Performance and Reaction times for identication task
%MIX2 - Contrasts
%Pc2 - Proportions for each contrast
%semPerf - Standard error for performance
%t_data - Reaction times
%sigma_data - Standard error for reaction times
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
%performance
errorbar(MIX2, Pc2, semPerf, 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)
hold on
%fitting a general curve for model proportions - > exponential ? l? Palmer
%et al.
Pc= @(y,x) (1-y(1))./(1+exp(-2.*y(2).*x.^y(3)));
[y,R,J,CovB]=nlinfit(MIX2,Pc2,Pc,[0, 4.5, 0.5]);

%confidence intervals
%generate x for plotting
x=min(MIX2):0.00001:max(MIX2);
%predicted standard error for the model
%se_perf_model=sqrt(4.*Pc(y, x).*(1-Pc(y, x))./(Number_trials./8));
%confidence intervals - bottom and upper limit
%bot_lim=Pc(y, x)-se_perf_model;
%up_lim=Pc(y, x)+se_perf_model;
%plotting model
%fill([x';flipud(x')],[bot_lim';flipud(up_lim')],[230/255 230/255 230/255],'linestyle','none');
%plot(MIX2, p_model, 'k', 'LineWidth', 1.5)
plot(x, Pc(y, x), 'r', 'LineWidth', 1.5)

hold off

xlabel('Concentration contrast','FontName','Arial','FontSize',12);
ylabel('Accuracy','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12,'XScale','log');
axis square
ylim([0.5 1]);
xlim([0 0.1]);

%reaction times
figure
errorbar(MIX2, t_data, sigma_data, 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)

%fuitting general curve for model reaction times -> hyperbolic tangent ? l?
%Palmer et al.
tT= @(y,x) y(1)./((y(2).*x).^y(3)).*tanh(y(1).*((y(2).*x).^y(3)))+y(4);
[y,~,~,~]=nlinfit(MIX2,t_data,tT,[16, 3.5, 0.675, 285]);

hold on
plot(x, tT(y, x), 'r', 'LineWidth', 1.5)
hold off

axis square
ylim([260 420]);
xlim([0 0.1]);
xlabel('Concentration contrast','FontName','Arial','FontSize',12);
ylabel('Reaction times (ms)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12,'XScale','log');