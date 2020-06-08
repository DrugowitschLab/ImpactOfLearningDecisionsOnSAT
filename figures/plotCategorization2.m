function p=plotCategorization2(MIX2,Pc2, semPerf,t_data, sigma_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%function to plot Performance and Reaction times for categorization task
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
%fitting a general curve for model proportions - > exponential a la Palmer
%et al.
Pc= @(y,x) (1-y(1))./(1+exp(-2.*y(2).*x.^y(3)));
[y,R,J,CovB]=nlinfit(MIX2,Pc2,Pc,[0, 4.5, 0.5]);

x=min(MIX2):0.001:max(MIX2);

plot(x, Pc(y, x), 'r', 'LineWidth', 1.5)

hold off

xlabel('Concentration contrast','FontName','Arial','FontSize',12);
ylabel('Accuracy','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);
axis square
ylim([0.5 1]);
xlim([0 1]);

%reaction times
figure
errorbar(MIX2, t_data, sigma_data, 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)

hold on
%fuitting general curve for model reaction times -> hyperbolic tangent a la
%Palmer et al.
tT= @(y,x) y(1)./((y(2).*x).^y(3)).*tanh(y(1).*((y(2).*x).^y(3)))+y(4);
[y,~,~,~]=nlinfit(MIX2,t_data,tT,[0, 4.5, 0.5 0.2]);

%plot(MIX2, t_model, 'k', 'LineWidth', 1.5)
plot(x, tT(y, x), 'r', 'LineWidth', 1.5)
hold off


axis square
ylim([260 420]);
xlim([0 1]);
xlabel('Concentration contrast','FontName','Arial','FontSize',12);
ylabel('Reaction times (ms)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);