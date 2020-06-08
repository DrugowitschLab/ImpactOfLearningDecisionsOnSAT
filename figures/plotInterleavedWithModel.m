function p=plotInterleavedWithModel(t_data, sigma_data, r, n, semPerf, t_model, p_model)

%this function plots the data in two ways - > collapsed, as in the paper -
%Figures 1 and 2 -> extended logarithm, Figures 3 and 4.
Collapsed_mixes=[1 0.6 0.68-0.32 0.56-0.44];

Full_mixes=[Collapsed_mixes.*1 Collapsed_mixes.*0.1 Collapsed_mixes.*0.01 Collapsed_mixes.*0.001];

figure
for j=1:4:13
    i=0;
    hold on
    errorbar(Collapsed_mixes, r(i+j:i+j+3)./n(i+j:i+j+3), semPerf(i+j:i+j+3), 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)
    hold on
    plot(Collapsed_mixes, p_model(i+j:i+j+3), 'r', 'LineWidth', 1.5)
    hold off
end

xlabel('Mixture contrast','FontName','Arial','FontSize',12);
ylabel('Accuracy','FontName','Arial','FontSize',12);

set(gca,'FontName','Arial','FontSize',12);
axis square;
ylim([0.5 1]);
xlim([0 1]);

figure
for j=1:4:13
    i=0;
    hold on
    errorbar(Collapsed_mixes, t_data(i+j:i+j+3), sigma_data(i+j:i+j+3), 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)
    hold on
    plot(Collapsed_mixes, t_model(i+j:i+j+3), 'r', 'LineWidth', 1.5)
    hold off
end

xlabel('Mixture contrast','FontName','Arial','FontSize',12);
ylabel('Reaction times (s)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);
axis square;
ylim([0.260 0.420]);
xlim([0 1]);

