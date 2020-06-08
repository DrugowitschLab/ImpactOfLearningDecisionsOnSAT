function p=plotInterleaved(t_data, sigma_data, r, n, semPerf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%function to plot Performance and Reaction times for interleaved task
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Collapsed_mixes=[1 0.6 0.68-0.32 0.56-0.44];

for j=1:4:13
    hold on
    p=r(j:j+3)./n(j:j+3);
    figure(1),errorbar(Collapsed_mixes, r(j:j+3)./n(j:j+3), semPerf(j:j+3), 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)
    
    Pc= @(y,x) (1-y(1))./(1+exp(-2.*y(2).*x.^y(3)));
    [y,R,J,CovB]=nlinfit(Collapsed_mixes,p,Pc,[0, 4.5, 0.5]);
    x=min(Collapsed_mixes):0.01:max(Collapsed_mixes);
    plot(x, Pc(y, x), 'k', 'LineWidth', 1.5)
    
    hold off
end

figure(1)
xlabel('Mixture contrast','FontName','Arial','FontSize',12);
ylabel('Accuracy','FontName','Arial','FontSize',12);

set(gca,'FontName','Arial','FontSize',12);
axis square;
ylim([0.5 1]);
xlim([0 1]);


figure(2)
for j=1:4:13
    hold on
    errorbar(Collapsed_mixes, t_data(j:j+3), sigma_data(j:j+3), 'o','MarkerEdgeColor','k', 'MarkerFaceColor','k', 'MarkerSize',8)

    ttt=t_data(j:j+3);
    
    %Palmer et al.
    tT= @(y,x) y(1)./((y(2).*x).^y(3)).*tanh(y(1).*((y(2).*x).^y(3)))+y(4);
    [y,~,~,~]=nlinfit(Collapsed_mixes,ttt,tT,[0, 4.5, 0.5 0.2]);

    %plot(Collapsed_mixes, t_model, 'k', 'LineWidth', 1.5)
    plot(x, tT(y, x), 'k', 'LineWidth', 1.5)
    
    
    hold off
 end


figure(2)
xlabel('Mixture contrast','FontName','Arial','FontSize',12);
ylabel('Reaction times (s)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);
axis square;
ylim([0.260 0.420]);
xlim([0 1]);

