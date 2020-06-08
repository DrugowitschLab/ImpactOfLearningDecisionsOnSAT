%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - code for Figure 8
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

%Load data for categorization task
load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'adfcolbiaslapse_psychchron1_categ'])


%extract relevant variables for plotting categorization
TE_CAT.w=-w;
TE_CAT.w_b=bias;
TE_CAT.Correct=corr;
TE_CAT.intX=x;
TE_CAT.rts=dt;
TE_CAT.Stimulus=stimid;
TE_CAT.ChoiceDir=resp;

%variable to play with residual time, here its ignored
tR=0;

%Load data for identification task
load(['..' filesep 'fitdata' filesep 'modelsims' filesep 'adfcolbiaslapse_psychchron1_ident'])

%extract relevant variables for plotting categorization (previously known
%as detection)
TE_DET.w=-w;
TE_DET.w_b=bias;
TE_DET.Correct=corr;
TE_DET.intX=x;
TE_DET.rts=dt;
TE_DET.Stimulus=stimid;
TE_DET.ChoiceDir=resp;


%calculate infered drifts = integrated evidece / time to decide
infered_drifts_cat=[TE_CAT.intX(:,1)./(TE_CAT.rts-tR) TE_CAT.intX(:,2)./(TE_CAT.rts-tR)];

%hard right decisions
hard_rights_cat=find(TE_CAT.Stimulus==5);

%infered for hard right decisions
infered_drifts_cat_hard_rights=infered_drifts_cat(hard_rights_cat,:);

%correct infered drifts
corrects=find(infered_drifts_cat_hard_rights(:,1)-infered_drifts_cat_hard_rights(:,2)>0);
%incorrect infered drifts
incorrects=find(infered_drifts_cat_hard_rights(:,1)-infered_drifts_cat_hard_rights(:,2)<0);

%weights for hard rights
weights_cat_hard_rights= TE_CAT.w(hard_rights_cat,:);


%plotting correct infered drifts in blue
marker_size=5;
figure,plot(infered_drifts_cat_hard_rights(corrects,1).*0.7, infered_drifts_cat_hard_rights(corrects,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b')

%plotting incorrect infered drifts in blue
hold on
plot(infered_drifts_cat_hard_rights(incorrects,1).*0.7, infered_drifts_cat_hard_rights(incorrects,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r')
hold off

xlim([-40 120])
ylim([-40 120])
axis square

xlabel('Inferred drift rate 1 (au)','FontName','Arial','FontSize',12);
ylabel('Inferred drift rate 2 (au)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);


%look at corrects for hard rights
corrects=find(infered_drifts_cat_hard_rights(:,1).*weights_cat_hard_rights(:,1)-infered_drifts_cat_hard_rights(:,2).*(-weights_cat_hard_rights(:,2))>0);
%look at corrects if weights are not considered (before multiplication)
corrects_before=find(infered_drifts_cat_hard_rights(:,1).*0.7-infered_drifts_cat_hard_rights(:,2).*(0.7)>0);

%look at incorrects for hard rights
incorrects=find(infered_drifts_cat_hard_rights(:,1).*weights_cat_hard_rights(:,1)-infered_drifts_cat_hard_rights(:,2).*(-weights_cat_hard_rights(:,2))<0);
%look at incorrects if weights are not considered (before multiplication)
incorrects_before=find(infered_drifts_cat_hard_rights(:,1).*0.7-infered_drifts_cat_hard_rights(:,2).*0.7<0);

%dark blue are corrects always
dark_blue=[intersect(corrects, corrects_before)];
%dark red are incorrects always
dark_red=[intersect(incorrects, incorrects_before)];

%blues now are incorrects that became correct
blues_now=[intersect(corrects, incorrects_before)];

%reds now are corrects that became incorrects
reds_now=[intersect(incorrects, corrects_before)];

%slope of categorization separation line
slopes=-weights_cat_hard_rights(:,1)./weights_cat_hard_rights(:,2);

figure

hold on

%plotting weight varliability
line_lightness=0.8;
xaux=-1000:1000;

sd=std(slopes)./3;

slopes_aux=1./sd:(sd-1./sd)./1:sd
for i=1:length(slopes_aux)
    plot(xaux,slopes_aux(i).*xaux,'Color', [line_lightness line_lightness line_lightness])
end

points_lightness=0.7;

%plotting always corrects
plot(infered_drifts_cat_hard_rights(dark_blue,1).*0.7, infered_drifts_cat_hard_rights(dark_blue,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor',[points_lightness points_lightness 1],...
    'MarkerFaceColor',[points_lightness points_lightness 1])
hold on
%plotting always incorrects
plot(infered_drifts_cat_hard_rights(dark_red,1).*0.7, infered_drifts_cat_hard_rights(dark_red,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor',[1 points_lightness points_lightness],...
    'MarkerFaceColor',[1 points_lightness points_lightness])

%plotting incorrects that became corrects
plot(infered_drifts_cat_hard_rights(blues_now,1).*0.7, infered_drifts_cat_hard_rights(blues_now,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b')

%plotting corrects that became incorrects
plot(infered_drifts_cat_hard_rights(reds_now,1).*0.7, infered_drifts_cat_hard_rights(reds_now,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r')


hold off
xlim([-40 120])
ylim([-40 120])
axis square
xlabel('Inferred drift rate 1 (au)','FontName','Arial','FontSize',12);
ylabel('Inferred drift rate 2 (au)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);



%histograms
%for categorization

figure
nbins=125;

%values for weight * infered drifts
infer_times_weights=[infered_drifts_cat_hard_rights(:,1).*weights_cat_hard_rights(:,1) infered_drifts_cat_hard_rights(:,2).*(-weights_cat_hard_rights(:,2))];
infer_times_weights_cat=infer_times_weights;

% bin windows
x=min(infer_times_weights_cat(:,1)-infer_times_weights_cat(:,2)):max((infer_times_weights_cat(:,1)-infer_times_weights_cat(:,2))-min(infer_times_weights_cat(:,1)-infer_times_weights_cat(:,2)))./(nbins-1):max((infer_times_weights_cat(:,1)-infer_times_weights_cat(:,2)));

%counting
[n_cat_dark_blue, centers]=hist((infered_drifts_cat_hard_rights(dark_blue,1)-infered_drifts_cat_hard_rights(dark_blue,2))*0.7,nbins, 'Edges', x);

%plotting histograms for always corrects
stairs(centers, n_cat_dark_blue, 'k','LineWidth', 1, 'Color',[points_lightness points_lightness 1] );

hold on

%plotting histograms for always incorrects
[n_cat_dark_red, centers]=hist((infered_drifts_cat_hard_rights(dark_red,1)-infered_drifts_cat_hard_rights(dark_red,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_cat_dark_red, 'k','LineWidth', 1, 'Color', [1 points_lightness points_lightness]);

%plotting histograms for now correct
[n_cat_blues_now, centers]=hist((infered_drifts_cat_hard_rights(blues_now,1)-infered_drifts_cat_hard_rights(blues_now,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_cat_blues_now,'b','LineWidth', 1 );

%plotting histograms for now incorrect
[n_cat_reds_now, centers]=hist((infered_drifts_cat_hard_rights(reds_now,1)-infered_drifts_cat_hard_rights(reds_now,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_cat_reds_now,'r','LineWidth', 1);



hold off
xlim([-50 50])
axis square
box off
xlabel('\mu_{1} - \mu_{2} (a.u.)', 'FontSize', 12);
set(gca,'FontName','Arial','FontSize',12);


%same hisogram plotting as before but now only corrects and errors without weights
figure
[n_cat_corrects_before, centers]=hist((infered_drifts_cat_hard_rights(corrects_before,1)-infered_drifts_cat_hard_rights(corrects_before,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_cat_corrects_before, 'k','LineWidth', 1, 'Color','b' );
hold on

[n_cat_incorrects_before, centers]=hist((infered_drifts_cat_hard_rights(incorrects_before,1)-infered_drifts_cat_hard_rights(incorrects_before,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_cat_incorrects_before, 'k','LineWidth', 1, 'Color', 'r');

hold off
xlim([-50 50])
axis square
box off

xlabel('\mu_{1} - \mu_{2} (a.u.)', 'FontSize', 12);

set(gca,'FontName','Arial','FontSize',12);

%%
%here we do exactly the same as before but for identification (detection)
%task
infered_drifts_det=[TE_DET.intX(:,1)./(TE_DET.rts-tR) TE_DET.intX(:,2)./(TE_DET.rts-tR)];

hard_rights_det=find(TE_DET.Stimulus==5);
infered_drifts_det_hard_rights=infered_drifts_det(hard_rights_det,:);

weights_det_hard_rights= TE_DET.w(hard_rights_det,:);

infer_times_weights_det=infer_times_weights;


corrects=find(infered_drifts_det_hard_rights(:,1).*weights_det_hard_rights(:,1)-infered_drifts_det_hard_rights(:,2).*(-weights_det_hard_rights(:,2))>0);
incorrects=find(infered_drifts_det_hard_rights(:,1).*weights_det_hard_rights(:,1)-infered_drifts_det_hard_rights(:,2).*(-weights_det_hard_rights(:,2))<0);


figure, plot(infered_drifts_det_hard_rights(corrects,1).*0.7, infered_drifts_det_hard_rights(corrects,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b')
hold on
plot(infered_drifts_det_hard_rights(incorrects,1).*0.7, infered_drifts_det_hard_rights(incorrects,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r')
hold off


xlim([-40 120])
ylim([-40 120])
axis square

xlabel('Inferred drift rate 1 (au)','FontName','Arial','FontSize',12);
ylabel('Inferred drift rate 2 (au)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);


corrects=find(infered_drifts_det_hard_rights(:,1).*weights_det_hard_rights(:,1)-infered_drifts_det_hard_rights(:,2).*(-weights_det_hard_rights(:,2))>0);
corrects_before=find(infered_drifts_det_hard_rights(:,1).*0.7-infered_drifts_det_hard_rights(:,2).*(0.7)>0);

incorrects=find(infered_drifts_det_hard_rights(:,1).*weights_det_hard_rights(:,1)-infered_drifts_det_hard_rights(:,2).*(-weights_det_hard_rights(:,2))<0);
incorrects_before=find(infered_drifts_det_hard_rights(:,1).*0.7-infered_drifts_det_hard_rights(:,2).*0.7<0);

dark_blue=[intersect(corrects, corrects_before)];
dark_red=[intersect(incorrects, incorrects_before)];

blues_now=[intersect(corrects, incorrects_before)];

reds_now=[intersect(incorrects, corrects_before)];

slopes=-weights_det_hard_rights(:,1)./weights_det_hard_rights(:,2);

figure

hold on


xaux=-1000:1000;

sd=std(slopes)./3;

slopes_aux=1./sd:(sd-1./sd)./1:sd
for i=1:length(slopes_aux)
    plot(xaux,slopes_aux(i).*xaux,'Color', [line_lightness line_lightness line_lightness])
end


plot(infered_drifts_det_hard_rights(dark_blue,1).*0.7, infered_drifts_det_hard_rights(dark_blue,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor',[points_lightness points_lightness 1],...
    'MarkerFaceColor',[points_lightness points_lightness 1])
hold on

plot(infered_drifts_det_hard_rights(dark_red,1).*0.7, infered_drifts_det_hard_rights(dark_red,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor',[1 points_lightness points_lightness],...
    'MarkerFaceColor',[1 points_lightness points_lightness])

plot(infered_drifts_det_hard_rights(blues_now,1).*0.7, infered_drifts_det_hard_rights(blues_now,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b')

plot(infered_drifts_det_hard_rights(reds_now,1).*0.7, infered_drifts_det_hard_rights(reds_now,2).*0.7, 'o','MarkerSize',marker_size,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r')


hold off


xlim([-40 120])
ylim([-40 120])
axis square

xlabel('Inferred drift rate 1 (au)','FontName','Arial','FontSize',12);
ylabel('Inferred drift rate 2 (au)','FontName','Arial','FontSize',12);
set(gca,'FontName','Arial','FontSize',12);




%histograms
%for identification

figure
nbins=125;

x=min(infer_times_weights_det(:,1)-infer_times_weights_det(:,2)):max((infer_times_weights_det(:,1)-infer_times_weights_det(:,2))-min(infer_times_weights_det(:,1)-infer_times_weights_det(:,2)))./(nbins-1):max((infer_times_weights_det(:,1)-infer_times_weights_det(:,2)));

[n_det_dark_blue, centers]=hist((infered_drifts_det_hard_rights(dark_blue,1)-infered_drifts_det_hard_rights(dark_blue,2))*0.7,nbins, 'Edges', x);
dark_red=[intersect(incorrects, incorrects_before)];

stairs(centers, n_det_dark_blue, 'k','LineWidth', 1, 'Color',[points_lightness points_lightness 1] );

hold on

[n_det_dark_red, centers]=hist((infered_drifts_det_hard_rights(dark_red,1)-infered_drifts_det_hard_rights(dark_red,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_det_dark_red, 'k','LineWidth', 1, 'Color', [1 points_lightness points_lightness]);


[n_det_blues_now, centers]=hist((infered_drifts_det_hard_rights(blues_now,1)-infered_drifts_det_hard_rights(blues_now,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_det_blues_now,'b','LineWidth', 1 );

[n_det_reds_now, centers]=hist((infered_drifts_det_hard_rights(reds_now,1)-infered_drifts_det_hard_rights(reds_now,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_det_reds_now,'r','LineWidth', 1);



hold off

xlim([-50 50])
axis square


box off

xlabel('\mu_{1} - \mu_{2} (a.u.)', 'FontSize', 11);

set(gca,'FontName','Arial','FontSize',11);


figure
[n_det_corrects_before, centers]=hist((infered_drifts_det_hard_rights(corrects_before,1)-infered_drifts_det_hard_rights(corrects_before,2))*0.7,nbins, 'Edges', x);

%stairs(centers, n_det_blackies, 'k','LineWidth', 1);
stairs(centers, n_det_corrects_before, 'k','LineWidth', 1, 'Color','b' );

hold on

[n_det_incorrects_before, centers]=hist((infered_drifts_det_hard_rights(incorrects_before,1)-infered_drifts_det_hard_rights(incorrects_before,2))*0.7,nbins, 'Edges', x);
stairs(centers, n_det_incorrects_before, 'k','LineWidth', 1, 'Color', 'r');

hold off


xlim([-50 50])
axis square


box off

xlabel('\mu_{1} - \mu_{2} (a.u.)', 'FontSize', 11);

set(gca,'FontName','Arial','FontSize',11);