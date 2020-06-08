function psych = psych_func(y,x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca
%a psychometric curve - y(1) and y(4) lapses
% y(3) - std - slope
% y(4) - mean - bias
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psych = (y(1)+(y(4)).*normcdf(x,y(2),y(3)));