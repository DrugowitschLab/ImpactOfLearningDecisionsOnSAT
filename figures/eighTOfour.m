function [Pc2, RTS2, MIX2]=eighTOfour(Pc, RTs, STIM)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - eighTOfour function
%I use this function to transform from 8 (full) to 4 (contrast) data
%INPUT VARIABLES
%Pc - proportins for full
%RTs - Reaction times for full
%STIM - stimuli for full
%OUTPUT VARIABLES
%Pc2 - proportions for collapsed
%RTS2 - Reaction times for collapsed
%MIX2 - Collapsed stimuli
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Pc2=[(1-Pc(1))+Pc(8) (1-Pc(2))+Pc(7) (1-Pc(3))+Pc(6) (1-Pc(4))+Pc(5)]/2;
Pc2=fliplr(Pc2);
RTS2=[RTs(1)+RTs(8) RTs(2)+RTs(7) RTs(3)+RTs(6) RTs(4)+RTs(5)]/2;
RTS2=fliplr(RTS2);
CONTRAST=[STIM(8)-STIM(1) STIM(7)-STIM(2) STIM(6)-STIM(3) STIM(5)-STIM(4)];
MIX2=fliplr(CONTRAST);
