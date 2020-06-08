function [Pc2, RTS2, MIX2, sem_RTS2, n2]=eighTOfour2(Pc, RTs, STIM, sem_RTs, n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%amendonca - eighTOfour2 function
%second version of eightTOfour function. This time to include inversion for
%standard error stuff
%I use this function to transform from 8 (full) to 4 (contrast) data
%INPUT VARIABLES
%Pc - proportins for full
%RTs - Reaction times for full
%STIM - stimuli for full
%sem_RTs - standard error for reaction times
%n - number of trials
%OUTPUT VARIABLES
%Pc2 - proportions for collapsed
%RTS2 - Reaction times for collapsed
%MIX2 - Collapsed stimuli
%sem_RTS2 - collapsed standard error of the mean
%n2 - collapsed number of trials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Pc2=[(1-Pc(1))+Pc(8) (1-Pc(2))+Pc(7) (1-Pc(3))+Pc(6) (1-Pc(4))+Pc(5)]/2;
RTS2=[RTs(1)+RTs(8) RTs(2)+RTs(7) RTs(3)+RTs(6) RTs(4)+RTs(5)]/2;
n2=[n(1)+n(8) n(2)+n(7) n(3)+n(6) n(4)+n(5)];
sem_RTS2=sqrt([(n(1).*sem_RTs(1).^2+n(8).*sem_RTs(8).^2)./(n(1)+n(8)) (n(2).*sem_RTs(2).^2+n(7).*sem_RTs(7).^2)./(n(2)+n(7))...
    (n(3).*sem_RTs(3).^2+n(6).*sem_RTs(6).^2)./(n(3)+n(6)) (n(4).*sem_RTs(4).^2+n(5).*sem_RTs(5).^2)./(n(4)+n(5))])./2;
CONTRAST=[STIM(8)-STIM(1) STIM(7)-STIM(2) STIM(6)-STIM(3) STIM(5)-STIM(4)];
MIX2=(CONTRAST);
