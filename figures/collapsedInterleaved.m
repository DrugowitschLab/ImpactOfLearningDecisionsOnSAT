function [corrects, trials, t_data, sigma_data]=collapsedInterleaved(r, n, sampling_times,sigma_d)
corrects=zeros(1,16);
trials=zeros(1,16);
t_data=zeros(1,16);
sigma_data=zeros(1,16);

for i=1:4
    for j=0:8:24
        %collapse is 1->5 (easy); 2->6; 3->7; 4->8 (difficult) and so on
        corrects(1,i+j/2)=(n(i+j)-r(i+j)+r(i+j+4))./2;
        trials(1,i+j/2)=(n(i+j)+n(i+j+4))./2;
        t_data(1,i+j/2)=(sampling_times(i+j)+sampling_times(i+j+4))./2;
        sigma_data(1,i+j/2)=(sigma_d(i+j)+sigma_d(i+j+4))./2;
    end
end