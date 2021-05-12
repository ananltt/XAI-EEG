clear all
close all
clc

%% Load il segnale EEG e qualche proprietà

[signal, header] = sload('dataset/A01E.gdf');

DUR = 7.5;      %Durata del segmento di interesse
fs = 250;       %Sampling frequency
N = DUR * fs;   %Numero di sample in ogni segmento di interesse

sigElements = [header.EVENT.POS, header.EVENT.DUR, header.EVENT.TYP]; %Extract the segments' information

c = zeros(288,1);
k = 1;
for i = 1:length(sigElements(:,1))
    elements = sigElements(:,3);
    if elements(i) == 769 || elements(i) == 770 || elements(i) == 771 || elements(i) == 772
        c(k)=i;
        k = k+1;
    end
end

% k = 0;
% for i = 1:length(sigElements(:,1))
%     c_dur = sigElements(:,2);
%     c_dur = c_dur(i);
%     
%     if c_dur == 1875
%         c_pos = sigElements(:,1);
%         c_pos = c_pos(i);
%         
%         s = signal(c_pos:(c_pos+c_dur-1),1:22);
%         
%         path = "dataset/EEG/S1_"+k+".mat";
%         save(path, 's');
%         k = k+1;
%     end
% end   