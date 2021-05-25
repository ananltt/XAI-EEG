%% XAI-EEG data loading

clear all
close all
clc

%% Load EEG, extract interesting segments and save them as mat

reference = [];
folder = "../dataset/EEG/";

found = false;
for subject = 1:9
    path = convertStringsToChars("../dataset/A0"+subject+"T.gdf");
    [signal, header] = sload(path);

    DUR = 7.5;      %Segment duration
    fs = 250;       %Sampling frequency
    N = DUR * fs;   %Number of samples in the segment

    sigElements = [header.EVENT.POS, header.EVENT.DUR, header.EVENT.TYP]; %Extract the segments' information

    k = 1;

    for i = 1:(length(sigElements(:,1))-1)

        c_pos = sigElements(:,1); %Current initial position
        c_pos = c_pos(i);

        c_dur = sigElements(:,2); %Current duration
        c_dur = c_dur(i);

        c_ele = sigElements(:,3); %Current segment type (i+1???)
        c_ele = c_ele(i+1);

        if (c_ele == 769 || c_ele == 770) 
            
            begin = c_pos + 500;
            end_ = begin + 1000 - 1;
            s = (signal(begin:end_, 1:22))'; %Extract the signal of interest
            
            for i=1:size(s, 1)
                s(i,:) = normalize(s(i,:));
                s(i,:) = rescale(s(i,:));
            end
            
            path = folder+"S"+subject+"_"+k+".mat"; %Save the signal
            save(path, 's');
            
            if c_ele == 769, label = [1, 0]; end
            if c_ele == 770, label = [0, 1]; end
            
            current = [subject, k, label];
            reference(end+1, :) = current;
            
            k = k+1; 
            
            if found == false
                figure
                plot(1:1000, s(1,:), 'k', 'LineWidth', 1)
                grid on;
                xlabel('Samples')
                ylabel('EEG signal [ \mu V]')
                title('EEG signal')
                found = true;
            end
        end

    end
end

            
path = folder + "reference.csv";
writematrix(reference, path);
