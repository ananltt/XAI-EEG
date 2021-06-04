%% XAI-EEG data loading

clear all
close all
clc 

%%

file_name = '../dataset/A01E.gdf';
dataset_folder_name = '../dataset/EEG';
mkdir([dataset_folder_name]);

%%
for i = 1:9
    
    file_name = convertStringsToChars("../dataset/A0"+i+"E.gdf");
    
    [s,h] = sload(file_name);
    
    event_type = h.EVENT.TYP;
    event_pos = h.EVENT.POS;
    event_duration = h.EVENT.DUR;
    
    % Start of the various runs
    runs_idx = event_pos(event_type == 32766);
    runs_idx = runs_idx(4:end);
    
    % Point of the first run (The initial registration was performed for calibration)
    start_point = runs_idx(1);
    
    % Select the runs data and remove EOG channels
    data = s(start_point:end, 1:22);
    
    % Shift indices (Caused by removing initial part)
    runs_idx = runs_idx - start_point + 1; %+1 add for the matlab management of indeces
    event_pos = event_pos - start_point;
    event_type = event_type(event_pos >= 0);
    event_duration = event_duration(event_pos >= 0);
    event_pos = event_pos(event_pos >= 0);
    
    % Remove Nan that indicates tha start of trials (and adjust indeces)
    for j = 1:length(runs_idx)
        data(runs_idx(j):runs_idx(j) + 100, :) = [];
        
        event_pos(event_pos >= runs_idx(j)) = event_pos(event_pos >= runs_idx(j)) - 100;
    end
    
    % Remove unwanted Nan
    data = fillmissing(data,'linear');
    
    %Save position/event type/duration in a single matrix
    event_matrix = zeros(length(event_pos), 3);
    event_matrix(:, 1) = event_pos;
    event_matrix(:, 2) = event_type;
    event_matrix(:, 3) = event_duration;
    
    %Save variables
    save(strcat(dataset_folder_name, '/S' + string(i) + '_data.mat'), 'data')
    save(strcat(dataset_folder_name, '/S' + string(i) + '_label.mat'), 'event_matrix')

end

% clear all
% close all
% clc
% 
% %% Load EEG, extract interesting segments and save them as mat
% 
% reference = [];
% folder = "../dataset/EEG/";
% 
% found = false;
% 
% for subject = 1:9
%     data = [];
%     event_matrix = [];
% 
%     path = convertStringsToChars("../dataset/A0"+subject+"T.gdf");
%     [signal, header] = sload(path);
% 
%     DUR = 7.5;      %Segment duration
%     fs = 250;       %Sampling frequency
%     N = DUR * fs;   %Number of samples in the segment
% 
%     sigElements = [header.EVENT.POS, header.EVENT.DUR, header.EVENT.TYP]; %Extract the segments' information
% 
%     k = 1;
% 
%     for i = 1:(length(sigElements(:,1))-1)
% 
%         c_pos = sigElements(:,1); %Current initial position
%         c_pos = c_pos(i);
% 
%         c_dur = sigElements(:,2); %Current duration
%         c_dur = c_dur(i);
% 
%         c_ele = sigElements(:,3); %Current segment type (i+1???)
%         c_ele = c_ele(i+1);
% 
%         if (c_ele == 769 || c_ele == 770)
% 
%             begin = c_pos + 500;
%             end_ = begin + 1000 - 1;
%             s = (signal(begin:end_, 1:22))'; %Extract the signal of interest
% 
%             for i=1:size(s, 1)
%                 s(i,:) = normalize(s(i,:));
%                 s(i,:) = rescale(s(i,:));
%             end
% 
%             s = transpose(s);
% 
%             if c_ele == 769, label = [1, 0]; end
%             if c_ele == 770, label = [0, 1]; end
% 
%             current = [subject, k, label];
%             reference(end+1, :) = current;
% 
%              ev_mat = zeros(length(c_pos), 3);
%              ev_mat(:, 1) = c_pos;
%              ev_mat(:, 2) = c_ele;
%              ev_mat(:, 3) = c_dur;
% 
%              event_matrix = [event_matrix; ev_mat];
% 
%             k = k+1;
% 
%             data = [data; s];
% %             if found == false
% %                 figure
% %                 plot(1:1000, s(1,:), 'k', 'LineWidth', 1)
% %                 grid on;
% %                 xlabel('Samples')
% %                 ylabel('EEG signal [ \mu V]')
% %                 title('EEG signal')
% %                 found = true;
% %             end
%         end
% 
%     end
% 
%     path = folder+"S"+subject+"_data.mat";
%     save(path, 'data');
% 
%     path2 = folder+"S"+subject+"_label.mat";
%     save(path2, 'event_matrix');
% 
% 
% end
% 
%             
% path = folder + "reference.csv";
% writematrix(reference, path);
