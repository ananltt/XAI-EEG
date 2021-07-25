%% XAI-EEG data loading

clear all
close all
clc 

dataset_folder_name = '../dataset/EEG';
mkdir([dataset_folder_name]);

for i = 1:9
    
    file_name = convertStringsToChars("../dataset/A0"+i+"T.gdf");
    
    [s,h] = sload(file_name);
    
    event_type = h.EVENT.TYP;
    event_pos = h.EVENT.POS;
    event_duration = h.EVENT.DUR;
    
    % Start of the various runs
    runs_idx = event_pos(event_type == 32766);
    runs_idx = runs_idx(4:end);
    
    % Point of the first run
    start_point = runs_idx(1);
    
    % Select the runs data and consider only EEG
    data = s(start_point:end, 1:22);
    
    % Shift indices (Caused by removing initial part)
    runs_idx = runs_idx - start_point + 1; 
    event_pos = event_pos - start_point;
    event_type = event_type(event_pos >= 0);
    event_duration = event_duration(event_pos >= 0);
    event_pos = event_pos(event_pos >= 0);
    
    % Remove Nan that indicates tha start of trials
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

