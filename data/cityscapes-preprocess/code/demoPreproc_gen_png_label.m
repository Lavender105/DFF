% --------------------------------------------------------
% Copyright (c) Yuan Hu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Generate instance-sensitive multi-label semantic edges on the Cityscapes dataset
% 2. Create filelists for the generated data and labels
% --------------------------------------------------------

function demoPreproc_gen_png_label()
clc; clear; close all;

%% Add library paths
path = genpath('../../../lib/matlab');
addpath(path);
%% 

%% Setup Directories and Suffixes
dataRoot = '/home/huyuan/seal/data/cityscapes-preprocess/data_orig'; %'../data_orig';
genDataRoot = '../data_proc'; % debug
suffixImage = '_leftImg8bit.png';
suffixColor = '_gtFine_color.png';
suffixLabelIds = '_gtFine_labelIds.png';
suffixInstIds = '_gtFine_instanceIds.png';
suffixTrainIds = '_gtFine_trainIds.png';
suffixPolygons = '_gtFine_polygons.json';
suffixEdge = '_gtFine_edge.mat';
suffixEdge_png = '_gtFine_edge.png';

%% Setup Parameters
numCls = 19;
radius = 2;
flagPngFile = true; % Output .png edge label files

%% Setup Parallel Pool
numWorker = 6; % Number of matlab workers for parallel computing
delete(gcp('nocreate'));
parpool('local', numWorker);

%% Generate Output Directory
if(exist(genDataRoot, 'file')==0)
    mkdir(genDataRoot);
end

%% Preprocess Training Data and Labels
setList = {'train', 'val', 'test'};
for idxSet = 1:length(setList)
    setName = setList{idxSet};
    if(flagPngFile)
        fidList = fopen([genDataRoot '/' setName '.txt'], 'w');
    end
    dataList = cell(1, 1);
    countFile = 0;
    cityList = dir([dataRoot '/leftImg8bit/' setName]);
    for idxCity = 3:length(cityList)
        cityName = cityList(idxCity).name;
        if(exist([genDataRoot '/leftImg8bit/' setName '/' cityName], 'file')==0)
            mkdir([genDataRoot '/leftImg8bit/' setName '/' cityName]);
        end
        if(exist([genDataRoot '/gtFine/' setName '/' cityName], 'file')==0)
            mkdir([genDataRoot '/gtFine/' setName '/' cityName]);
        end
        fileList = dir([dataRoot '/leftImg8bit/' setName '/' cityName '/*.png']);
        
        % Generate and write data
        display(['Set: ' setName ', City: ' cityName])
        parfor_progress(length(fileList));
        parfor idxFile = 1:length(fileList)
        	assert(strcmp(fileList(idxFile).name(end-length(suffixImage)+1:end), suffixImage), 'suffixImage mismatch!')
            fileName = fileList(idxFile).name(1:end-length(suffixImage));
            % Copy image
            copyfile([dataRoot '/leftImg8bit/' setName '/' cityName '/' fileName suffixImage], [genDataRoot '/leftImg8bit/' setName '/' cityName '/' fileName suffixImage]);
            % Copy gt files
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixColor], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixColor]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixPolygons], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixPolygons]);
            if(~strcmp(setName, 'test'))
                % Transform label id map to train id map and write
                labelIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
                instIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]);
                trainIdMap = labelid2trainid(labelIdMap);
                imwrite(trainIdMap, [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixTrainIds], 'png');
                % Transform color map to edge map and write
                edgeMapBin = seg2edge(instIdMap, radius, [2 3]', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
                [height, width, ~] = size(trainIdMap);
                labelEdge = cell(numCls, 1);
                labelEdge2 = zeros(height, width, 'uint32');
                
                % Generate .png edge label
                labelEdge_b = zeros(height, width, 'uint8');
                labelEdge_g = zeros(height, width, 'uint8');
                labelEdge_r = zeros(height, width, 'uint8');
                labelEdge_png = zeros(height, width, 3, 'uint8');
                for idxCls = 1:numCls
                    idxSeg = trainIdMap == idxCls-1;
                    if(sum(idxSeg(:))~=0)
                        segMap = zeros(size(instIdMap));
                        segMap(idxSeg) = instIdMap(idxSeg);
                        idxEdge = seg2edge_fast(segMap, edgeMapBin, radius, [], 'regular');
                        labelEdge{idxCls, 1} = sparse(idxEdge);
                        labelEdge2(idxEdge) = labelEdge2(idxEdge) + 2^(idxCls-1);
                        if idxCls>=1 && idxCls<=8
                            labelEdge_b(idxEdge) = labelEdge_b(idxEdge) + 2^(idxCls-1);
                        elseif idxCls>=9 && idxCls<=16
                            labelEdge_g(idxEdge) = labelEdge_g(idxEdge) + 2^(idxCls-8-1);
                        else
                            labelEdge_r(idxEdge) = labelEdge_r(idxEdge) + 2^(idxCls-8-8-1);
                        end
                    else
                        labelEdge{idxCls, 1} = sparse(false(height, width));
                    end
                end
                labelEdge_png = cat(3, labelEdge_r, labelEdge_g, labelEdge_b);
                
                if(flagPngFile)
                    imwrite(labelEdge_png, [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixEdge_png], 'png');
                end
                savelabeledge([genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixEdge], labelEdge); % parfor does not support directly using save.
            end
            parfor_progress();
        end
        parfor_progress(0);
        
        % Create file lists
        for idxFile = 1:length(fileList)
            countFile = countFile + 1;
            fileName = fileList(idxFile).name(1:end-length(suffixImage));
            if(ismember(setName, {'train', 'val'}))
                if(flagPngFile)
                    fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage ' /gtFine/' setName '/' cityName '/' fileName suffixEdge_png '\n']);
                end
                dataList{countFile, 1} = ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage];
                dataList{countFile, 2} = ['/gtFine/' setName '/' cityName '/' fileName suffixEdge];
            else
                if(flagPngFile)
                    fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage '\n']);
                end
                dataList{countFile, 1} = ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage];
            end
        end
    end
    if(flagPngFile)
        fclose(fidList); %#ok<*UNRCH>
    end
    save([genDataRoot '/' setName '.mat'], 'dataList');
end
