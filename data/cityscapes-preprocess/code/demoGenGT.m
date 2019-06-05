% --------------------------------------------------------
% Copyright (c) Yuan Hu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Generate ground truths for evaluation on Cityscapes
% 2. Create filelists for the generated ground truths
% --------------------------------------------------------

function demoGenGT()

clc; clear; close all;

%% Add library paths
path = genpath('../../../lib/matlab');
addpath(path);

%% Setup Directories
dataRoot = '/home/huyuan/seal/data/cityscapes-preprocess/data_orig'; %'../data_orig';
genDataRoot = {'../gt_eval/gt_thin', '../gt_eval/gt_raw'};
suffixImage = '_leftImg8bit.png';
suffixLabelIds = '_gtFine_labelIds.png';
suffixInstIds = '_gtFine_instanceIds.png';

%% Setup Parameters
numCls = 19;
radius = [1, 2];
scale = 0.5;
edgeType = 'regular';

%% Setup parallel pool
numWorker = 6; % Number of matlab workers for parallel computing
delete(gcp('nocreate'));
parpool('local', numWorker);

%% Generate Preprocessed Dataset
for idx = 1:length(genDataRoot)
    % Create output directories
    genDataClsRoot = [genDataRoot{idx} '/cls'];
    genDataInstRoot = [genDataRoot{idx} '/inst'];
    if(exist(genDataClsRoot, 'file')==0)
        mkdir(genDataClsRoot);
    end
    if(exist(genDataInstRoot, 'file')==0)
        mkdir(genDataInstRoot);
    end
    
    setList = {'val', 'train'};
    for idxSet = 1:length(setList)
        setName = setList{idxSet};
        dataList = cell(1, 1);
        countFile = 0;
        cityList = dir([dataRoot '/leftImg8bit/' setName]);
        for idxCity = 3:length(cityList)
            cityName = cityList(idxCity, 1).name;
            fileList = dir([dataRoot '/leftImg8bit/' setName '/' cityName '/*.png']);
            
            % compute edges and write generated data
            display(['Set: ' setName ', City: ' cityName])
            parfor_progress(length(fileList));
            parfor idxFile = 1:length(fileList)
                assert(strcmp(fileList(idxFile).name(end-length(suffixImage)+1:end), suffixImage), 'suffixImage mismatch!')
                fileName = fileList(idxFile).name(1:end-length(suffixImage));
                if(idx==1)
                    labelIdMap = imresize(imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]), scale, 'nearest');
                    instIdMap = imresize(imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]), scale, 'nearest');
                else
                    labelIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
                    instIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]);
                end
                trainIdMap = labelid2trainid(labelIdMap);
                edgeMapBin = seg2edge(instIdMap, radius(idx), [2 3]', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
                [height, width, ~] = size(trainIdMap);
                
                % process instance-insensitive/sensitive GTs
                GTcls = [];
                GTcls.Boundaries = cell(numCls, 1);
                GTcls.Segmentation = zeros(height, width);
                GTcls.CategoriesPresent = [];
                GTinst = [];
                GTinst.Boundaries = cell(numCls, 1);
                GTinst.Segmentation = zeros(height, width);
                GTinst.Categories = [];
                
                for idxCls = 1:numCls
                    idxSeg = trainIdMap == idxCls-1;
                    if(sum(idxSeg(:))~=0)
                        segMap = zeros(size(instIdMap));
                        segMap(idxSeg) = instIdMap(idxSeg);
                        if(idx==1)
                            GTcls.Boundaries{idxCls, 1} = sparse(bwmorph(seg2edge_fast(idxSeg, edgeMapBin, radius(idx), [], edgeType), 'thin', inf));
                            GTcls.Segmentation(idxSeg) = idxCls;
                            GTcls.CategoriesPresent = [GTcls.CategoriesPresent; idxCls];
                            GTinst.Boundaries{idxCls, 1} = sparse(bwmorph(seg2edge_fast(segMap, edgeMapBin, radius(idx), [], edgeType), 'thin', inf));
                        else
                            GTcls.Boundaries{idxCls, 1} = sparse(imresize(seg2edge_fast(idxSeg, edgeMapBin, radius(idx), [], edgeType), scale, 'nearest'));
                            GTcls.Segmentation(idxSeg) = idxCls;
                            GTcls.CategoriesPresent = [GTcls.CategoriesPresent; idxCls];
                            GTinst.Boundaries{idxCls, 1} = sparse(imresize(seg2edge_fast(segMap, edgeMapBin, radius(idx), [], edgeType), scale, 'nearest'));
                        end
                    else
                        GTcls.Boundaries{idxCls, 1} = sparse(false(height, width));
                        GTinst.Boundaries{idxCls, 1} = sparse(false(height, width));
                    end
                end
                
                idxInst2 = 0;
                for idxInst = unique(instIdMap)'
                    idxSeg = instIdMap == idxInst;
                    idxCls = unique(trainIdMap(idxSeg));
                    if(length(idxCls)==1)
                        if(idxCls>=0 && idxCls<=numCls-1)
                            idxInst2 = idxInst2 + 1;
                            GTinst.Segmentation(idxSeg) = idxInst2;
                            GTinst.Categories(idxInst2, 1) = idxCls + 1;
                        end
                    else
                        error('Inconsistency between instIdMap and trainIdMap');
                    end
                end
                
                if(idx==2)
                    GTcls.Segmentation = imresize(GTinst.Segmentation, scale, 'nearest');
                    GTinst.Segmentation = imresize(GTinst.Segmentation, scale, 'nearest');
                end
                
                saveGTcls([genDataClsRoot '/' fileName suffixImage(1:max(strfind(suffixImage, '.'))-1) '.mat'], GTcls); % gt/image names must be the same
                saveGTinst([genDataInstRoot '/' fileName suffixImage(1:max(strfind(suffixImage, '.'))-1) '.mat'], GTinst); % gt/image names must be the same
                parfor_progress();
            end
            parfor_progress(0);
            
            % Write file lists
            for idxFile = 1:length(fileList)
                countFile = countFile + 1;
                fileName = fileList(idxFile).name(1:end-length(suffixImage));
                dataList{countFile} = [fileName suffixImage(1:max(strfind(suffixImage, '.'))-1)]; % gt/image names must be the same
            end
        end
        save([genDataRoot{idx} '/' setName '.mat'], 'dataList');
    end
end
