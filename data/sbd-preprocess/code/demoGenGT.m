% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Generate ground truths for evaluation on SBD
% 2. Create filelists for the generated ground truths
% --------------------------------------------------------

function demoGenGT()

clc; clear; close all;

%% Add library paths
path = genpath('../../../lib/matlab');
addpath(path);

%% Setup Directories
dataRoot = '/home/huyuan/seal/data/sbd-preprocess/data_orig'; %'../data_orig';
genDataRoot = {'../gt_eval/gt_orig_thin', '../gt_eval/gt_orig_raw'};

%% Setup Parameters
numCls = 20;
radius = 2;
edgeType = 'regular';
numVal = 1000; % param not effective when flagSeed is true & seed exists
flagSeed = true;

%% Setup Parallel Pool
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
    
    setList = {'train', 'val'};
    for setID = 1:length(setList)
        setName = setList{1, setID};
        fidIn = fopen([dataRoot '/' setName '.txt']);
        fileName = fgetl(fidIn);
        fileNameSet = cell(1,1);
        countFile = 0;
        while ischar(fileName)
            countFile = countFile + 1;
            fileNameSet{countFile} = fileName;
            fileName = fgetl(fidIn);
        end
        fclose(fidIn);
        
        if(strcmp(setName, 'train'))
            if(exist('./seed.mat', 'file') && flagSeed)
                s = load('./seed.mat');
                fields = fieldnames(s);
                valSet = s.(fields{1});
                numVal = sum(valSet); % Overwrite input numVal
            else
                valSet = false(countFile, 1);
                idxRand = randperm(countFile);
                valSet(idxRand(1:numVal), 1) = true;
                save('./seed.mat', 'valSet');
            end
        end
        
        % compute edges and write generated data
        disp(['Computing ' setName ' set boundaries'])
        parfor_progress(countFile);
        parfor idxFile = 1:countFile %parfor
            fileName = fileNameSet{idxFile};
            gt_cls = load([dataRoot '/cls/' fileName '.mat']);
            gt_inst = load([dataRoot '/inst/' fileName '.mat']);
            [height, width] = size(gt_inst.GTinst.Segmentation);
            
            % process instance-insensitive GTs
            if(idx==1)
                GTcls = gt_cls.GTcls;
            else
                labelEdgeCat = cell(numCls, 1);
                for idxCls = 1:numCls
                    idxSeg = gt_cls.GTcls.Segmentation == idxCls;
                    idxEdge = false(height, width);
                    if(sum(idxSeg(:))~=0)
                        idxEdge = seg2edge(idxSeg, radius, [], edgeType);
                    end
                    labelEdgeCat{idxCls, 1} = sparse(idxEdge);
                end
                GTcls = gt_cls.GTcls;
                GTcls.Boundaries = labelEdgeCat;
            end
            saveGTcls([genDataClsRoot '/' fileName '.mat'], GTcls);
            
            % process instance-sensitive GTs
            GTinst = [];
            GTinst.Segmentation = gt_inst.GTinst.Segmentation;
            GTinst.Categories = gt_inst.GTinst.Categories;
            labelEdgeInst = cell(numCls, 1);
            for idxCls = 1:numCls
                labelEdgeInst{idxCls} = sparse(false(size(gt_inst.GTinst.Segmentation)));
            end
            if(idx==1)
                for idxLabel = 1:length(gt_inst.GTinst.Categories)
                    idxCls = gt_inst.GTinst.Categories(idxLabel);
                    labelEdgeInst{idxCls} = sparse( full(labelEdgeInst{idxCls}) | full(gt_inst.GTinst.Boundaries{idxLabel}) );
                end
                for idxCls = 1:numCls
                    if(sum(labelEdgeInst{idxCls}(:))>0)
                        labelEdgeInst{idxCls} = sparse(bwmorph(full(labelEdgeInst{idxCls}), 'thin', inf));
                    end
                end
            else
                set_cls = unique(gt_inst.GTinst.Categories)';
                for idxCls = set_cls
                    set_inst = find(gt_inst.GTinst.Categories == idxCls)';
                    seg_map_inst = zeros(height, width);
                    for inst = set_inst
                        seg_map_inst(gt_inst.GTinst.Segmentation==inst) = inst;
                    end
                    labelEdgeInst{idxCls, 1} = sparse(seg2edge(seg_map_inst, radius, [], edgeType));
                end
            end
            GTinst.Boundaries = labelEdgeInst;
            saveGTinst([genDataInstRoot '/' fileName '.mat'], GTinst);
            
            parfor_progress();
        end
        parfor_progress(0);
        
        % Write file lists
        disp(['Generating ' setName ' set file lists'])
        listTrain = cell(countFile-numVal, 1);
        listVal = cell(numVal, 1);
        listTest = cell(countFile, 1);
        countTrain = 0;
        countVal = 0;
        parfor_progress(countFile);
        for idxFile = 1:countFile
            fileName = fileNameSet{idxFile}; % gt/image names must be the same
            if(strcmp(setName, 'train'))
                if(~valSet(idxFile, 1))
                    countTrain = countTrain+1;
                    listTrain{countTrain} = fileName;
                else
                    countVal = countVal+1;
                    listVal{countVal} = fileName;
                end
            else
                listTest{idxFile} = fileName;
            end
            parfor_progress();
        end
        parfor_progress(0);
        if(strcmp(setName, 'train'))
            save([genDataRoot{idx} '/train.mat'], 'listTrain');
            save([genDataRoot{idx} '/val.mat'], 'listVal');
        else
            save([genDataRoot{idx} '/test.mat'], 'listTest');
        end
    end
end
