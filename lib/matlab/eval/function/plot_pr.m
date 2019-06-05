% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This function is used to collect results and plot precision-recall curves.
% The function supports plotting multiple evaluation results on the same figure.
% To do that, input the first two arguments as: {scoreDir1; scoreDir2; ...}
% and {mtdNames1; mtdNames2; ...}
%
% Input arguments:
%   scoreDir    : Directory containing evaluation scores
%   mtdNames    : List of method names corresponding to evaluation scores
%   numCls      : Number of classes
%   plotDir     : Directory to put plotted PR curves
%   categories  : List of class names
%   flagAP      : Flag to include AP score in plot legend or not
% --------------------------------------------------------

function plot_pr(scoreDir, mtdNames, plotDir, categories, flagAP)

if(exist(plotDir, 'file')==0)
    mkdir(plotDir);
end

numCls = length(categories);
numMtd = length(scoreDir);
assert(length(scoreDir)==length(mtdNames), ...
    'Number of input dirs must be equal to input method names!');

%% Collect evaluation results
resultF = zeros(numCls, numMtd);
for i = 1:numMtd
    scoreLst = dir([scoreDir{i, 1} '/class_*.mat']);
    numResult = size(scoreLst, 1);
    for j = 1:numResult
        result_name = scoreLst(j, 1).name;
        idx = find(result_name=='_', 1, 'last');
        idxCls = str2double(result_name(idx+1:end-4));
        load([scoreDir{i, 1} '/class_' num2str(idxCls)])
        resultF(idxCls, i) = result_cls{2, 1}(4);
    end
end
resultF = resultF'.*100;

%% Summarize evaluation results
fprintf('====================== Summary MF-ODS ======================\n\n');
for idxCls = 1:numCls
    fprintf('%3d %14s:  ', idxCls, categories{idxCls});
    for idxMtd = 1:numMtd
        fprintf('%.2f   ', resultF(idxMtd, idxCls))
    end
    fprintf('\n');
end
fprintf('\n');
MF_ODS_Mean = mean(resultF);
fprintf('        Mean F-ODS:  ');
for idxMtd = 1:numMtd
    fprintf('%.2f   ', MF_ODS_Mean(idxMtd));
end
fprintf('\n\n');

%% Plot class-wise PR curves
for idxCls = 1:numCls
    fprintf(['Plotting class %d "' categories{idxCls} '" precision-recall curve\n'], idxCls);
    resultLst = cell(numMtd, 1);

    for idxMtd = 1:numMtd
        s = load(fullfile(scoreDir{idxMtd}, ['/class_' num2str(idxCls) '.mat']));
        names = fieldnames(s);
        resultLst{idxMtd} = s.(names{1});
    end
    
    % plot figure
    [F_ODS, ~, AP, H] = plot_pr_multiple(resultLst);
    legendLst = cell(numMtd, 1);
    for idxMtd = 1:numMtd
        if(flagAP)
            legendLst{idxMtd, 1} = ['[F=' num2str(F_ODS(idxMtd), '%1.3f') ' AP=' num2str(AP(idxMtd), '%1.3f') '] ' mtdNames{idxMtd, 1}];
        else
            legendLst{idxMtd, 1} = ['[F=' num2str(F_ODS(idxMtd), '%1.3f') '] ' mtdNames{idxMtd, 1}];
        end
    end
    set(gca,'fontsize',10)
    title(categories{idxCls},'FontSize',14,'FontWeight','bold')
    xlabel('Recall','FontSize',14,'FontWeight','bold')
    ylabel('Precision','FontSize',14,'FontWeight','bold')
    hLegend = legend(H, legendLst, 'Location', 'SouthWest');
    set(hLegend, 'FontSize', 12);
    
    % save figure
    print(gcf, fullfile(plotDir, ['/class_' num2str(idxCls, '%03d') '.pdf']),'-dpdf')
    close(gcf);
end