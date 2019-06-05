function [F_ODS, F_OIS, AP, H] = plot_pr_multiple(resultLst)

% color list
cols={'r';'b';'m';'g';'c';'r--';'b--';'m--';'g--';'c--'} ;

% load background and plot grid
openfig('isoF.fig', 'new', 'invisible'); hold on % Certain Matlab versions may crash if visible  
for i = 0.1:0.1:0.9
    plot([0.01, 0.99], [i, i], 'linewidth', 0.5, 'color', [0.8 0.8 0.8]);
    plot([i, i], [0.01, 0.99], 'linewidth', 0.5, 'color', [0.8 0.8 0.8]);
end

% initialize output
F_ODS = zeros(length(resultLst), 1);
F_OIS = zeros(length(resultLst), 1);
AP = zeros(length(resultLst), 1);
H = zeros(1, length(resultLst));

for idxMtd = 1:length(resultLst)
    if(idxMtd>length(cols))
        col = 'c--';
    else
        col = cols{idxMtd};
    end
    prvals = resultLst{idxMtd}{1};
    f = prvals(:, 2)>=0.01;
    prvals = prvals(f, :);

    evalRes = resultLst{idxMtd}{2};
    if size(prvals, 1)>1,
        H(idxMtd) = plot(prvals(1:end, 2), prvals(1:end, 3), 'color', col, 'LineWidth', 3);
    else
        H(idxMtd) = plot(evalRes(2), evalRes(3), 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', col, 'MarkerSize',6);
    end
    
    F_ODS(idxMtd) = evalRes(4);
    F_OIS(idxMtd) = evalRes(7);
    AP(idxMtd) = evalRes(8);
end
hold off