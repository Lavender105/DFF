% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This script is used to:
% 1. Calculate class-wise Precision (P), Recall (R) and F-measure (F)
% 2. Output bestF, bestP, bestR, bestT at optimal dataset scale (ODS)
% 3. Output F_max, P_max, R_max at optimal image scale (OIS)
% 4. Output class-wise Average Precision (AP)
% --------------------------------------------------------

function result_cls = collect_eval_bdry(result_img)

dR = 0.01;
num_file = size(result_img, 1);
AA = result_img{1, 1}; % thresh cntR sumR cntP sumP
thresh = AA(:, 1);
nthresh = numel(thresh);
cntR_total = zeros(nthresh,1);
sumR_total = zeros(nthresh,1);
cntP_total = zeros(nthresh,1);
sumP_total = zeros(nthresh,1);
cntR_max = 0;
sumR_max = 0;
cntP_max = 0;
sumP_max = 0;

for i = 1:num_file
    AA  = result_img{i, 1};
    cntR = AA(:, 2);
    sumR = AA(:, 3);
    cntP = AA(:, 4);
    sumP = AA(:, 5);

    % Accumulate for ODS
    cntR_total = cntR_total + cntR;
    sumR_total = sumR_total + sumR;
    cntP_total = cntP_total + cntP;
    sumP_total = sumP_total + sumP;
    
    % Accumulate for OIS
    R = cntR ./ (sumR + (sumR==0));
    P = cntP ./ (sumP + (sumP==0));
    F = fmeasure(R, P);
    [~, idx_ois] = max(F);
    % fprintf(1, '%f cr:%f sr:%f cp:%f sp:%f \n', max(F), cntR(idx_ois), sumR(idx_ois), cntP(idx_ois),sumP(idx_ois));
    cntR_max = cntR_max + cntR(idx_ois);
    sumR_max = sumR_max + sumR(idx_ois);
    cntP_max = cntP_max + cntP(idx_ois);
    sumP_max = sumP_max + sumP(idx_ois);
end
R = cntR_total ./ (sumR_total + (sumR_total==0));
P = cntP_total ./ (sumP_total + (sumP_total==0));
F = fmeasure(R, P);
[bestT, bestR, bestP, bestF] = maxF(thresh, R, P);

R_max = cntR_max ./ (sumR_max + (sumR_max==0));
P_max = cntP_max ./ (sumP_max + (sumP_max==0));
F_max = fmeasure(R_max, P_max);

[Ru, indR, ~] = unique(R);
Pu = P(indR);
Ri = 0 : dR : 1;
if numel(Ru)>1,
    P_int1 = interp1(Ru, Pu, Ri);
    P_int1(isnan(P_int1)) = 0;
    AP = sum(P_int1)*dR;
else
    AP = 0;
end

result_cls = cell(2, 1);
result_cls{1, 1} = [thresh, R, P, F];
result_cls{2, 1} = [bestT, bestR, bestP, bestF, R_max, P_max, F_max, AP];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute f-measure fromm recall and precision
function [f] = fmeasure(r,p)
f = 2*p.*r./(p+r+((p+r)==0));

% interpolate to find best F and coordinates thereof
function [bestT, bestR, bestP, bestF] = maxF(thresh, R, P)
bestT = thresh(1);
bestR = R(1);
bestP = P(1);
bestF = fmeasure(R(1),P(1));
for i = 2:numel(thresh),
    for d = linspace(0,1),
        t = thresh(i)*d + thresh(i-1)*(1-d);
        r = R(i)*d + R(i-1)*(1-d);
        p = P(i)*d + P(i-1)*(1-d);
        f = fmeasure(r,p);
        if f > bestF,
            bestT = t;
            bestR = r;
            bestP = p;
            bestF = f;
        end
    end
end

