function [thresh, cntR, sumR, cntP, sumP] = evaluate_bdry(pb, bdry, nthresh, thinpb, maxDist)
% Calculate precision/recall curve.
% INPUT
%	pb          : The candidate boundary image. Must be double
%	bdry		: Ground truth boundaries
%   nthresh     : Number of points in PR curve
%   MaxDist     : Edge misalignment tolerance threshold
%   thinpb      : Option to apply morphological thinning on evaluated boundaries
%
% OUTPUT
%	thresh		: Vector of threshold values.
%	cntR,sumR	: Ratio gives recall.
%	cntP,sumP	: Ratio gives precision.

%%%%%Initialize
%setup thresholds
thresh = linspace(1/(nthresh+1), 1-1/(nthresh+1), nthresh)';

% zero all counts
cntR = zeros(size(thresh));
sumR = zeros(size(thresh));
cntP = zeros(size(thresh));
sumP = zeros(size(thresh));

%%%%%Evaluate
for t = 1:nthresh,
    bmap = double(pb>=thresh(t));
    % thin the thresholded pb to make sure boundaries are standard thickness
    if thinpb,
        bmap = double(bwmorph(bmap, 'thin', inf));
    end
    
    % compute the correspondence
    [match1, match2] = correspondPixels(bmap, double(bdry), maxDist);
    num_match1 = sum(match1(:)>0);
    num_match2 = sum(match2(:)>0);
    assert(num_match1==num_match2, 'Output match numbers not equal!');
    assert(num_match1>=0, 'Match # can not be negative!');
    assert(num_match1<=sum(bmap(:)), 'Match # larger than true positive!');
    assert(num_match2<=sum(bdry(:)), 'Match # larger than predicted positive!');

    % compute recall
    sumR(t) = sumR(t) + sum(bdry(:));
    cntR(t) = cntR(t) + num_match2;
    
    % compute precision
    sumP(t) = sumP(t) + sum(bmap(:));
    cntP(t) = cntP(t) + num_match1;
end
