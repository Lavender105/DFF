% This function takes edge candidates and input segment, producing binary
% bdrys.
function [idxEdge] = seg2edge_fast(seg, edgeCand, radius, labelIgnore, edge_type)
% Get dimensions
[height, width, chn] = size(seg);
if(~isempty(labelIgnore))
    if(chn~=size(labelIgnore, 2))
        error('Channel dimension not matching ignored label dimension!')
    end
end

% Set the considered neighborhood
radius_search = max(ceil(radius), 1);
[X, Y] = meshgrid(1:width, 1:height);
[x, y] = meshgrid(-radius_search:radius_search, -radius_search:radius_search);

% Columnize everything
X = X(:); Y = Y(:);
x = x(:); y = y(:);
edgeCand = edgeCand~=0;
XCand = X(edgeCand);
YCand = Y(edgeCand);
idxCand = find(edgeCand);
if(chn == 1)
    seg = seg(:);
else
    seg = reshape(seg, [height*width chn]);
end

% Build circular neighborhood
idxNeigh = sqrt(x.^2 + y.^2) <= radius;
x = x(idxNeigh); y = y(idxNeigh);
numPxlImg = length(X);
numPxlNeigh = length(x);

% Compute Gaussian weight
idxEdge = false(numPxlImg, 1);
for i = 1:numPxlNeigh
    XNeigh = XCand+x(i);
    YNeigh = YCand+y(i);
    idxSelect = XNeigh >= 1 & XNeigh <= width & YNeigh >=1 & YNeigh <= height;
    idxValid = idxCand(idxSelect);
    
    XCenter = X(idxValid);
    YCenter = Y(idxValid);
    XNeigh = XNeigh(idxSelect);
    YNeigh = YNeigh(idxSelect);
    LCenter = seg(sub2ind([height width], YCenter, XCenter), :);
    LNeigh = seg(sub2ind([height width], YNeigh, XNeigh), :);
    
    if(strcmp(edge_type, 'regular'))
        idxDiff = find(any(LCenter~=LNeigh, 2));
    elseif(strcmp(edge_type, 'inner'))
        idxDiff = find(any(LCenter~=LNeigh, 2) & any(LCenter~=0, 2) & all(LNeigh==0, 2) );
    elseif(strcmp(edge_type, 'outer'))
        idxDiff = find(any(LCenter~=LNeigh, 2) & all(LCenter==0, 2) & any(LNeigh~=0, 2) );
    else
        error('Wrong edge type input!');
    end
    
    LCenterEdge = LCenter(idxDiff, :);
    LNeighEdge = LNeigh(idxDiff, :);
    idxIgnore2 = false(length(idxDiff), 1);
    for j = 1:size(labelIgnore, 1)
        idxIgnore2 = idxIgnore2 | ( all(bsxfun(@eq, LCenterEdge, labelIgnore(j, :)), 2) | all(bsxfun(@eq, LNeighEdge, labelIgnore(j, :)), 2) );
    end
    
    idxDiffGT = idxDiff(~idxIgnore2);
    idxEdge(idxValid(idxDiffGT)) = true;
end
idxEdge = reshape(idxEdge, [height, width]);