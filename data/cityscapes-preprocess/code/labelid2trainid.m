function [trainId] = labelid2trainid(labelId)
[height, width, chn] = size(labelId);
if(chn>1)
    display('Warning! Input label has multiple channels!')
    labelId = labelId(:, :, 1);
end
trainId = 255.*ones(height, width, 'uint8');
map = [7 0; 8 1; 11 2; 12 3; 13 4; 17 5; 19 6; 20 7; 21 8; 22 9; 23 10; 24 11; 25 12; 26 13; 27 14; 28 15; 31 16; 32 17; 33 18];
numCls = size(map, 1);
for idxCls = 1:numCls
    idxLabel = labelId == map(idxCls, 1);
    trainId(idxLabel) = map(idxCls, 2);
end