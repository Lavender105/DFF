function [label_crop, mask] = crop_label(label_in, crop_param)

% Get the input dimensions
chn_label = length(label_in);
[height_edge, width_edge] = size(label_in{1});
if(crop_param(3)>height_edge || crop_param(4)>width_edge)
    error('Input crop parameter and label size mismatch!')
end

if(length(crop_param)==7)
    height_crop = crop_param(5);
    width_crop = crop_param(6);
    flag_mirror = crop_param(7);
elseif(length(crop_param)==6)
    height_crop = crop_param(5);
    width_crop = crop_param(5);
    flag_mirror = crop_param(6);
else
    error('Invalid input crop parameters!')
end

% Perform random mirror
if(flag_mirror==1)
    for i = 1:chn_label
        label_in{i} = label_in{i}(:, (width_edge+1)-(1:width_edge));
    end
end

% Randomly crop the label_in and perform bit operation
label_crop = cell(chn_label, 1);
for i = 1:chn_label
    label_crop{i} = false(height_crop, width_crop);
    label_crop{i}(1:crop_param(3), 1:crop_param(4)) = full(label_in{i}(crop_param(1)+1:crop_param(1)+crop_param(3), crop_param(2)+1:crop_param(2)+crop_param(4)));
end

% Compute non-ignore mask
mask = false(height_crop, width_crop);
mask(1:crop_param(3), 1:crop_param(4)) = true;