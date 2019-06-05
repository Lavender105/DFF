function [image_crop, crop_param] = rand_crop_img(image_in, image_mean, crop_size, mirror)
% image_crop: randomly cropped image

% Transform from uint8 to double
image_in = double(image_in);

% Get the input dimensions
[height_img, width_img, chn_img] = size(image_in);

% Perform random mirror
flag_mirror = 0;
if(mirror)
    if(rand(1)>0.5)
        flag_mirror = 1;
        image_in = image_in(:, (width_img+1)-(1:width_img), :);
    end
end

% Subtract the image mean
if(chn_img~=length(image_mean))
    error('Dimension mismatch between image_in channel and image_in mean!')
end
for i = 1:chn_img
    image_in(:,:,i) = image_in(:,:,i) - image_mean(i);
end

% Randomly crop the image
if(size(crop_size,1)==1 && size(crop_size,2)==2)
    height_crop = crop_size(1);
    width_crop = crop_size(2);
elseif(size(crop_size,1)==1 && size(crop_size,2)==1)
    height_crop = crop_size;
    width_crop = crop_size;
else
    error('Invalid input crop size!')
end
height_min = min([height_img, height_crop]);% Height of overlapping area
width_min = min([width_img, width_crop]);% Width of overalpping area
offset_h = randi([0, max(0, (height_img-height_crop))]);% Random y offset
offset_w = randi([0, max(0, (width_img-width_crop))]);% Random x offset

% Output the cropped image
image_crop = zeros(height_crop, width_crop, chn_img);
image_crop(1:height_min, 1:width_min, :) = image_in(offset_h+1:offset_h+height_min, offset_w+1:offset_w+width_min, :);

% Output the cropping parameters
crop_param = [offset_h, offset_w, height_min, width_min, crop_size, flag_mirror];