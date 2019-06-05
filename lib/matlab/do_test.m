function do_test(net, param, data_dir, list_dir, result_dir)
%% load file list
s = load(list_dir);
fields = fieldnames(s);
list_test = s.(fields{1});

%% Load color map
load(param.colormap);
colors = 255-colors;

%% Dataset dimensions
test_num = size(list_test, 1);
num_cls = param.num_cls;

%% Image preprocessing parameters
image_mean = param.mean;
crop_size = param.crop_size_test;

%% Obtain file names
file_name = cell(test_num, 1);
for idx_test = 1:test_num
    idx_str = strfind(list_test{idx_test, 1}, '/');
    file_name{idx_test} = list_test{idx_test, 1}(idx_str(end)+1:end-4);
end

%% Reshape input size
net.blobs('data').reshape([crop_size, crop_size, 3, 1]); % reshape blob 'data'
net.reshape();

%% Network testing
for idx_test = 1:test_num
    display(['Predicting on test image ' num2str(idx_test) ', image name: ' file_name{idx_test} '.png'])
    % Load image & random crop
    image = double(imread([data_dir list_test{idx_test}]));
    [height, width, chn] = size(image);
    image_mean_sub = image;
    for i = 1:chn
        image_mean_sub(:,:,i) = image_mean_sub(:,:,i) - image_mean(i);
    end
    pad_y = 0;
    pad_x = 0;
    if(height < crop_size)
        pad_y = crop_size - height;
    end
    if(width < crop_size)
        pad_x = crop_size - width;
    end
    height_pad = height + pad_y;
    width_pad = width + pad_x;
    image_pad = zeros(height_pad, width_pad, chn);
    image_pad(1:height, 1:width, :) = image_mean_sub;
    step_num_y = ceil(height_pad/crop_size);
    step_num_x = ceil(width_pad/crop_size);

    score_pred = zeros(height_pad, width_pad, num_cls);
    mat_count = zeros(height_pad, width_pad, 1);
    for i = 0:step_num_y-1
        offset_y = round((height_pad-crop_size)*(i/((step_num_y-1)+(step_num_y==1))));
        for j = 0:step_num_x-1
            offset_x = round((width_pad-crop_size)*(j/((step_num_x-1)+(step_num_x==1))));

            % Net forward to obtain edge probability
            image_crop = image_pad(offset_y+1:offset_y+crop_size, offset_x+1:offset_x+crop_size, :);
            net.blobs('data').set_data(permute(single(image_crop(:,:,[3,2,1])), [2,1,3]));
            net.forward_prefilled();
            score_pred(offset_y+1:offset_y+crop_size, offset_x+1:offset_x+crop_size, :) =...
                score_pred(offset_y+1:offset_y+crop_size, offset_x+1:offset_x+crop_size, :) +...
                permute(sigmoid(net.blobs('score_ce_fuse').get_data()), [2,1,3]); % Permute from blob format (WxHxCxN) to matlab format (HxWxCxN)
            mat_count(offset_y+1:offset_y+crop_size, offset_x+1:offset_x+crop_size, :) =...
                mat_count(offset_y+1:offset_y+crop_size, offset_x+1:offset_x+crop_size, :) + 1;
        end
    end
    score_pred = score_pred./repmat(mat_count, [1 1 num_cls]);
    score_pred = score_pred(1:height, 1:width, :);

    % Save prediction results
    bdry_vis = zeros(height*width, 3);
    bdry_sum = zeros(height*width, 1);
    bdry_max = zeros(height*width, 1);
    for idx_cls = 1:num_cls
        cls_dir = [result_dir '/class_' num2str(idx_cls, '%03d')];
        if(~exist(cls_dir, 'file'))
            mkdir(cls_dir);
        end
        imwrite(double(score_pred(:,:,idx_cls)), [cls_dir '/' file_name{idx_test} '.png'], 'png');
        bdry_cls = reshape(score_pred(:,:,idx_cls), [height*width, 1]);
        bdry_vis = bdry_vis + double(bdry_cls)*(colors(idx_cls,:)./255);
        bdry_sum = bdry_sum + bdry_cls(:);
        bdry_max = max(bdry_max, bdry_cls);
    end
    idx_bdry = bdry_sum > 0;
    bdry_vis(idx_bdry, :) = bdry_vis(idx_bdry, :)./repmat(bdry_sum(idx_bdry), [1, 3]);
    bdry_vis = bdry_vis.*repmat(bdry_max, [1 3]);
    bdry_vis = 1 - bdry_vis;
    bdry_vis = reshape(bdry_vis, [height, width, 3]);
    vis_dir = [result_dir '/pred_color'];
    if(~exist(vis_dir, 'file'))
        mkdir(vis_dir);
    end
    imwrite(bdry_vis, [vis_dir '/' file_name{idx_test} '.png'], 'png');
end