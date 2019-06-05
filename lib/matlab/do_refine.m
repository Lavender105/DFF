function do_refine(net, param, data_dir, list_dir, result_dir)
%% load file list
s = load(list_dir);
fields = fieldnames(s);
list_train = s.(fields{1});

%% Dataset dimensions
epoch_size = size(list_train, 1);

%% Hyperparameters
sigma_x = param.sigma_x;
sigma_y = param.sigma_y;
max_spatial_cost = param.max_spatial_cost;
outlier_cost = 100*(128 + max_spatial_cost);
mkv_flag = param.mkv_flag;
neigh_size = 0;
lambda = 0;
if(mkv_flag)
    neigh_size = param.neigh_size;
    lambda = param.lambda;
    outlier_cost = 100*(128 + max_spatial_cost + lambda*(2*neigh_size+1)^2*10);
end
image_mean = param.mean;
pred_size = param.crop_size_test;
par_size = param.par_size;

%% Create gt file directory
label_dir = [result_dir '/label_refine'];
vis_dir = [result_dir '/visualize'];
if(~exist(label_dir, 'file'))
    mkdir(label_dir);
end
if(~exist(vis_dir, 'file'))
    mkdir(vis_dir);
end

%% Obtain file names
file_name = cell(epoch_size, 1);
for idx_train = 1:epoch_size
    idx_str = strfind(list_train{idx_train, 1}, '/');
    file_name{idx_train} = list_train{idx_train, 1}(idx_str(end)+1:end-4);
end

%% Reshape input size
num_cls = param.num_cls;
s = load([data_dir list_train{1, 2}]);
names = fieldnames(s);
assert(num_cls == length(s.(names{1})), 'Input number of classes does not match GT!');
net.blobs('data').reshape([pred_size, pred_size, 3, 1]); % reshape blob 'data'
net.reshape();

% Edge probability prediction
num_batch = ceil(epoch_size/par_size);
for idx_batch = 1:num_batch
    disp(['Final alignment: processing batch ' num2str(idx_batch)]);
    % Network forward
    tic
    ps_real = min(idx_batch*par_size, epoch_size) - (idx_batch-1)*par_size; % Compute real parallel size
    data_par = cell(ps_real, 1);
    pred_par = cell(ps_real, 1);
    for idx_par = 1:ps_real
        % Compute global train id
        idx_train = (idx_batch-1)*par_size + idx_par;
        
        % Load image & crop
        image = double(imread([data_dir list_train{idx_train, 1}]));
        [image_crop, ~] = rand_crop_img(image, image_mean, pred_size, false); % Note: Do not flip the image!
        
        % Net forward to obtain edge probability
        if(sigma_x~=0 && sigma_y~=0)
            net.blobs('data').set_data(permute(single(image_crop(:,:,[3,2,1])), [2,1,3]));
            net.forward_prefilled();
            pred_par{idx_par} = permute(sigmoid(net.blobs('score_ce_fuse').get_data()), [2,1,3]); % Permute from blob format (WxHxCxN) to matlab format (HxWxCxN)
        end
        [height, width, ~] = size(image);
        data_par{idx_par} = image;
        assert(size(pred_par{idx_par}, 1)>=height && size(pred_par{idx_par}, 2)>=width, 'Prediction size must be equal or larger than input image size.');
        pred_par{idx_par} = pred_par{idx_par}(1:height, 1:width, :);
    end
    time_netfwd = toc;
    
    % Parallel label refinement
    tic
    parfor idx_par = 1:ps_real
        % Compute global train id
        idx_train = (idx_batch-1)*par_size + idx_par;
        
        % Obtain cropped images
        mask = true(size(data_par{idx_par}(:,:,1)));
        
        % Obtain cropped label
        s = load([data_dir list_train{idx_train, 2}]);
        names = fieldnames(s);
        gt = s.(names{1});
        
        % Solve the assignment problem
        labelEdge = cell(num_cls, 1);
        if((sigma_x==0 || sigma_y==0))
            for idx_cls = 1:num_cls
                labelEdge{idx_cls} = sparse(gt{idx_cls});
            end
        else
            for idx_cls = 1:num_cls
                if(sum(gt{idx_cls}(:)) ~= 0)
                    edge_gt_chn = full(gt{idx_cls});
                    
                    % Solve bdry alignment
                    gt_tan = bdry_tan(edge_gt_chn, 5);
                    [match1, match2] = solve_assign(double(mask), double(edge_gt_chn), double(pred_par{idx_par}(:,:,idx_cls)), gt_tan, sigma_x, sigma_y, max_spatial_cost, outlier_cost);
                    if(mkv_flag)
                        [~, match2] = solve_assign_pair(double(mask), double(edge_gt_chn), double(pred_par{idx_par}(:,:,idx_cls)), gt_tan, match2, sigma_x, sigma_y, max_spatial_cost, outlier_cost, neigh_size, lambda);
                        [match1, ~] = solve_assign_pair(double(mask), double(edge_gt_chn), double(pred_par{idx_par}(:,:,idx_cls)), gt_tan, match2, sigma_x, sigma_y, max_spatial_cost, outlier_cost, neigh_size, lambda);
                    end
                    edge_refine = logical(match1);
                    
                    % Visualize bdry alignment results
                    edge_vis = data_par{idx_par}./255;
                    [height, width, chn] = size(edge_vis);
                    edge_vis = reshape(edge_vis, [height*width, chn]);
                    idx_disgard = edge_gt_chn & ~edge_refine;
                    edge_vis(idx_disgard, 1) = 1;
                    edge_vis(idx_disgard, 2:3) = edge_vis(idx_disgard, 2:3).*0.5;
                    edge_vis(edge_refine, 3) = 1;
                    edge_vis(edge_refine, 1:2) = edge_vis(edge_refine, 1:2).*0.5;
                    edge_vis = reshape(edge_vis, [height, width, chn]);
                    imwrite(edge_vis, [vis_dir '/' file_name{idx_train} '_cls_' num2str(idx_cls, '%02d') '.png'], 'png');
                    
                    labelEdge{idx_cls} = sparse(edge_refine);
                else
                    labelEdge{idx_cls} = sparse(gt{idx_cls});
                end
            end
        end
        savelabeledge([label_dir '/' file_name{idx_train} '.mat'], labelEdge);
    end
    time_align = toc;
    if(mkv_flag)
        display(['Bdry refinement takes ' num2str(time_netfwd) '+' num2str(time_align) ' seconds (sigma_x: ' num2str(sigma_x) ', sigma_y: ' num2str(sigma_y) ', lambda: ' num2str(lambda) ')'])
    else
        display(['Bdry refinement takes ' num2str(time_netfwd) '+' num2str(time_align) ' seconds (sigma_x: ' num2str(sigma_x) ', sigma_y: ' num2str(sigma_y) ')'])
    end
end


end