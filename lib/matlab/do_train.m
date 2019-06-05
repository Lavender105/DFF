function do_train(solver, param, data_dir, list_dir, result_dir)
%% load file list
s = load(list_dir);
fields = fieldnames(s);
list_train = s.(fields{1});

%% Dataset dimensions
epoch_size = size(list_train, 1);

%% Hyperparameters
iter_num = param.iter_num;
iter_size = param.solver.iter_size;
mirror = param.mirror;
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
vis_align = param.vis_align;
image_mean = param.mean;
crop_size = param.crop_size_train;
par_size = param.par_size;

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
chn_label = ceil((num_cls+1)/32);
solver.net.blobs('data').reshape([crop_size, crop_size, 3, 1]); % reshape blob 'data'
solver.net.blobs('label').reshape([crop_size, crop_size, chn_label, 1]); % reshape blob 'label'
solver.net.reshape();

%% Network training
if(param.resume)
    assert(par_size == param.state.par_size,...
        'Current par_size should match the par_size of resume model!');
    epoch = param.state.epoch;
    idx_iter = param.state.idx_iter;
else
    epoch = 0;
    idx_iter = 0;
end
epoch_train = 0;
batch_train = 0;
update_train = 0;
state.epoch = 0;
state.idxRand = [];
state.idx_iter = 0;
state.idx_batch = 1;
state.idx_update = 1;
state.par_size = par_size; % Store for sanity check when resume
flag_stop = false;
solver.IterStart();
while(1)
    % Create directories
    vis_dir = [result_dir '/epoch_' num2str(epoch)];
    if(sigma_x~=0 && sigma_y~=0)
        if(~exist(vis_dir, 'file'))
            mkdir(vis_dir);
        end
    end
    
    % Random shuffling
    if(epoch_train==0 && param.resume)
        idxRand = param.state.idxRand;
    else
        idxRand = randperm(epoch_size);
    end
    
    % Simultaneous edge alignment and learning
    num_batch = ceil(epoch_size/par_size);
    
    if(batch_train==0 && param.resume)
        batch_start = param.state.idx_batch;
    else
        batch_start = 1;
    end
    for idx_batch = batch_start:num_batch
        skip_align = (epoch==0 && idx_batch==1);
        
        % Random image and label cropping
        tic
        ps_real = min(idx_batch*par_size, epoch_size) - (idx_batch-1)*par_size; % Compute real parallel size
        data_par = cell(ps_real, 1);
        pred_par = cell(ps_real, 1);
        param_par = cell(ps_real, 1);
        parfor idx_par = 1:ps_real
            % Compute global train id
            idx_train = (idx_batch-1)*par_size + idx_par;
            
            % Load image & random crop
            image = imread([data_dir list_train{idxRand(idx_train), 1}]);
            [image_crop, param_par{idx_par}] = rand_crop_img(image, image_mean, crop_size, mirror);
            
            % Label transform & permutation
            data_par{idx_par} = permute(single(image_crop(:,:,[3,2,1])), [2,1,3]); % Transform to BGR & permute to blob format (WxHxCxN)
        end

        % Network forward
        for idx_par = 1:ps_real
            % Net forward to obtain edge probability
            if(~(skip_align || (sigma_x==0 || sigma_y==0)))
                solver.net.blobs('data').set_data(data_par{idx_par});
                solver.net.forward_prefilled();
                pred_par{idx_par} = permute(sigmoid(solver.net.blobs('score_ce_fuse').get_data()), [2,1,3]); % Permute from blob format (WxHxCxN) to matlab format (HxWxCxN)
            end
        end
        time_netfwd = toc;
        
        % Parallel label refinement
        tic
        label_par = cell(ps_real, 1);
        parfor idx_par = 1:ps_real
            % Compute global train id
            idx_train = (idx_batch-1)*par_size + idx_par;
            
            % Obtain cropped images
            image_crop = permute(data_par{idx_par}, [2,1,3]);
            image_crop = image_crop(:,:,[3,2,1]);
            
            % Obtain cropped label
            s = load([data_dir list_train{idxRand(idx_train), 2}]);
            names = fieldnames(s);
            [gt_crop, mask] = crop_label(s.(names{1}), param_par{idx_par});
            
            % Solve the edge alignment problem
            gt_refine = cell(num_cls, 1);
            if(sigma_x==0 || sigma_y==0)
                gt_refine = gt_crop;
            else
                if(skip_align)
                    gt_refine = gt_crop;
                    for idx_cls = 1:num_cls
                        if(sum(gt_crop{idx_cls}(:)) ~= 0)
                            % Visualize bdry alignment results
                            if(vis_align)
                                if(mod(idxRand(idx_train), 20) == 0) % Perform visualization for every 1 out of 20 images
                                    edge_gt_chn = full(gt_crop{idx_cls});
                                    edge_refine = gt_refine{idx_cls};
                                    edge_vis = (image_crop + reshape(ones(numel(edge_gt_chn), 1)*image_mean, [size(edge_gt_chn), 3]))./255;
                                    [height, width, chn] = size(edge_vis);
                                    edge_vis = reshape(edge_vis, [height*width, chn]);
                                    idx_disgard = edge_gt_chn & ~edge_refine;
                                    edge_vis(idx_disgard, 1) = 1;
                                    edge_vis(idx_disgard, 2:3) = edge_vis(idx_disgard, 2:3).*0.5;
                                    edge_vis(edge_refine, 3) = 1;
                                    edge_vis(edge_refine, 1:2) = edge_vis(edge_refine, 1:2).*0.5;
                                    edge_vis = reshape(edge_vis, [height, width, chn]);
                                    imwrite(edge_vis, [vis_dir '/' file_name{idxRand(idx_train)} '_cls_' num2str(idx_cls, '%02d') '.png'], 'png');
                                end
                            end
                        end
                    end
                else
                    for idx_cls = 1:num_cls
                        if(sum(gt_crop{idx_cls}(:)) ~= 0)
                            % Solve bdry alignment
                            edge_gt_chn = gt_crop{idx_cls};
                            gt_tan = bdry_tan(edge_gt_chn, 5);
                            [match1, match2] = solve_assign(double(mask), double(edge_gt_chn), double(pred_par{idx_par}(:,:,idx_cls)), gt_tan, sigma_x, sigma_y, max_spatial_cost, outlier_cost);
                            if(mkv_flag)
                                [~, match2] = solve_assign_pair(double(mask), double(edge_gt_chn), double(pred_par{idx_par}(:,:,idx_cls)), gt_tan, match2, sigma_x, sigma_y, max_spatial_cost, outlier_cost, neigh_size, lambda);
                                [match1, ~] = solve_assign_pair(double(mask), double(edge_gt_chn), double(pred_par{idx_par}(:,:,idx_cls)), gt_tan, match2, sigma_x, sigma_y, max_spatial_cost, outlier_cost, neigh_size, lambda);
                            end
                            edge_refine = logical(match1);
                            gt_refine{idx_cls} = edge_refine;
                            
                            % Visualize bdry alignment results
                            if(vis_align)
                                if(mod(idxRand(idx_train), 20) == 0) % Perform visualization for every 1 out of 20 images
                                    edge_vis = (image_crop + reshape(ones(numel(edge_gt_chn), 1)*image_mean, [size(edge_gt_chn), 3]))./255;
                                    [height, width, chn] = size(edge_vis);
                                    edge_vis = reshape(edge_vis, [height*width, chn]);
                                    idx_disgard = edge_gt_chn & ~edge_refine;
                                    edge_vis(idx_disgard, 1) = 1;
                                    edge_vis(idx_disgard, 2:3) = edge_vis(idx_disgard, 2:3).*0.5;
                                    edge_vis(edge_refine, 3) = 1;
                                    edge_vis(edge_refine, 1:2) = edge_vis(edge_refine, 1:2).*0.5;
                                    edge_vis = reshape(edge_vis, [height, width, chn]);
                                    imwrite(edge_vis, [vis_dir '/' file_name{idxRand(idx_train)} '_cls_' num2str(idx_cls, '%02d') '.png'], 'png');
                                end
                            end
                        else
                            gt_refine{idx_cls} = gt_crop{idx_cls};
                        end
                    end
                end
            end
            label_par{idx_par} = permute(gen_label(gt_refine, mask), [2,1,3]); % Generate bit labels and permute to blob format (WxHxCxN)
        end
        time_align = toc;
        if(mkv_flag)
            display(['Bdry refinement takes ' num2str(time_netfwd) '+' num2str(time_align) ' seconds (sigma_x: ' num2str(sigma_x) ', sigma_y: ' num2str(sigma_y) ', lambda: ' num2str(lambda) ')'])
        else
            display(['Bdry refinement takes ' num2str(time_netfwd) '+' num2str(time_align) ' seconds (sigma_x: ' num2str(sigma_x) ', sigma_y: ' num2str(sigma_y) ')'])
        end
        
        % Network parameter learning
        num_update = ceil(ps_real/iter_size);
        update_size = ps_real/num_update;
        if(update_train==0 && param.resume)
            update_start = param.state.idx_update;
        else
            update_start = 1;
        end
        for idx_update = update_start:num_update
            solver.IterInit();
            for idx_par = floor((idx_update-1)*update_size)+1 : floor(idx_update*update_size)
                % Set input blobs
                solver.net.blobs('data').set_data(data_par{idx_par});
                solver.net.blobs('label').set_data(label_par{idx_par});
                % iter solver
                solver.IterRun();
            end
            solver.IterUpdate();
            
            % Increase the number of steps (iter/update) by 1
            idx_iter = idx_iter + 1;
            update_train = update_train + 1;
            
            % If model snapshot is triggered, save the training state of
            % next step for resumption
            if(mod(idx_iter, param.solver.snapshot) == 0)
                % Save the idx_iter of next step
                state.idx_iter = idx_iter;
                
                % Check whether snapshot happened exactly at the last model
                % update within current batch
                flag_update_lst = false;
                if(idx_update+1>num_update)
                    % If yes, set the model update state to the beginning
                    % for next batch
                    state.idx_update = 1;
                    flag_update_lst = true;
                else
                    % If no, increase the model update state by 1
                    state.idx_update = idx_update+1;
                end
                
                % Check whether the last model update happened
                flag_batch_lst = false;
                if(flag_update_lst)
                    % If yes, need to increase the batch state by 1. Check
                    % whether snapshot happened exactly at the last batch
                    % within current epoch
                    if(idx_batch+1>num_batch)
                        % If yes, set the batch state to the beginning for
                        % next epoch
                        state.idx_batch = 1;
                        flag_batch_lst = true;
                    else
                        % If no, increase the batch state by 1
                        state.idx_batch = idx_batch+1;
                    end
                else
                    % If no, stay within the current batch and epoch
                    state.idx_batch = idx_batch;
                end
                
                % Check whether the last batch happened
                if(flag_batch_lst)
                    % If yes, need to increase the epoch state by 1 and
                    % reshuffle the epoch
                    state.idxRand = randperm(epoch_size);
                    state.epoch = epoch + 1;
                else
                    % If no, stay within current epoch
                    state.idxRand = idxRand;
                    state.epoch = epoch;
                end
                save([param.solver.snapshot_prefix(2:end-1) '_iter_' num2str(idx_iter) '.mat'], 'state');
            end
            
            % If reaching the maximum iter number, stop training  
            if(idx_iter>=iter_num)
                flag_stop = true;
                break;
            end
        end
        if(flag_stop)
            break;
        end
        batch_train = batch_train + 1;
    end
    if(flag_stop)
        break;
    end
    epoch = epoch+1;
    epoch_train = epoch_train+1;
end