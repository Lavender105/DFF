function visualize_gt(file_dir, list_dir, color_dir, vis_dir)
%% Setup Parallel Pool
num_worker = 6; %12; % Number of matlab workers for parallel computing
delete(gcp('nocreate'));
parpool('local', num_worker);

%% Load Inputs
s = load(list_dir);
names = fieldnames(s);
file_list = s.(names{1});
num_file = size(file_list, 1);
img_list = cell(num_file, 1);
gt_list = cell(num_file, 1);
for idx_file = 1:num_file
    img_list{idx_file} = file_list{idx_file, 1};
    gt_list{idx_file} = file_list{idx_file, 2};
end
s = load(color_dir);
names = fieldnames(s);
colors = s.(names{1});

%% Main Program
if(~exist(vis_dir, 'file'))
    mkdir(vis_dir);
end
parfor_progress(num_file);
parfor idx_gt = 1:num_file %parfor
    fileName = gt_list{idx_gt}(max(strfind(gt_list{idx_gt}, '/'))+1:max(strfind(gt_list{idx_gt}, '.'))-1);
    s = load([file_dir gt_list{idx_gt}]);
    names = fieldnames(s);
    gt = s.(names{1});
    [height, width] = size(gt{1});
    bdry_vis = zeros(height*width, 3);
    bdry_sum = zeros(height*width, 1);
    num_cls = size(gt, 1);
    for idx_cls = 1:num_cls
        bdry_cls = reshape(full(gt{idx_cls, 1}), [height*width, 1]);
        bdry_vis = bdry_vis + double(bdry_cls)*(colors(idx_cls,:)./255);
        bdry_sum = bdry_sum + bdry_cls(:);
    end
    idx_bdry = bdry_sum > 0;
    bdry_vis(idx_bdry, :) = bdry_vis(idx_bdry, :)./repmat(bdry_sum(idx_bdry), [1, 3]);
    bdry_vis(~idx_bdry, :) = 1;
    bdry_vis = reshape(bdry_vis, [height, width, 3]);
    imwrite(bdry_vis, [vis_dir '/' fileName '.png'], 'png');
    parfor_progress();
end
parfor_progress(0);

end