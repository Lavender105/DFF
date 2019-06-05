% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This script is used to:
% 1. Serve as the top-level function for result evaluation
% 2. Set evaluation parameters and directories
%
% Input arguments:
%   file_list   : Name list of the files to be evaluated
%   gt_dir      : Directory containing ground truths
%   eval_dir    : Directory containing predictions to be evaluated
%   result_dir  : Directory to put evaluation results
%   categories  : List of semantic category names
%   [margin]    : Size of margin to be ignored
%   [nthresh]   : Number of points in PR curve
%   [MaxDist]   : Edge misalignment tolerance threshold
%   [thinpb]    : Option to apply morphological thinning on evaluated boundaries
% --------------------------------------------------------

function evaluation(file_list, gt_dir, eval_dir, result_dir, categories, margin, nthresh, thinpb, maxDist)

if(nargin<6), margin = 0; end;
if(nargin<7), nthresh = 99; end;
if(nargin<8), thinpb = true; end;
if(nargin<9), maxDist = 0.0075; end;
assert(iscell(eval_dir), 'eval_dir must be a cell array!')
assert(iscell(result_dir), 'result_dir must be a cell array!')
assert(length(eval_dir) == length(result_dir), 'size of eval_dir and result_dir must be equal!')

%% Setup Parallel Pool
num_worker = 6; %12;
delete(gcp('nocreate'));
parpool('local', num_worker);

%% Load the evaluation file list
s = load(file_list);
names = fieldnames(s);
list_eval = s.(names{1});
% list_eval = {'frankfurt_000000_000294_leftImg8bit'} %hy added to debug

%% Perform evaluation
for idx_dir = 1:length(eval_dir)
    % Benchmark each category
    if(exist(result_dir{idx_dir}, 'file')==0)
        mkdir(result_dir{idx_dir});
    end
    num_cls = length(categories);
    for idx_cls = 1:num_cls %%1
        fprintf('Benchmarking boundaries for category %d: %s\n', idx_cls, categories{idx_cls});
        result_cls = benchmark_category(list_eval, eval_dir{idx_dir}, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist);
        save([result_dir{idx_dir} '/class_' num2str(idx_cls) '.mat'], 'result_cls');
    end
    
    % Summarize evaluation results
    result_list = dir([result_dir{idx_dir} '/class_*.mat']);
    num_result = size(result_list, 1);
    result_f = zeros(num_cls, 1);
    for idx_result = 1:num_result
        result_name = result_list(idx_result, 1).name;
        idx = find(result_name=='_', 1, 'last');
        idx_cls = str2double(result_name(idx+1:end-4));
        load([result_dir{idx_dir} '/class_' num2str(idx_cls)])
        result_f(idx_cls) = result_cls{2, 1}(4);
    end
    fprintf('====================== Summary MF-ODS ======================\n\n');
    for idx_cls = 1:num_cls
        fprintf('%2d %14s:  %.3f\n', idx_cls, categories{idx_cls}, result_f(idx_cls));
    end
    fprintf('\n      Mean MF-ODS:  %.3f\n\n', mean(result_f));
end

end