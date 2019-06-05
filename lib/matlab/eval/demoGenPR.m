clc; clear; close all;
path = genpath('../');
addpath(path)

%% Collect and plot Cityscapes results
categories = categories_city();
% Original GT (Thin)
result_dir = {'../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_thin/dff';...
              '../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_thin/casenet'};
plot_pr(result_dir, {'DFF'; 'CASENet'}, '../../../exper/cityscapes/result/evaluation/test/gt_thin/pr_curve', categories, false);
% Original GT (Raw)
result_dir = {'../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_raw/dff';...
              '../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_raw/casenet'};
plot_pr(result_dir, {'DFF'; 'CASENet'}, '../../../exper/cityscapes/result/evaluation/test/gt_raw/pr_curve', categories, false);

%% Collect and plot SBD results
categories = categories_sbd();
% Original GT (Thin)
result_dir = {'../../../exps/sbd/result/evaluation/test/inst/gt_orig_thin/dff';...
              '../../../exps/sbd/result/evaluation/test/inst/gt_orig_thin/casenet'};
plot_pr(result_dir, {'DFF'; 'CASENet'}, '../../../exper/sbd/result/evaluation/test/inst/gt_orig_thin/pr_curve', categories, false);
% Original GT (Raw)
result_dir = {'../../../exps/sbd/result/evaluation/test/inst/gt_orig_raw/dff';...
              '../../../exps/sbd/result/evaluation/test/inst/gt_orig_raw/casenet'};
plot_pr(result_dir, {'DFF'; 'CASENet'}, '../../../exper/sbd/result/evaluation/test/inst/gt_orig_raw/pr_curve', categories, false);
