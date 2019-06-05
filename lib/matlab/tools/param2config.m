function config = param2config(prefix, param)
config = [prefix '_' param.loss_type];
if(~(param.sigma_x==0 || param.sigma_y==0))
    str1 = ['_sx' num2str(param.sigma_x) '_sy' num2str(param.sigma_y)];
    if(param.mkv_flag)
        str2 = ['_lda' num2str(param.lambda)];
    else
        str2 = [];
    end
    config = [config str1 str2];
end
