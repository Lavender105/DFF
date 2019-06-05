function param = arg_parser(param, argin)
assert(length(argin)>=1,...
    'Wrong input argument: argin must contain at least 1 input.');
assert(ischar(argin{1}) && ismember(argin{1}, {'reweight', 'unweight'}),...
    'Wrong input argument: argin{1}(loss_type) must be one of {reweight, unweight}');
param.loss_type = argin{1};
if(length(argin)==1)
    param.sigma_x = 0;
    param.sigma_y = 0;
    param.mkv_flag = false;
elseif(length(argin)>=3)
    assert(isreal(argin{2}) && isreal(argin{3}) && argin{2}>=0 && argin{3}>=0,...
        'Wrong input argument: argin{2}(sigma_x) and argin{3}(sigma_y) must be nonnegative real numbers.')
    if(argin{2}>0 && argin{3}>0)
        param.sigma_x = argin{2};
        param.sigma_y = argin{3};
    else
        param.sigma_x = 0;
        param.sigma_y = 0;
    end
    if(length(argin)==3)
        param.mkv_flag = false;
    elseif(length(argin)==4)
        assert(isreal(argin{4}) && argin{4}>=0,...
            'Wrong input argument: argin{4}(lambda) must be a nonnegative real.')
        if(argin{2}>0 && argin{3}>0 && argin{4}>0)
            param.mkv_flag = true;
            param.lambda = argin{4};
        else
            param.mkv_flag = false;
        end
    else
        error('Wrong input format!')
    end
else
    error('Wrong input format!')
end