% This function estimates the boundary tangent orientation and outputs the
% angle between boundary tangent and x-axis
function bdry_tan = bdry_tan(bdry, k_size)
bdry = logical(bdry);
[height, width] = size(bdry);
bdry_tan = zeros(height, width);
bdry_thin = bwmorph(bdry, 'thin', 'inf');
[Y_Bd, X_Bd] = ind2sub([height, width], find(bdry_thin(:)));
X_Min = max(min(X_Bd-k_size, width), 1);
X_Max = max(min(X_Bd+k_size, width), 1);
Y_Min = max(min(Y_Bd-k_size, height), 1);
Y_Max = max(min(Y_Bd+k_size, height), 1);
num_bdry = length(X_Bd);
theta = zeros(num_bdry, 1);
for i = 1:num_bdry
    bdry_patch = bdry_thin(Y_Min(i):Y_Max(i), X_Min(i):X_Max(i));
    y_orig = Y_Bd(i)-Y_Min(i)+1;
    x_orig = X_Bd(i)-X_Min(i)+1;
    D_Geo = bwdistgeodesic(bdry_patch, x_orig, y_orig);
    idx_neigh = D_Geo>=1 & D_Geo<=k_size;
    [y_neigh, x_neigh] = ind2sub(size(bdry_patch), find(idx_neigh(:)));
    vec_neigh = [y_orig-y_neigh, x_neigh-x_orig]; % Note y is flipped
    vec_neigh_norm = vec_neigh./repmat(sqrt(sum(vec_neigh.^2, 2)), [1 2]);
    num_neigh = length(y_neigh);
    vec_norm = sum(vec_neigh_norm, 1)./num_neigh;
    if(num_neigh>=2*(k_size-1) && norm(vec_norm)>0.5)
        % sharp turn cases
        vec_tan = [-vec_norm(2), vec_norm(1)];
    else
        % all other cases
        cov_mat = vec_neigh'*vec_neigh;
        [V,D] = eig(cov_mat);
        D_Diag = diag(D);
        [~,idx_sort] = sort(D_Diag,'descend');
        V_Sort=V(:,idx_sort);
        vec_tan = V_Sort(:,1);
    end
    theta(i) = atan(vec_tan(1)/vec_tan(2));
end
bdry_tan(bdry_thin) = theta;

tan_sum = zeros(height, width);
tan_weight = zeros(height, width);
for dy = -5:5
    Y_Neigh = Y_Bd + dy;
    idx_valid_y = Y_Neigh>=1 & Y_Neigh<=height;
    for dx = -5:5
        X_Neigh = X_Bd + dx;
        idx_valid_x = X_Neigh>=1 & X_Neigh<=width;
        idx_valid = find(idx_valid_y & idx_valid_x);
        idx_neigh = sub2ind([height, width], Y_Neigh(idx_valid), X_Neigh(idx_valid));
        weight = exp(-(dy^2+dx^2)/(2*4));
        idx_first = tan_weight(idx_neigh) == 0;
        idx_exist = find(~idx_first);
        tan_sum(idx_neigh(idx_first)) = weight.*theta(idx_valid(idx_first));
        if(~isempty(idx_exist))
            tan_ref = tan_sum(idx_neigh(idx_exist))./tan_weight(idx_neigh(idx_exist));
            idx_flip1 = tan_ref-theta(idx_valid(idx_exist)) > pi/2;
            idx_flip2 = tan_ref-theta(idx_valid(idx_exist)) < -pi/2;
            idx_noflip = ~(idx_flip1|idx_flip2);
            tan_sum(idx_neigh(idx_exist(idx_flip1))) = tan_sum(idx_neigh(idx_exist(idx_flip1))) + weight.*(theta(idx_valid(idx_exist(idx_flip1)))+pi);
            tan_sum(idx_neigh(idx_exist(idx_flip2))) = tan_sum(idx_neigh(idx_exist(idx_flip2))) + weight.*(theta(idx_valid(idx_exist(idx_flip2)))-pi);
            tan_sum(idx_neigh(idx_exist(idx_noflip))) = tan_sum(idx_neigh(idx_exist(idx_noflip))) + weight.*theta(idx_valid(idx_exist(idx_noflip)));
        end
        tan_weight(idx_neigh) = tan_weight(idx_neigh) + weight;
    end
end
idx_weight = tan_weight>0;
tan_sum(idx_weight) = tan_sum(idx_weight)./tan_weight(idx_weight);
bdry_tan(bdry) = tan_sum(bdry);
idx_out1 = bdry_tan>pi/2;
idx_out2 = bdry_tan<-pi/2;
bdry_tan(idx_out1) = bdry_tan(idx_out1) - ceil((bdry_tan(idx_out1)-pi/2)/pi).*pi;
bdry_tan(idx_out2) = bdry_tan(idx_out2) + ceil((-bdry_tan(idx_out2)-pi/2)/pi).*pi;

% bdry_diff = bdry&(~bdry_thin);
% [Y_Diff, X_Diff] = ind2sub([height, width], find(bdry_diff(:)));
% idx_nn = knnsearch([Y_Bd,X_Bd], [Y_Diff,X_Diff]);
% bdry_tan(bdry_diff) = theta(idx_nn);
