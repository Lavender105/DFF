% Generates bit-wise labels for Caffe loss input
% input: multi-channel edge or seg map
% mask: valid pixels
function label = gen_label(input, mask)
[height, width] = size(input{1});
chn_input = length(input);
chn_label = ceil((chn_input+1)/32);
label = zeros(height, width, chn_label, 'uint32');
label(:, :, chn_label) = uint32(~mask).*2^31;% ignore certain pixels
for i = 1:chn_input
    chn_quotient = ceil(i/32);
    chn_remainder = mod((i-1), 32);
    edge_out_chn = label(:,:,chn_quotient);
    idx_pos = (input{i} > 0) & mask;
    edge_out_chn(idx_pos) = edge_out_chn(idx_pos) + 2^chn_remainder;
    label(:,:,chn_quotient) = edge_out_chn;
end
label = typecast(label(:), 'single');
label = reshape(label, [height, width, chn_label]);