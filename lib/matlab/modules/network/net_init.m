function net = net_init(gpu_id, net_prototxt, model)
% clear and set device
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

% initialize solver network
net = caffe.Net(net_prototxt, 'test');
display('Loading net done!')

% load model if there is one
if(~isempty(model))
    net.copy_from(model);
    display('Loading net model done!')
end
