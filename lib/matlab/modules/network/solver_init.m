function solver = solver_init(gpu_id, solver_prototxt, model_init, resume)
% clear and set device
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

% initialize solver network
solver = caffe.Solver(solver_prototxt);
display('Loading solver done!')

% load model if there is one
if(~isempty(model_init))
    if(resume)
        solver.restore(model_init);
    else
        solver.net.copy_from(model_init);
    end
    display('Loading initialization model done!')
end
