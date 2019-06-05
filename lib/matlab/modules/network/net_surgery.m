function net_surgery(solver)
% network surgery
w_fuse = solver.net.params('ce_fusion', 1).get_data();
w_fuse(1,1,1,:) = 1;
solver.net.params('ce_fusion', 1).set_data(w_fuse);
