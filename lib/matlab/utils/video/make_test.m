function make_test(img_root, img_list_dir, gt_dir, pred_dir1, pred_dir2, vid_dir, vid_name, frame_rate, quality)
    if(~exist(vid_dir, 'file'))
        mkdir(vid_dir);
    end
    s = load(img_list_dir);
    fields = fieldnames(s);
    img_list = s.(fields{1});
    gt_list = dir([gt_dir '/*.png']);
    pred_list1 = dir([pred_dir1 '/*fuse.png']);
    pred_list2 = dir([pred_dir2 '/*fuse.png']);
    assert(size(img_list, 1)==length(gt_list) && size(img_list, 1)==length(pred_list1) &&...
           size(img_list, 1)==length(pred_list2), 'Number of images mismatch!');
    num_frame = length(img_list);
    writerObj = VideoWriter([vid_dir '/' vid_name]);
    writerObj.FrameRate = frame_rate;
    writerObj.Quality = quality;
    open(writerObj);
    for idx_frame = 1:num_frame
        disp(['Processing frame: ' num2str(idx_frame)]);
        img = imresize(imread([img_root '/' img_list{idx_frame, 1}]), 0.5);
        gt = imresize(imread([gt_dir '/' gt_list(idx_frame).name]), 0.5);
        pred1 = imresize(imread([pred_dir1 '/' pred_list1(idx_frame).name]), 0.5);
        pred2 = imresize(imread([pred_dir2 '/' pred_list2(idx_frame).name]), 0.5);
        img = insertText(img, [12 6], 'Input', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        gt = insertText(gt, [12 6], 'GT', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        pred1 = insertText(pred1, [12 6], 'CASENet(reproduced)', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        pred2 = insertText(pred2, [12 6], 'DFF', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        frame1 = horzcat(img, gt);
        frame2 = horzcat(pred1, pred2);
        frame = vertcat(frame1, frame2);
        writeVideo(writerObj, frame);
    end
    close(writerObj);
end