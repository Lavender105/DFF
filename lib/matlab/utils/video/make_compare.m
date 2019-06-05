function make_compare(img_dir, pred_dir1, pred_dir2, vid_dir, vid_name, frame_rate, quality)
    if(~exist(vid_dir, 'file'))
        mkdir(vid_dir);
    end
    img_list = dir([img_dir '/*.png']);
    pred_list1 = dir([pred_dir1 '/*fuse.png']);
    pred_list2 = dir([pred_dir2 '/*fuse.png']);
    assert(length(img_list)==length(pred_list1) && length(img_list)==length(pred_list2), 'Number of images mismatch!');
    num_frame = length(img_list);
    writerObj = VideoWriter([vid_dir '/' vid_name]);
    writerObj.FrameRate = frame_rate;
    writerObj.Quality = quality;
    open(writerObj);
    for idx_frame = 1:num_frame
        disp(['Processing frame: ' num2str(idx_frame)]);
        img = imresize(imread([img_dir '/' img_list(idx_frame).name]), 0.5);
        pred1 = imresize(imread([pred_dir1 '/' pred_list1(idx_frame).name]), 0.5);
        pred2 = imresize(imread([pred_dir2 '/' pred_list2(idx_frame).name]), 0.5);
        img = insertText(img, [12 6], 'Input', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        pred1 = insertText(pred1, [12 6], 'CASENet(reproduced)', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        pred2 = insertText(pred2, [12 6], 'DFF', 'FontSize', 30, 'TextColor', [0 0 255], 'BoxOpacity', 0);
        [height, width, chn] = size(img);
        frame = zeros(2*height, 2*width, chn, 'uint8');
        frame(1:height, 0.5*width+1:1.5*width, :) = img;
        frame(height+1:end, :, :) = horzcat(pred1, pred2);

        writeVideo(writerObj, frame);
    end
    close(writerObj);
end