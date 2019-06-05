function make_video(img_dir, pred_dir, vid_dir, vid_name, frame_rate, quality)
    if(~exist(vid_dir, 'file'))
        mkdir(vid_dir);
    end
    img_list = dir([img_dir '/*.png']);
    pred_list = dir([pred_dir '/*fuse.png']);
    assert(length(img_list)==length(pred_list), 'Number of images mismatch!');
    writerObj = VideoWriter([vid_dir '/' vid_name]);
    writerObj.FrameRate = frame_rate;
    writerObj.Quality = quality;
    open(writerObj);
    num_frame = size(pred_list, 1);
    for idx_frame = 1:num_frame
        disp(['Processing frame: ' num2str(idx_frame)]);
        img = imread([img_dir '/' img_list(idx_frame).name]);
        [height, width, ~] = size(img);
        frame = imread([pred_dir '/' pred_list(idx_frame).name]);
        frame(1:0.25*height, 1:0.25*width, :) = imresize(img, 0.25);
        writeVideo(writerObj, frame);
    end
    close(writerObj);
end