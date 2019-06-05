function make_clswise(img_dir, result_dir, vid_dir, vid_name, frame_rate, quality)
if(~exist(vid_dir, 'file'))
    mkdir(vid_dir);
end
img_list = dir([img_dir '/*.png']);
folder_list = dir([result_dir '/class_*']);
class_list = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic Light',...
              'Traffic Sign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider',...
              'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'};
pred_list_set = cell(length(folder_list), 1);
for idx_cls = 1:length(folder_list)
    pred_list_set{idx_cls} = dir([result_dir '/' folder_list(idx_cls).name '/*.png']);
    assert(length(img_list)==length(pred_list_set{idx_cls}), 'Number of images mismatch!');
end
writerObj = VideoWriter([vid_dir '/' vid_name]);
writerObj.FrameRate = frame_rate;
writerObj.Quality = quality;
open(writerObj);
num_frame = length(img_list);
for idx_frame = 1:num_frame
    disp(['Processing frame: ' num2str(idx_frame)]);
    frame_set = cell(20, 1);
    frame_set{1} = imresize(imread([img_dir '/' img_list(idx_frame).name]), 0.25);
    frame_set{1} = insertText(frame_set{1}, [6 3], 'Input', 'FontSize', 15, 'TextColor', [0 255 0], 'BoxOpacity', 0);
    for idx_cls=1:length(folder_list)
        frame_set{idx_cls+1} = imresize(imread([result_dir '/' folder_list(idx_cls).name '/' pred_list_set{idx_cls}(idx_frame).name]), 0.25);
        frame_set{idx_cls+1} = repmat(frame_set{idx_cls+1}, [1 1 3]);
        frame_set{idx_cls+1} = im2uint8(frame_set{idx_cls+1}); % hy added
        frame_set{idx_cls+1} = insertText(frame_set{idx_cls+1}, [6 3], class_list{idx_cls}, 'FontSize', 15, 'TextColor', [0 255 0], 'BoxOpacity', 0);
    end
    frame1 = horzcat(frame_set{1}, frame_set{2}, frame_set{3}, frame_set{4});
    frame2 = horzcat(frame_set{5}, frame_set{6}, frame_set{7}, frame_set{8});
    frame3 = horzcat(frame_set{9}, frame_set{10}, frame_set{11}, frame_set{12});
    frame4 = horzcat(frame_set{13}, frame_set{14}, frame_set{15}, frame_set{16});
    frame5 = horzcat(frame_set{17}, frame_set{18}, frame_set{19}, frame_set{20});
    frame = vertcat(frame1, frame2, frame3, frame4, frame5);
    writeVideo(writerObj, frame);
end
close(writerObj);
end