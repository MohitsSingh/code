function [ pos_imgs ] = getPosSubImgs(VOCopts, train_gt,box_data,debug_)

if (nargin < 4)
    debug_ = false;
end
pos_imgs = {};

for t = 1:length(box_data)
    t/length(box_data)
    k = box_data(t).image_ind;
    I = imread(sprintf(VOCopts.imgpath,train_gt(k).filename(1:end-4)));
    curBoxes = box_data(t).boxes(~box_data(t).isDifficult,:);
    curImgs = [];
    if (any(curBoxes))
        curImgs = multiCrop2(I,curBoxes);
        pos_imgs = [pos_imgs,curImgs];
    end
    if (debug_)
        clf; subplot(1,2,1); imagesc2(I); hold on;
        plotBoxes(curBoxes);
        if (~isempty(curImgs))
            m = mImage(curImgs);
            subplot(1,2,2); imagesc2(m);
        end
        %         drawnow; pause
    end
end

%GETPOSSUBIMGS Summary of this function goes here
%   Detailed explanation goes here


end

