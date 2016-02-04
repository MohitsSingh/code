function showDetections(conf,images,detections)

    bb = {};
    ddd = [];
    for k = 1:length(detections)
        d = detections{k}.bbs{1};
        d(:,11) = k;
        bb{k} = d;
    end
    bb(cellfun(@isempty,bb)) = [];
    bb = cat(1,bb{:});
    m = multiCrop(conf,images,bb);

    for k = 1:length(images)
        I = images{k};
       	if (ischar(I)), I = imread(I); end
        clf; imagesc(I); axis image; hold on;
        d = detections{k};
        plotBoxes(d.bbs{1},'g','LineWidth',2);pause;
    end
end