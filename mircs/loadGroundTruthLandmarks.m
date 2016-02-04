function [goods,gt_mouth_corners] = loadGroundTruthLandmarks(dbPath,image_data)

goods = false(size(image_data));
gt_mouth_corners = cell(size(image_data));
for t = 1:length(image_data)
    annoPath = j2m(dbPath,image_data(t));
    try
        load(annoPath);
        if ~curLandmarks.MouthLeftCorner.skipped && ~curLandmarks.MouthRightCorner.skipped
            gt_mouth_corners{t} = [curLandmarks.MouthLeftCorner.pts;curLandmarks.MouthRightCorner.pts];
            goods(t) = true;
        end
    catch e % skip missing file...
    end
end