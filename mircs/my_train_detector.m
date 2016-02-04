function patchModels = my_train_detector(gtDir,paths,negImagePaths,param)
% Train a sliding window detector
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

[gt_polys,orientations] = load_annotations(gtDir,paths);
windowSize = param.windowSize;
nSamples = length(gt_polys)
pos_images = {};
for iSample = 1:nSamples
    cur_poly = gt_polys{iSample};
    
    % for now, just make it square
    I = imread2(paths{iSample});
% % %     cur_poly = rotate_pts(cur_poly,pi*orientations(iSample)/180,fliplr(size2(I))/2);
% % %     I = imrotate(I,orientations(iSample),'bilinear','crop');
    bb = makeSquare(pts2Box(cur_poly));
    bb = round(inflatebbox(bb,[windowSize windowSize],'both',true));
    pos_images{iSample} = imResample(im2single(cropper(I,bb)),windowSize,'bilinear');
end

patchModels = train_detector(pos_images,negImagePaths,param);%,train_boxes');
%imagesc(vl_hog('render',patchModels))

function [gt_polys,orientations] = load_annotations(gtDir,paths)
gt_polys = cell(size(paths));
orientations = zeros(size(paths));
for t = 1:length(paths)
    curPath = paths{t};
    [~,name,ext] = fileparts(curPath);
    fName = fullfile(gtDir,[name ext  '.txt']);
    [objs,bbs] = bbGt( 'bbLoad', fName);
    [gt_polys{t},orientations(t)] = bbgt_to_poly(objs(1));    
end
