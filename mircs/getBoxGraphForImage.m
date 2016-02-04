function [boxes,boxGraph] = getBoxGraphForImage(imgData,I,gt_graph,boxes,opts)
% actually, sampling a chain.
gt_boxes = cellfun3(@(x) x.bbox,gt_graph);

% check overlap of each box with each g.t. box.
anchorBox = gt_boxes(1,:);
%
% find boxes "touching" the anchor box
boxes = boxes(:,1:4);
face_in_boxes = BoxIntersection2(boxes, imgData.faceBox);
[~,~,faceArea] = BoxSize(imgData.faceBox);
size(boxes,1)

boxes(face_in_boxes./faceArea > .4,:) = [];
size(boxes,1)
[~,~,bArea] = BoxSize(boxes);
boxes(bArea./ prod(size2(I)) > .3,:) = [];
[~,~,bArea] = BoxSize(boxes);
boxes(bArea./ faceArea > 1,:) = [];

sal_opts.show = false;
maxImageSize = 200;
sal_opts.maxImageSize = maxImageSize;
spSize = 100;
sal_opts.pixNumInSP = spSize;
conf.get_full_image = true;
[sal1,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I),sal_opts);
sal1 = imResample(sal1,size2(I),'bilinear');
box_sals = sum_boxes(sal1,boxes);
[~,~,bArea] = BoxSize(boxes);
box_sals = box_sals./bArea;
sal_thresh = .03;
boxes(box_sals<sal_thresh,:) = [];

if opts.donms
    a = nms(boxes,opts.nmsfactor);
    boxes = boxes(a,:);
end
boxes = [anchorBox;boxes];
% do a bfs over the first
boxGraph = buildBoxGraph(boxes);


