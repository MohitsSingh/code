
function gt_graph = get_gt_graph(imgData,nodes,params,I)

% get the ground-truth for this image, if available
[xy,goods] = loadKeypointsGroundTruth(imgData,params.requiredKeypoints);
xy = xy(:,1:2);
mouthCenter = xy(3,:);
gt_graph = {};
faceBox = imgData.faceBox;

[~,face_h,face_a] = BoxSize(faceBox);

for iNode = 1:length(nodes)
    curNode = nodes(iNode);
    switch curNode.name
        case 'mouth'
            sz = curNode.spec.size*face_h;
            curNode.bbox = inflatebbox(mouthCenter, sz, 'both', true);
            curNode.poly = box2Pts(curNode.bbox);
            curNode.roiMask = poly2mask2(curNode.poly,size2(I));
        case 'obj'
            if ~(isempty(imgData.objects))
                
                objs = imgData.objects;
                iFace = strcmp({objs.name},'face');
                objs = objs(~iFace);
                if isempty(objs)
                    curNode.bbox = [0 0 0 0];
                    curNode.poly = [0 0 0 0];
                else
                    toKeeps = find([objs.toKeep]);
                    curNode.poly = {objs(toKeeps).poly};
                    roiMask =  poly2mask2(curNode.poly,size2(I));
                    curNode.bbox = region2Box(roiMask);
                    curNode.roiMask = roiMask;
                end
            else
                curNode.bbox = [0 0 0 0];
                curNode.poly = [0 0 0 0];
            end
        case 'hand'
            if (size(imgData.hands,1)==1)
                hands_to_keep = 1;
            else
                hands_to_keep = imgData.hands_to_keep;
            end
            if (~isempty(imgData.hands))
                curNode.bbox = imgData.hands(hands_to_keep,:);
            else
                curNode.bbox = [0 0 0 0];
            end
            curNode.poly = box2Pts(curNode.bbox);
            curNode.roiMask = poly2mask2(curNode.poly,size2(I));
    end
    gt_graph{iNode} = curNode;
end

% fix the gt graph so labels are mutually exclusive
% % origin = boxCenters(gt_graph{1}.bbox);
% % for t = 2:length(gt_graph)-1
% %     curMask = gt_graph{t}.roiMask;
% %     nextMask = gt_graph{t+1}.roiMask;
% %     curMask = curMask & ~ nextMask;
% %     [L,numComponents] = bwlabel(curMask);
% %     if numComponents > 1
% %         stats = regionprops(L,'Centroid');
% %         [~,id] = min(l2(cat(1,stats.Centroid),origin));
% %         curMask = L == id;
% %     end
% %     gt_graph{t}.roiMask = curMask;
% % end
% %
