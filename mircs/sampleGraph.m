function configurations = sampleGraph(imgData,useGT,nodes,edges,params,I,regionSampler,nSamples)
% actually, sampling a chain.
if nargin < 8
    nSamples=1;
end

% get the ground-truth for this image, if available
[xy,goods] = loadKeypointsGroundTruth(imgData,params.requiredKeypoints);
xy = xy(:,1:2);
mouthCenter = xy(3,:);
configuration = {};
faceBox = imgData.faceBox;

% regionSampler.borders = inflatebbox(faceBox,3,'both',false);
[~,face_h,face_a] = BoxSize(faceBox);

% sz = face_h/4;
if useGT
    configurations = {};
    for iNode = 1:length(nodes)
        curNode = nodes(iNode);
        switch curNode.name
            case 'mouth'
                sz = curNode.spec.size*face_h;
                curNode.bbox = inflatebbox(mouthCenter, sz, 'both', true);
                curNode.poly = box2Pts(curNode.bbox);
            case 'obj'
                if ~(isempty(imgData.objects))
                    toKeeps = [imgData.objects.toKeep];
                    roiMask =  poly2mask2({imgData.objects(toKeeps).poly},size2(I));
                    curNode.bbox = region2Box(roiMask);
                    curNode.poly = box2Pts(curNode.bbox);
                else
                    curNode.bbox = [];
                    curNode.poly = [];
                end
            case 'hand'
                if (size(imgData.hands,1)==1)
                    hands_to_keep = 1;
                else
                    hands_to_keep = imgData.hands_to_keep;
                end
                if (~isempty(imgData.hands))
                    curNode.bbox = imgData.hands(hands_to_keep,:);
                    curNode.poly = box2Pts(curNode.bbox);
                end
        end
        configuration{iNode} = curNode;
    end
    configurations = {configuration};
else
    %     regionSampler.minAreaInsideBorders = 1;
    regionSampler.clearRoi();
    regionSampler.borders = [];
    configurations = {};
% % %     cand_boxes = regionSampler.sampleEdgeBoxes(I);
% % %     cand_boxes = cand_boxes(:,1:4);
% % %     [~,~,a] = BoxSize(cand_boxes); % don't allow objects to be larger than
% % %     % twice the head
% % %     %goods = goods & a < 2*face_a;
% % %     cand_boxes(a > 1*face_a,:) = [];
    %thetas = 0:15:360;    
    
    
    theta_start = 0;
    theta_end = theta_start+180;
    b = (theta_end-theta_start)/nSamples;
    %thetas = 0:b:360-b;%0:10:350;
    thetas = theta_start:b:theta_end;%0:10:350;
    theta_order = randperm(length(thetas));

    for z = 1:nSamples
        newConfig = {};
        for iNode = 1:length(nodes)
            curNode = nodes(iNode);
            % still using the ground-truth mouth center
            switch curNode.name
                case 'mouth'
                    sz = curNode.spec.size*face_h;
                    curNode.bbox = inflatebbox(mouthCenter, sz, 'both', true);
                    curNode.poly = box2Pts(curNode.bbox);
                case 'obj'
                    %                 startPt = configuration(iNode-1)%imgData.face_landmarks.xy(3,:);
                    startPt = boxCenters(newConfig{iNode-1}.bbox);
                    avgWidth = curNode.spec.avgWidth*face_h;
                    avgLength = curNode.spec.avgLength*face_h;
%                     thetas = 0:15:360;    
                    
                    pp = theta_order(z);
                    curTheta = thetas(pp);
                    [rois] = hingedSample(startPt,avgWidth,avgLength,curTheta);
                    theta = rois{1}.theta;
                    %                 obj_poly = imgData.gt_obj;
                    %                 gt_region = poly2mask2(obj_poly,size2(I));
                    %                 roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
                    %                 [~,ints,uns] = regionsOverlap(roiMasks,gt_region);
                    %                 [r,ir] = max(ints);
                    curNode.params = rois{1};
                    curNode.poly = rois{1}.xy;
                    curNode.bbox = pts2Box(curNode.poly);
                    
                    %                 curNode.poly = rois{ir};
                    %                     curNode.poly =
                case 'hand'
                    % generate a bounding box around the end of the previous
                    % node
                    % %                     prevNode = newConfig{iNode-1};
                    % %                     bb = prevNode.bbox;
                    % %                     [ovps,ints] = boxesOverlap(cand_boxes,bb);
                    % %                     goods = ovps > 0 & ovps < 1;
                    % %                     f = find(goods);
                    % %                     p = f(randi(length(f)));
                    % %                     curNode.bbox = cand_boxes(p,:);
                    % %                     curNode.poly = box2Pts(curNode.bbox);
                    prevNode = newConfig{iNode-1};
                    p = prevNode.params;
                    u = p.endPoint-p.startPoint;
%                     plotPolygons(p.endPoint,'m+');
                    u1 = p.endPoint+u/4;
%                     quiver(p.endPoint(1),p.endPoint(2),u(1),u(2));
                    sz = [curNode.spec.avgWidth curNode.spec.avgLength];
                    curNode.bbox = inflatebbox(u1,sz*face_h,'both',true);
                    curNode.poly = box2Pts(curNode.bbox);
                    %curNode.poly = box2Pts(imgData.gt_hand);
                    % get some edge boxes which is attached to the previous
                    % box, or generate a box on your own
            end
            newConfig{iNode} = curNode;
        end
        configurations{end+1} = newConfig;
    end
end