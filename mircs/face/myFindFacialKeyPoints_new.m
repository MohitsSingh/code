function [kp_global,kp_local] = myFindFacialKeyPoints_new(conf,I,bb,XX,kdtree,trainImgs,trainRects,ptsData,kpParams,all_offsets,kdtree2,XX2)

smallPatchSize = [32 32];
smallCellSize = 4;

debug_ = kpParams.debug_;
wSize = kpParams.wSize;
extractHogsHelper = kpParams.extractHogsHelper;
im_subset = kpParams.im_subset;
requiredKeypoints = kpParams.requiredKeypoints;
%rotations = -30:15:30;
rotations = 0;%-30:15:30;
inflations = 1;

% rotations = -30:5:30;
inflations = 1;
orig_bb = bb;
bb = rotateAndInflate(bb,rotations,inflations);
kp_global = zeros(length(requiredKeypoints),5);
kp_local = zeros(length(requiredKeypoints),5);
subImg = {};
for iBB = 1:length(bb)
    boxFrom = bb{iBB};
    boxTo = [1 1 wSize wSize];
    [subImg{iBB},T]= rectifyWindow(I,boxFrom,wSize([1 1]));
end

% get global prediction
knn = 5;
x = extractHogsHelper(subImg);
x = cat(2,x{:});
knn = round(knn^.5)^2;
[id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);
ids_orig = id;
dists_orig = dists;

[s,is] = min(sum(dists(:,:),1),[],2);
% all_dists(iDet) = sum(dists(:,is));

neighborInds = im_subset(id(:,is));
kps = getKPCoordinates(ptsData(neighborInds),trainRects(neighborInds,:)-1,requiredKeypoints);

a = BoxSize(trainRects(neighborInds,:));
f = wSize./a;
for ikp = 1:size(kps,1)
    kps(ikp,:,:) = f(ikp)*kps(ikp,:,:);
end
pMap = zeros(wSize);
% create a prediction for the points using the nearest neighbors.
myImgs = trainImgs(id(:,is));
myImgs = cellfun2(@(x) imResample(x,[wSize wSize]),myImgs);

T = cp2tform(box2Pts(boxTo),bb{is},'affine');
% make a probability map to start with
Z = zeros(wSize);
boxes = zeros(length(requiredKeypoints),5);
kp_global2 = zeros(size(kp_global))

for iReqKeyPoint = 1:length(requiredKeypoints)
    xy = reshape(kps(:,iReqKeyPoint,:),[],2);
    [u,v] = meshgrid(1:size(Z,2),1:size(Z,1));
    xy(any(isnan(xy),2),:) = [];
    dd = l2([u(:) v(:)],xy);
    sig_ = 20;
    dd = reshape(sum(exp(-dd/sig_)/2,2),size(Z));
    dd_I = imtransform(dd,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
    kp_global2(iReqKeyPoint,:) = pMapToBoxes(dd,5,1);
    kp_global(iReqKeyPoint,:) = pMapToBoxes(dd_I,5,1);
    Z = Z+dd;
end

II = imResample(subImg{is},[wSize,wSize]);
%         T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');

for ptsType =1:length(requiredKeypoints)    
    myStartPoint = boxCenters(kp_global2(ptsType,1:4));    
    ptHistory = zeros(20,2);
    ptHistory(1,:) = myStartPoint;
    for tt = 1:50
        myGlobalBox = inflatebbox(round([myStartPoint myStartPoint]),smallPatchSize,'both',true);
        myFeat = double(col(fhog2(im2single(cropper(II,myGlobalBox)),smallCellSize)));
        
        [id2,dists2] = vl_kdtreequery(kdtree2,XX2,myFeat,'NUMNEIGHBORS',1,'MAXNUMCOMPARISONS',1000);
        clf; imagesc2(II); plotPolygons(myStartPoint,'r*');
        plotBoxes(myGlobalBox,'g-','LineWidth',2);    
        
        predicted_offset = reshape((all_offsets(id2,ptsType,:)),[],2);
        predicted_offset = mean(predicted_offset,1);
        %[xs,xsCum] = fernsRegApply( myFeat, ff_x)
        %[ys,ysCum] = fernsRegApply( myFeat, ff_y);
        newPt = myStartPoint+predicted_offset;
        plotPolygons(newPt,'m*');
        myStartPoint = newPt;
        ptHistory(tt,:) = myStartPoint;
        plotPolygons(ptHistory(1:tt,:),'g-.');
        drawnow        
%         pause
    end                    
    kp_local(ptsType,1:2) = myStartPoint;    
end

%%
    function bb = rotateAndInflate(bb,rotations,inflations)
        bb = repmat(bb,length(rotations),1);
        bb = rotate_bbs(bb,I,rotations,false);
        bbb = {};
        for iInflation = 1:length(inflations)
            for ibb = 1:length(bb)
                curBB = bb{ibb};
                [x,y] = inflatePolygon(curBB(:,1),curBB(:,2),inflations(iInflation));
                bbb{end+1} = [x y];
            end
        end
        bb = bbb;
        
    end
end