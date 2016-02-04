function [kp_global,kp_local,kp_preds] = myFindFacialKeyPoints(conf,I,bb,XX,kdtree,trainImgs,trainRects,ptsData,kpParams)

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
knn = 25;
x = extractHogsHelper(subImg);
% x = extractHogsHelper({I});
x = cat(2,x{:});
knn = round(knn^.5)^2;
[id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);
% dists = l2(x',XX);
% [dists,id] = sort(dists,'ascend');
% id = id(1:knn)';
% dists = dists(1:knn)';
%[id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);
ids_orig = id;
dists_orig = dists;

[s,is] = min(sum(dists(:,:),1),[],2);
% all_dists(iDet) = sum(dists(:,is));

neighborInds = im_subset(id(:,is));
kps = getKPCoordinates(ptsData(neighborInds),trainRects(neighborInds,:)-1,requiredKeypoints);


figure(2),clf; subplot(1,2,1); imagesc2(I);
 subplot(1,2,2); imagesc2(mImage(trainImgs(id)));
 

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
for iReqKeyPoint = 1:length(requiredKeypoints)
    xy = reshape(kps(:,iReqKeyPoint,:),[],2);
    [u,v] = meshgrid(1:size(Z,2),1:size(Z,1));
    xy(any(isnan(xy),2),:) = [];
    dd = l2([u(:) v(:)],xy);
    sig_ = 20;
    dd = reshape(sum(exp(-dd/sig_)/2,2),size(Z));
    dd_I = imtransform(dd,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
    kp_global(iReqKeyPoint,:) = pMapToBoxes(dd_I,5,1);
    Z = Z+dd;
end

if (nargout == 1)
    return 
end
II = imResample(subImg{is},[wSize,wSize]);
%         T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');
for ptsType = 1:length(requiredKeypoints)
    curPts = reshape(kps(:,ptsType,:),[],2);
    bad_pts = (any(isnan(curPts),2));
    bad_pts = bad_pts | any(curPts < 1,2) | any(curPts > size(II,1),2);
    curPts(bad_pts,:) = [];
    if (none(curPts))
        kp_local(ptsType,:) = -inf;
        continue;
    end
    %         myImgs(bad_pts) = [];
    annParams.maxFeats = 100;
    annParams.cutoff = inf;
    [features,offsets] = prepareANNStarData(conf,myImgs(~bad_pts),curPts,annParams);
    kdtree_feats = vl_kdtreebuild(features,'Distance','L2');
    %imagesc2(myImgs{1}); plotPolygons(squeeze(kps(1,:,:)),'g+');
    curResize = 1;%;resizeRatios(iResize);
    flip = false;
    params.rot = 0;
    %             params.rot = rots(iu);
    params.resizeRatio = curResize;
    params.flip = flip;
    
    params.max_nn_checks = 100;
    params.nn = 1;
    params.stepSize = 1;
    
    [pp] = predictBoxesANNStar(conf,II,features,offsets,kdtree_feats,params);
    %         pp = predictBoxesRegression(conf,II,features,offsets,kdtree_feats,params);
    pp_I = imtransform(pp,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
    kp_local(ptsType,:)=pMapToBoxes(pp_I,5,1);
    pMap=pMap+normalise(pp.^2);
end



%     break
curBoxes = kp_local;


bc1 = boxCenters(kp_global);
bc2 = boxCenters(kp_local);
bc_dist = sum((bc1-bc2).^2,2).^.5;
bad_local = bc_dist > 30;
goods_1= kp_global(:,end) > 2;
kp_local_tmp = kp_local;
kp_local_tmp(bad_local,1:4) = kp_global(bad_local,1:4);
goods = goods_1 & ~bad_local;
kp_preds = kp_local_tmp;


% if (~debug_)
%     save(curOutPath,'curKP_global','curKP_local');
% end
if ~debug_
    return
end

% transform back to I's coordinates
boxTo = [1 1 fliplr(size2(Z))];
T = cp2tform(box2Pts(boxTo),bb{is},'affine');
Z_I = imtransform(Z,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);

mm = 3; nn = 3;
figure(1);
clf;
vl_tightsubplot(mm,nn,1);
imagesc2(I); plotBoxes(orig_bb,'g-');
%plotBoxes(faceBox,'r--','LineWidth',2);
vl_tightsubplot(mm,nn,2);
imagesc2(imResample(subImg{is},[wSize,wSize]));
vl_tightsubplot(mm,nn,3);
M = mImage(myImgs);
imagesc2(M);
% %
vl_tightsubplot(mm,nn,5);
imagesc2(I);

global_pred_boxes = kp_global;
plotBoxes(global_pred_boxes);
bbb = boxCenters(global_pred_boxes);
vl_tightsubplot(mm,nn,2);
vl_tightsubplot(mm,nn,6);
imagesc2(I);
plotBoxes(curBoxes);

vl_tightsubplot(mm,nn,1);plotPolygons(bb{is},'m--');
% pause

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