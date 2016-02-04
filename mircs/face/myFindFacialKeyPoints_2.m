function [kp_global] = myFindFacialKeyPoints_2(conf,I,bb,XX,kdtree,trainImgs,trainRects,ptsData,kpParams,net);

debug_ = kpParams.debug_;
wSize = kpParams.wSize;
extractHogsHelper = kpParams.extractHogsHelper;
im_subset = kpParams.im_subset;
requiredKeypoints = kpParams.requiredKeypoints;
%rotations = -30:15:30;
rotations = 0;%-30:15:30;
rotations = -30:5:30;
inflations =[.8:.2:1.2];
orig_bb = bb;
bb = rotateAndInflate(bb,rotations,inflations);
kp_global = zeros(length(requiredKeypoints),5);
subImg = {};
for iBB = 1:length(bb)
    boxFrom = bb{iBB};
    boxTo = [1 1 wSize wSize];
    [subImg{iBB},T]= rectifyWindow(I,boxFrom,wSize([1 1]));
end

% get global prediction
knn = 25;

subImg = cellfun2(@im2uint8,subImg);

if (~isempty(net))
    
    imo = cnn_imagenet_get_batch(subImg, 'averageImage',net.normalization.averageImage,...
        'border',net.normalization.border,'keepAspect',net.normalization.keepAspect,...
        'numThreads', 1, ...
        'prefetch', false,...
        'augmentation', 'none','imageSize',net.normalization.imageSize);
    
    res = vl_simplenn(net, imo);
    % XX = double(squeeze((res(20).x)));
    x = double(squeeze((res(18).x)));
    x = normalize_vec(x);
else
    x = extractHogsHelper(subImg);
    x = cat(2,x{:});
end
knn = round(knn^.5)^2;
[id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);

% DD = (x'*XX)';

% DD = l2(XX',x');
% [dists,id] = sort(DD,1,'ascend');dists = dists(1:knn,:); id = id(1:knn,:);
row(id(1:10))

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

%         T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');

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
% plotBoxes(curBoxes);

vl_tightsubplot(mm,nn,1);plotPolygons(bb{is},'m--');
pause

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