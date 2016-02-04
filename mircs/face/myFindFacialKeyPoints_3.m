function [kp_global] = myFindFacialKeyPoints_3(conf,I,XX,kdtree,trainImgs,ptsData,kpParams,net,predictors)

debug_ = kpParams.debug_;
wSize = kpParams.wSize;
% extractHogsHelper = kpParams.extractHogsHelper;
I_orig = I;
I = imResample(I,[wSize wSize]);
requiredKeypoints = kpParams.requiredKeypoints;
kp_global = zeros(length(requiredKeypoints),5);
% get global prediction
knn = kpParams.knn;
subImg = I;
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
    zero_borders = false;
    x = getImageStackHOG(subImg,[wSize wSize],true,zero_borders);    
end


knn = round(knn^.5)^2;
[id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);
% [id,dists] = vl_kdtreequery(kdtree,XX,X_t_2,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);
z = sum(exp(-dists/10),1);
[u,iu] = max(z);

% I = m{iu};
%%
%
% vl_tightsubplot(2,2,1); imagesc2(I);
% vl_tightsubplot(2,2,3); imagesc2(m{iu});
% vl_tightsubplot(2,2,2); imagesc2(mImage(trainImgs(id)));
%%

is = iu;


% try to deform the hog field for a better fit....

% all_dists(iDet) = sum(dists(:,is));
neighborInds = id(:,is);
myImgs = trainImgs(neighborInds);
kps = getKPCoordinates_2(ptsData(neighborInds),requiredKeypoints);
for iNeighbor = 1:size(kps,1)
    kps(iNeighbor,:,:) = wSize(1)*kps(iNeighbor,:,:)/size(myImgs{iNeighbor},1);
end
myImgs = cellfun2(@(x) imResample(x,[wSize wSize]),myImgs);
% for u = 1:length(id)
%     clf; imagesc2(myImgs{u});
%     plotPolygons(squeeze(kps(u,:,:)),'g.');
%     pause
% end

% extract patches around the "real" keypoints

pMap = zeros(wSize);
% create a prediction for the points using the nearest neighbors.
kp_local = NaN(length(requiredKeypoints),5);
doAnn = true;
doLearnPatches = false;

% make a probability map to start with, and vote using it
Z = zeros(wSize);
boxes = zeros(length(requiredKeypoints),5);
for iReqKeyPoint = 1:length(requiredKeypoints)
    xy = reshape(kps(:,iReqKeyPoint,:),[],2);
    [u,v] = meshgrid(1:size(Z,2),1:size(Z,1));
    xy(any(isnan(xy),2),:) = [];
    dd = l2([u(:) v(:)],xy);
    sig_ = 20;
    dd = reshape(sum(exp(-dd/sig_)/2,2),size(Z));
    curPts = reshape(kps(:,iReqKeyPoint,:),[],2);
    bad_pts = (any(isnan(curPts),2));
    bad_pts = bad_pts | any(curPts < 1,2);
    curPts(bad_pts,:) = [];
    if (none(curPts))
        kp_local(iReqKeyPoint,:) = -inf;
        continue;
    end
    
    smallWSize = 30;
    
    if (doLearnPatches)
        pp = dd;
        featureBox = pMapToBoxes(pp,smallWSize,1);
        goodPts = find(~bad_pts);
        %         profile on
        x_crops = [];
        crops = [];
        for iIter = 1:10
%                                   smallWSize = smallWSize-1;
            tic
            iIter
                       
%             my_crop = cropper(I,featureBox(1,:));
            p2 = box2Pts([1 1 smallWSize smallWSize]);
            p1 = box2Pts(featureBox(1:4));
%             p1 = box2Pts([1 1 100 100]);
            T = cp2tform(p1, p2 ,'affine');                                                
            my_crop = imtransform(im2uint8(I),T,'bicubic','XData',[1 smallWSize],'YData',[1 smallWSize]);
            featureBox = round(repmat(featureBox(1:4),length(goodPts),1));                        
                        
            outputs = curPts-featureBox(:,1:2)+1;
            %         outputs = curPts;
            
            if (isempty(crops))            
                crops = multiCrop(conf,myImgs(goodPts),featureBox);
                x_crops = getImageStackHOG(crops,[smallWSize smallWSize],true,false,4);
            end
            
            D = l2(x_crops',getImageStackHOG(my_crop,[smallWSize smallWSize],true,false,4)');
            [u,iu] = sort(D,'ascend');
            %         x2(crops(iu));
            sig_ = 1;
            %             kde_est = sum((repmat(exp(-D/sig_),1,2).*outputs)/sum(exp(-D/sig_)),1);
            kde_est = mean(outputs(iu(1:5),:));
            cur_pk_pred = kde_est+featureBox(1,1:2)-1;
            %         cur_pk_pred = outputs(iu(1),:);
            clf;subplot(2,2,1);imagesc2(I); plotBoxes(featureBox(1,:),'g-');
            plotPolygons(boxCenters(featureBox(1,:)),'gs');
            subplot(2,2,2);imagesc2(imResample(my_crop,1,'bilinear'));plot(smallWSize/2,smallWSize/2,'g+');
            plotPolygons(kde_est,'r+');
            nToShow = 5;
            subplot(2,2,3);imagesc2(mImage(crops(1:nToShow)));
            %         subplot(2,2,1);imagesc2(I); plotBoxes(featureBox(1,:),'g-');
            subplot(2,2,1);
            plotPolygons(cur_pk_pred,'r+');
            featureBox = inflatebbox(cur_pk_pred,smallWSize,'both',true);
            plotBoxes(featureBox,'r-');
            
            %             subplot(2,2,4);
            subplot(2,2,4);imagesc2(mImage(crops(iu(1:nToShow))));
            
            %subplot(2,2,4);imagesc2(mImage(crops(iu)));
            drawnow
            pause(.1-toc)
            
        end
        %         profile viewer
        kp_local(iReqKeyPoint,:) = [cur_pk_pred cur_pk_pred  1];
%         pause
        
        %         for u = 1:length(crops)
        %             figure(1);
        %             clf; imagesc2(crops{u}); plotPolygons(curPts(goodPts(u),:)-featureBox(1,[1 2])+1,'r+');
        %             figure(2);
        %             clf; imagesc2(myImgs{u}); plotPolygons(curPts(goodPts(u),:),'r+');
        %             pause
        %         end
        %         for u = 1:length(crops)
        %             clf; imagesc2(crops{u}); plotPolygons(curPts(goodPts(u),:)-featureBox([1 2]),'r+');
        %             pause
        %         end
    end
    
    if (doAnn)
        
        if exist('predictors','var')
            features = predictors(iReqKeyPoint).features;
            offsets = predictors(iReqKeyPoint).offsets;
            kdtree_feats = predictors(iReqKeyPoint).kdtree;
        else
        %         myImgs(bad_pts) = [];
            annParams.maxFeats = inf;
            annParams.cutoff = inf;
            [features,offsets] = prepareANNStarData(conf,myImgs(~bad_pts),curPts,annParams);
            kdtree_feats = vl_kdtreebuild(features,'Distance','L2');
        end
        %imagesc2(myImgs{1}); plotPolygons(squeeze(kps(1,:,:)),'g+');
        curResize = 1;%;resizeRatios(iResize);
        flip = false;
        params.rot = 0;
        params.resizeRatio = curResize;
        params.flip = flip;
        params.max_nn_checks = 100;
        params.nn = 1;
        params.stepSize = 1;
        pp = dd;
        
        
        pp = predictBoxesANNStar(conf,I,features,offsets,kdtree_feats,params);
     
%         for u = 1:5
%             featureBox = pMapToBoxes(pp,20,1);
%             clf; imagesc2(pp); hold on; plotBoxes(featureBox);
%              clf; imagesc2(showProb(I,pp));
%             drawnow
%             %         pause(.1);
%             pp = predictBoxesANNStar(conf,I,features,offsets,kdtree_feats,params,featureBox);
%            
%         end
        kp_local(iReqKeyPoint,:) = pMapToBoxes(pp,5,1);
    end
    kp_global(iReqKeyPoint,:) = pMapToBoxes(dd,5,1);
    Z = Z+dd;
end


% transform back to I's coordinates

mm = 2; nn = 2;

%figure(1);
clf;
vl_tightsubplot(mm,nn,1);
imagesc2(I);
%plotBoxes(faceBox,'r--','LineWidth',2);
vl_tightsubplot(mm,nn,2);
M = mImage(myImgs);
imagesc2(M);
% %
vl_tightsubplot(mm,nn,3);
imagesc2(I);

global_pred_boxes = kp_global;
% plotBoxes(global_pred_boxes);
bbb = boxCenters(global_pred_boxes);
plotPolygons(bbb,'g.');
plotPolygons(boxCenters(kp_local),'r.');
vl_tightsubplot(mm,nn,4);
%
% imagesc2(I);
% for ikp = 1:size(kps,2)
%     plotPolygons(boxCenters(kp_local),'g.');
% end

% plotBoxes(curBoxes);


pause

%%
end