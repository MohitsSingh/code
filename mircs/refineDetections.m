%function [nns,new_bb] = matchFaces(conf,X_images,imgs,img_ids,orig_bbs,X_ref,ref_imgs,knn,ID,debug_)
% function [new_bb,scores] = refineDetections(conf,imgs,bb_orig,ref_features,...
%     imageSet.frames,frameScale,subScale,debug_info)
function [bbs,dists] = refineDetections(conf,imgs,sub_images,...
    orig_bbs,X_ref,windowSize,debug_info)
if (nargin < 8)
    debug_ = false;
end
knn = 15;
bbs = zeros(length(imgs),4);
feat_fun = @(x) col(fhog(x,4));

for k = 1:length(sub_images)
    sub_images{k} = imResample(sub_images{k},[windowSize],'bilinear');
end
if (length(sub_images)==1)
    X_images = feat_fun(im2single(sub_images{1}));
else
    X_images = fevalArrays(im2single(cat(4,sub_images{:})),feat_fun);
end
D = l2(X_images',X_ref');
[~,ID] = sort(D,2,'ascend');
ID = ID(:,1:min(1000,size(ID,2)));
knn = min(knn,size(ID,2));
nns = zeros(length(imgs),knn);
dists = zeros(length(imgs),knn);

for iImage = 1:length(imgs)
%     iImage
    id = ID(iImage,:);
    bb = orig_bbs(iImage,:);
    index = id(1:knn);
    curNeighbors = X_ref(:,id);
    bbs(iImage,:) = bb(1:4);
    nns(iImage,:) = id(1:knn);
    %     if (bb(end)<0)
    %         continue;
    %     end
    %     nns_train(iImage,:) = index;
    curX = single(X_images(:,iImage));
    %I = im2single(imgs{iImage});
    
    dist = sum(bsxfun(@minus,curX,X_ref(:,index)).^2);
    
    bestDist = mean(dist.^.5);
    I_orig = getImage(conf,imgs{iImage});
    
    sz = bb(3)-bb(1)+1;
    rr = sz/windowSize(1);
    rots = [0];
    tRange = -4:2:4;
    scales = [.8 1];
    nRects = length(rots)*length(scales)*length(tRange)^2;
    rects = zeros(nRects,5);
    iRect = 0;
    sub_imgs = zeros(windowSize(1),windowSize(2),3,nRects);
    for iRot = 1:length(rots)
        curRot = rots(iRot);
        for iScale = 1:length(scales)
            %             for jScale = 1:length(scales)
            curScaleX = scales(iScale);
            curScaleY = scales(iScale);
            for iX = 1:length(tRange)
                curX = tRange(iX)*curScaleX*rr;
                for iY = 1:length(tRange)
                    curY = tRange(iY)*curScaleY*rr;
                    iRect = iRect+1;
                    bb_sub = bb(1:4) +[curX curY curX curY];
                    bb_sub = round(inflatebbox(bb_sub,[curScaleX curScaleY],'both',false));
                    rects(iRect,1:4) = bb_sub;
                    startLocs = [bb_sub([2 1]) 1];
                    endLocs = [bb_sub([4 3]) 3];
                    sub_imgs(:,:,:,iRect) = imResample(arrayCrop(I_orig,startLocs,endLocs),[windowSize],'bilinear');
                end
            end
        end
    end
    
    %     figure,imagesc(I_orig);
    %     hold on; plotBoxes2(rects(:,[2 1 4 3]));
    xx =fevalArrays(im2single(sub_imgs),feat_fun);
    
    dd = l2(xx',curNeighbors');
    [dist,index] = sort(dd,2,'ascend');
    dist = dist(:,1:knn);
    index = id(index(:,1:knn));
    dist_ = mean(dist.^.5,2);
    [d_,id_] = min(dist_);
    dists(iImage,:) = dist(id_,:);
    bestDist = d_;
    bestIDX = index(id_,:);
    bestBB = rects(id_,1:4);
    nns(iImage,:) = bestIDX;
    bbs(iImage,:) = bestBB;
    if (debug_)
        disp('done');
        disp(bbs(iImage,:));
    end
    if (exist('debug_info','var'))
        clf; subplot(2,3,1); imagesc(sub_images{iImage}); axis image;
        ims = cat(4,debug_info.ref_imgs{id(1:knn)});
        subplot(2,3,2); montage2(ims,struct('hasChn',true));
        I_sub = sub_imgs(:,:,:,id_);
        %subplot(2,3,4); imagesc(I_sub); axis image;
        subplot(2,3,4); imagesc(I_orig); axis image; hold on; plotBoxes2(bestBB([2 1 4 3]),'g');
        subplot(2,3,5); montage2(cat(4,debug_info.ref_imgs{nns(iImage,:)}),struct('hasChn',true));
        drawnow
        pause;
        clf;
    end
end