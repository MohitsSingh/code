function [nns,bbs] = matchFaces(conf,X_images,imgs,img_ids,orig_bbs,X_ref,ref_imgs,knn,ID,debug_)
if (nargin < 10)
    debug_ = false;
end

nns = zeros(length(imgs),knn);
bbs = zeros(length(imgs),4);
for iImage = 803:length(imgs)
    
    iImage    
    id = ID(iImage,:);
    bb = orig_bbs(iImage,:);    
    index = id(1:knn);
    curNeighbors = X_ref(:,id);
    
    bbs(iImage,:) = bb(1:4);
    nns(iImage,:) = id(1:knn);
    if (bb(end)<0)
        continue;
    end
    %     nns_train(iImage,:) = index;
    curX = single(X_images(:,iImage));
    I = im2single(imgs{iImage});
    dist = sum(bsxfun(@minus,curX,X_ref(:,index)).^2);
    
    bestDist = sum(dist.^.5)
    if (debug_)
        clf; subplot(2,3,1); imagesc(I); axis image;
        ims = cat(4,ref_imgs{index});
        subplot(2,3,2); montage2(ims,struct('hasChn',true));                
        %E = fevalArrays(ims,@(x) gradientMag(im2single(x)));        
        
        %clf; imagesc(edge(mean(single(E),4),'canny')); colormap gray
        
        
%         computeSkinProbability(double(im));
        
%         clf; imagesc(mean(single(ims),4))
%         r = normalise(r);
%         subplot(2,3,3); imagesc(r); axis image;
    end
    %     pause;
    %     continue;
    % %TODO : INTERESTING: I take the neighbors if one the knn of the original image, inside
    % aflw. This way, the other neighbors don't necessarily have unwanted
    % artefacts.
    %
    %     if (debug_)
    %         ims = cat(4,ref_imgs{index});
    %         subplot(2,3,4); montage2(ims,struct('hasChn',true));
    %     end
    %     pause;continue;
    % Another strategy: search a bit around the detection window in the
    % original image, obtaining the window which minimizes the distance to
    % faces in aflw. I can also rotate it a bit in plane...
    %     figure(2); clf;
    
    I_orig = getImage(conf,img_ids{iImage});
    
    sz = bb(3)-bb(1)+1;
    rr = sz/80;
    %I_orig = imResample(I_orig,80/sz,'bilinear');
    %     I_orig = imresize(I_orig,80/sz,'bilinear');
    %     bb(1:4) = round(bb(1:4)*80/sz);
    bestBB = bb(1:4);   
    bestT = [0 0];
    bestS = 1;
    bestR = 0;
    %for iRot = -20:20:0;
    bestIDX = index;
    % define the rectangles....
    rots = [0];
    tRange = -8:4:8; 
    scales = [.8 1];
    nRects = length(rots)*length(scales)*length(tRange)^2;
    rects = zeros(nRects,5);
    iRect = 0;
    sub_imgs = zeros(80,80,3,nRects);
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
                    sub_imgs(:,:,:,iRect) = imResample(arrayCrop(I_orig,startLocs,endLocs),[80 80],'bilinear');
                end
            end
%             end
        end
    end
    
%     figure,imagesc(I_orig);
%     hold on; plotBoxes2(rects(:,[2 1 4 3]));    
    xx =fevalArrays(im2single(sub_imgs),@(x) col(fhog(x)));
        
    dd = l2(xx',curNeighbors');
    [dist,index] = sort(dd,2,'ascend');
    dist = dist(:,1:knn);
    index = id(index(:,1:knn));
    dist = sum(dist.^.5,2);
    
    [d_,id_] = min(dist);
    bestDist = d_
    bestIDX = index(id_,:);
    bestBB = rects(id_,1:4);
    nns(iImage,:) = bestIDX;
    bbs(iImage,:) = bestBB;
    if (debug_)
        disp('done');
%         disp([bestT bestS]);
        disp(bbs(iImage,:));
    end
    if (debug_)
        
        I_sub = sub_imgs(:,:,:,id_);        
        subplot(2,3,4); imagesc(I_sub); axis image;
        subplot(2,3,5); montage2(cat(4,ref_imgs{nns(iImage,:)}),struct('hasChn',true));
        
        
%         rr = ref_imgs(nns(iImage,:));
        
%         for kk = 1:length(rr)
%             segs = vl_slic(single(vl_xyz2luv(vl_rgb2xyz(im2single(rr{kk})))),50,.01);
%             clf; imagesc(paintSeg(rr{kk},segs)); pause;
%         end
        %rr = im2single(cat(4,ref_imgs{nns(iImage,:)}));
        rr = im2single(cat(4,ref_imgs{id(1:20)}));
%         E = fevalArrays(rr,@(x) exp(-bwdist(edge(rgb2gray(x),'canny')).^2/1));
%         imagesc(mean(E,3))
%             @(x) ((computeSkinProbability(x))
%         E = fevalArrays(255*rr,...
%             @(x) imfilter(normalise(computeSkinProbability(x)),ones(3)/9));
        E = fevalArrays(255*rr,...
            @(x) ((computeSkinProbability(x))));
        %subplot(2,3,6); imagesc(medfilt2(normalise(sum(E,3)))); axis image
        subplot(2,3,6); imagesc(normalise(mean(E,3))); axis image;
%         %r = repmat(sum(E,3),[1 1 3])+I;
% %        E = fevalArrays(ims,@(x) rgb2gray(x));
%         R = normalise(median(E,3));
        
%         subplot(2,3,3),imagesc(R); axis image
%         R = imfilter(R,fspecial('gauss',5,2));
%         subplot(2,3,6),imagesc(edge(R,'canny')); axis image
        
        disp(d_);
          bb(end)
       drawnow
        pause;
        clf;
    end
end