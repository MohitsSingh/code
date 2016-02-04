function [nns,bbs] = matchFaces(conf,X_images,imgs,img_ids,orig_bbs,X_ref,ref_imgs,forest,knn,ID,debug_)
if (nargin < 11)
    debug_ = false;
end

maxNumComparisons = inf;
nns = zeros(length(imgs),knn);
bbs = zeros(length(imgs),4);
for iImage = 801:length(imgs)
    
    iImage
    curX = single(X_images(:,iImage));
    I = im2single(imgs{iImage});
    %[index,dist] = vl_kdtreequery(forest,X_ref,curX,'numneighbors',knn,'MAXNUMCOMPARISONS',maxNumComparisons);
    index = ID(iImage,1:knn);
    
    
    %     nns_train(iImage,:) = index;
    dist = sum(bsxfun(@minus,curX,X_ref(:,index)).^2);
    curNeighbors = X_ref(:,ID(iImage,:));
    bestDist = sum(dist.^.5)
    if (debug_)
        clf; subplot(2,2,1); imagesc(I); axis image;
        ims = cat(4,ref_imgs{index});
        subplot(2,2,2); montage2(ims,struct('hasChn',true));
        E = fevalArrays(ims,@(x) gradientMag(im2single(x)));
        r = repmat(sum(E,3),[1 1 3])+I;
        r = normalise(r);
        subplot(2,2,3); imagesc(r); axis image;
    end
%     pause;
%     continue;
    % %TODO : INTERESTING: I take the neighbors if one the knn of the original image, inside
    % aflw. This way, the other neighbors don't necessarily have unwanted
    % artefacts.
%     [index,dist] = vl_kdtreequery(forest,X_ref,X_ref(:,index(5)),'numneighbors',knn,'MAXNUMCOMPARISONS',maxNumComparisons);
%     
%     if (debug_)
%         ims = cat(4,ref_imgs{index});
%         subplot(2,2,4); montage2(ims,struct('hasChn',true));
%     end
%     pause;continue;
    % Another strategy: search a bit around the detection window in the
    % original image, obtaining the window which minimizes the distance to
    % faces in aflw. I can also rotate it a bit in plane...
    %     figure(2); clf;
    bb = orig_bbs(iImage,:);
    I_orig = getImage(conf,img_ids{iImage});
    
    sz = bb(3)-bb(1)+1;
    %I_orig = imResample(I_orig,80/sz,'bilinear');
    I_orig = imresize(I_orig,80/sz,'bilinear');
    bb(1:4) = round(bb(1:4)*80/sz);
    bestBB = bb(1:4);
    tRange = -8:4:8; % this is cool!!
    bestT = [0 0];
    bestS = 1;
    bestR = 0;
    %for iRot = -20:20:0;
    bestIDX = index;
    
    for iRot = 0
        for iScale = [.8 1];
%         for iScale = 1.4
            for iX = 1:length(tRange)
                for iY = 1:length(tRange)
                    bb_sub = bb(1:4) +[tRange(iX) tRange(iY) tRange(iX) tRange(iY)];
                    bb_sub = (inflatebbox(bb_sub,iScale,'both',false));
                    %                     bb_sub = inflatebbox(bb_sub,[100 100],'both',true);
                    bb_sub = round(bb_sub);
                    I_sub = cropper(I_orig,bb_sub);
                    %                     I_sub = imrotate(I_sub,iRot,'bilinear','crop');
                    %                     I_sub = I_sub(10:89,10:89,:);
                    I_sub = imResample(I_sub,[80 80],'bilinear');
                    curX = col(fhog(im2single(I_sub)));
                    
                    %[index,dist] = vl_kdtreequery(forest,X_ref,curX,'numneighbors',knn,'MAXNUMCOMPARISONS',maxNumComparisons);
                    dd = l2(curX',curNeighbors');
                    [index,dist] = sort(dd,2,'ascend');
                    curDist = sum(dist.^.5);
                    if (curDist < bestDist)
                        bestDist = curDist;
                        bestT = [tRange(iX) tRange(iY)];
                        bestS = iScale;
                        bestIDX = index;
                        bestBB = bb_sub; % back to the original coordinates...
                    end
                end
            end
        end
    end
    
    nns(iImage,:) = bestIDX;
    bbs(iImage,:) = bestBB*sz/80;
    if (debug_)
        disp('done');
        disp([bestT bestS]);
        disp(bb(end));
    end
    if (debug_)
        bb_sub = bb(1:4) +[bestT bestT];
        bb_sub = (inflatebbox(bb_sub,bestS,'both',false));
        bb_sub = round(bb_sub);
        
        I_sub = cropper(I_orig,bb_sub);
        I_sub = imResample(I_sub,[80 80],'bilinear');
        curX = col(fhog(im2single(I_sub)));     
        dd = l2(curX',curNeighbors');
        [index,dist] = sort(dd,2,'ascend');
%        [index,dist] = vl_kdtreequery(forest,X_ref,curX,'numneighbors',knn,'MAXNUMCOMPARISONS',maxNumComparisons);
        ims = cat(4,ref_imgs{index});
        subplot(2,2,3); imagesc(I_sub); axis image;
        subplot(2,2,4); montage2(ims,struct('hasChn',true));
        disp(curDist);
        pause(.1);drawnow
        pause;
        clf;
    end
end