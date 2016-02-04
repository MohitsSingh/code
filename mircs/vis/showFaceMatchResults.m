function res = showFaceMatchResults(conf,imgs,img_ids,ref_imgs,nns,bbs,L_pts,imageSet,faces,landmarks,debug_)
if (nargin <11)
    debug_ = false;
end
% k = randperm(length(imgs));
% g = randperm(length(imgs));
ptsData = L_pts.ptsData;
poses = L_pts.poses;
ellipses = L_pts.ellipses;
res = struct('faceBoxes',{},'lipBoxes',{},'faceScores',{},'labels',{},'imageIDs','faceLandmarks');
res(1).imageIDs = imageSet.imageIDs;
res.labels = imageSet.labels;
res.faceBoxes = bbs;
faceLandmarks = struct('face_outline',{},'face_seg',{});
useLandmarks = true;
useEllipses = true;
doSeg = false;
% [allMasks,valid_] = getMasksFromLandmarks(landmarks,1:length(landmarks));
% for iImage = 1:length(imgs)
for iImage =1:length(imgs)
    %     301 = g(iG);
    %     if (iImage < 1100)
    %         continue;
    %     end
    iImage
    index = nns(iImage,:);
    %     index = index(1:1);
    %     I = imgs{iImage};
    
    
    
    if (useLandmarks)
        [masks_,polygons,valid_] = getMasksFromLandmarks(landmarks,index,ref_imgs);
        masks_ = masks_(valid_);
        
        bw = cat(3,masks_{:});
        
        bestCost = inf;
        bestT = [0 0 0 0];
        
        
        cur_outlines = {};
        for ii = 1:size(bw,3)
            perim = bwboundaries(bw(:,:,ii));

            perim  = perim{1};
            cur_outlines{ii} = (bbs(iImage,3)-bbs(iImage,1))/80*(fliplr(perim));
        end
        
        % try to optimize each target independently.
        [xy,T,curCost,inflateFactor] = refineOutline2(conf,imageSet.imageIDs{iImage},bbs(iImage,:),...
            cur_outlines,debug_);
        
        faceLandmarks(iImage).face_outline_r = xy;
        
                              
    else
        if (useEllipses)
            
            curEllipses = ellipses(index);
            bw = zeros(80,80,length(index),'double');
            
            for ii = 1:length(index)
                
                [x,y] = ellipse2points(curEllipses(ii));
                bw_ = poly2mask(x,y,80, 80);
                bw(:,:,ii) = bw_;
            end
        end
    end
    perim = bwboundaries(mean(bw,3)>=.5);
    faceLandmarks(iImage).face_outline = (bbs(iImage,3)-bbs(iImage,1))/80*fliplr(perim{1});
    
    
    % create landmarks images
    
    
    curPts = ptsData(index);
    ptNames = cat(1,curPts.pointNames);
    mLeft = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthLeftCorner'));
    mRight = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthRightCorner'));
    mCenter = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthCenter'));
    
    p1 = ptsData(index(1));
    m = mLeft | mRight | mCenter;
    m1 = m(1:size(p1.pts,1));
    
    if (any(m1))
        mouth_points = mean(p1.pts(m1,:),1);
    else
        ppp = cat(1,curPts.pts);
        mouth_points = mean(ppp(m,:),1);
    end
    mouth_box = inflatebbox([mouth_points mouth_points],[5 5],'both',true)*(bbs(iImage,3)-bbs(iImage,1))/80;
    
    
    % refine face outline.
    
    faceLandmarks(iImage).face_seg = refineOutline(conf,imageSet.imageIDs{iImage},bbs(iImage,:),...
        faceLandmarks(iImage).face_outline,debug_);
    
    % apply here a mini-refinement stage for the mouth.
    
    res.lipBoxes(iImage,:) = mouth_box;
    
    I = getImage(conf,img_ids{iImage});
    I = cropper(I,round(bbs(iImage,:)));
    if (debug_)
        clf;
        imagesc( I); axis image; hold on;
        plotBoxes2(mouth_box([2 1 4 3]),'g');
        plot(faceLandmarks(iImage).face_outline(:,1),faceLandmarks(iImage).face_outline(:,2));
    end
    
    % crop out of the neighboring images the corresponding lip areas.
    curLipImages = {};
    for iIm = 1:length(index)
        ptNames = curPts(iIm).pointNames;
        mLeft = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthLeftCorner'));
        mRight = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthRightCorner'));
        mCenter = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthCenter'));
        if (~any(mLeft | mRight | mCenter))
            continue;
        end
        pts = curPts(iIm).pts;
        winSize = [20 40];
        pts = round(mean(pts(mLeft | mRight | mCenter,:),1));
        pts_bb = inflatebbox([pts pts],fliplr(winSize),'both',true);
        curLipImage = cropper(ref_imgs{index(iIm)},pts_bb);
        if (size(curLipImage,1) == winSize(1) && size(curLipImage,2)==winSize(2))
            curLipImages{end+1} = curLipImage;
        end
    end
    %         curLipImages = cat(4,curLipImages{:});
    lip_x = fevalArrays(cat(4,curLipImages{:}),@(x) col(fhog(im2single(x),4)));
    mouth_box = round(boxCenters(mouth_box));
    mouth_box = inflatebbox([mouth_box mouth_box],fliplr(winSize)*size(I,1)/80,'both',true);
    sub_img_orig = cropper(I,round(mouth_box));
    
    %     if (debug_)
    %         debug_info.ref_imgs = curLipImages;
    %         [bbs_,dists] = refineDetections(conf,{I},{sub_img_orig},...
    %             mouth_box,lip_x,winSize,debug_info);
    %     else
    [bbs_,dists] = refineDetections(conf,{I},{sub_img_orig},...
        mouth_box,lip_x,winSize);
    
    
    res.lipBoxes(iImage,:) = bbs_;
end

res.faceLandmarks = faceLandmarks;

function [masks,polygons,valid_] = getMasksFromLandmarks(landmarks,index,ref_imgs)
masks = {};
polygons = {};
valid_ = true(size(index));
for k = 1:length(index)
    k
    xy = landmarks(index(k)).xy;
    
    if (isempty(xy))
        valid_(k) = false;
        continue;
    end
    bc = boxCenters(xy);
    
    % %     clf;
    % %
    % %
    % %
    % %
    % %     pause;
    if (size(bc,1)==68)
        outline_ = [68:-1:61 52:60 16:20 31:-1:27];
    else
        
        
        % get the outer mask
        
        %
        
        outline_ = [6:-1:1 16 25 27 22 28:39 15:-1:12];
        outline_1 = [outline_ outline_(1)];
        %          imagesc(ref_imgs{index(k)}); axis image; hold on;
        %         plot(bc(outline_1,1),bc(outline_1,2),'r-');
        %         for q = 1:size(bc,1)
        %             text(bc(q,1)+5,bc(q,2),num2str(q),'color','y');
        %         end
        %         pause;
        
    end
    
    bc = bc(outline_,:);
    
    %     bc = bc(convhull(bc(:,1),bc(:,2)),:);
    masks{k} = poly2mask(bc(:,1),bc(:,2),80, 80);
    polygons{k} = bc;
end

function [x,y] = ellipse2points(E,nPnts)
if (nargin < 2)
    nPnts = 30;
end
[x,y] = ellipse2points2(E.y,E.x,E.ra,E.rb,E.theta,nPnts);


function [x,y] = ellipse2points2(cRow,cCol,ra,rb,phi,nPnts)
ts = linspace(-pi,pi,nPnts+1);  cts = cos(ts); sts = sin(ts);
x = ra*cts*cos(-phi) + rb*sts*sin(-phi) + cCol;
y = rb*sts*cos(-phi) - ra*cts*sin(-phi) + cRow;

% masks = masks(valid_);
r