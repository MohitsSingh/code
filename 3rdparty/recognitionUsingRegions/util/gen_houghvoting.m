function [bbox, cmap] = gen_houghvoting(regions1,regions2,distmat,weight,bound,filename1,filename2)
% function bbox = gen_houghvoting(regions1,regions2,distmat,weight,bound)
%
% backproject bounding boxes from img1 to img2 based on region matching
% Inputs:       regions1: bags of regions in img1
%               regions2: bags of regions in img2
%               distmat:  region-to-region distance matrix
%               weight:   saliency of regions in img1
%               bound:    bounding boxes for img1
% Outputs:      bbox:     backprojected bounding boxes for img2
%               cmap:     confidence map information for img2
%
% Copyright @ Chunhui Gu, April 2009

isvisualize = 0;
if isvisualize, assert(nargin==8); end;

% Parameter setting
ratio = 1.1; lowerbound = 0.5;  % select matched regions
frac_window = 0.8;              % prune out large bounding boxes
param.th = 0; param.sw = 0.1; param.sh = 0.1; param.ss = 1.05; % non-max suppression parameters
% a conservative scale variation is param.ss = log(2);
% End parameter setting

bboxId = 0;
bbox.rect = zeros(0,4); bbox.score = zeros(0,1);
cmap = cell(1,length(regions1));
for regionId1 = 1:length(regions1),

    if weight(regionId1) <= 0,
        continue;
    end;
    
    region1 = regions1{regionId1};
    
    %% select matched regions
    [goodvec,gdist] = select_matchedregions(distmat(regionId1,:),ratio,lowerbound);
    r2 = regions2(goodvec);
    goodind = find(goodvec == true); %% added
    
    bId = 0; b.rect1 = []; b.rect = []; b.score = [];
    for regionId2 = 1:length(r2),
        
        region2 = r2{regionId2};
        [BB, bind] = backprojection(region1,bound,region2);
        
        Bb(1) = max(1,BB(1)); Bb(2) = max(1,BB(2));
        Bb(3) = min(size(region2,2),BB(3)); Bb(4) = min(size(region2,1),BB(4));

        rect1 = [bound(bind,1) bound(bind,2) bound(bind,3)-bound(bind,1) bound(bind,4)-bound(bind,2)];
        rect2 = [Bb(1) Bb(2) Bb(3)-Bb(1) Bb(4)-Bb(2)];
        score = weight(regionId1)/sum(weight)*(1-2*gdist(regionId2));

        if Bb(1) < Bb(3) && Bb(2) < Bb(4) && ...
            (Bb(3)-Bb(1))*(Bb(4)-Bb(2)) >= frac_window * (BB(3)-BB(1))*(BB(4)-BB(2)) ...
            && score > 0,

            bId = bId + 1;
            b.rect1(bId,1:4) = rect1;
            b.rect(bId,1:4) = rect2;
            b.score(bId,1) = score;
        else
            goodvec(goodind(regionId2)) = false;
        end;
        
    end;
    
    vgoodind = find(goodvec == true);
    cmap{regionId1}.matched_regIds = vgoodind;
    cmap{regionId1}.bbox = b.rect;
    cmap{regionId1}.chi2 = distmat(regionId1,vgoodind);
    cmap{regionId1}.weight = weight(regionId1);
    
    % non-suppression to eliminate redundant bounding boxes
    [drect,dscores] = mean_shift(b.rect,b.score,param,'max');
    
    nrects = length(dscores);
    if nrects > 0,
        bbox.rect(bboxId+1:bboxId+nrects,:) = drect;
        bbox.score(bboxId+1:bboxId+nrects,:) = dscores;
        bboxId = bboxId + nrects;
    end;
    
    if isvisualize,
        img1 = im2double(imread(filename1));
        img2 = im2double(imread(filename2));
        figure(1); clf; hold on;
        subplot(2,2,1); imshow(img1,[]);
        title(['W = ' num2str(weight(regionId1))]);
        subplot(2,2,2); imshow(img2,[]);
        subplot(2,2,3); imshow(region1,[]);
        for bId = 1:length(b.score),
            subplot(2,2,1); rectangle('Position',b.rect1(bId,:),'EdgeColor','r');
            subplot(2,2,2); rectangle('Position',b.rect(bId,:),'EdgeColor','r');
            title(['Score = ' num2str(b.score(bId))]);
            subplot(2,2,4); imshow(region2,[]);
        end;
        
        subplot(2,2,4); imshow(img2,[]);
        for ii = 1:nrects,
            rectangle('Position',drect(ii,:),'EdgeColor','r');
        end;

        keyboard;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select matched regions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [goodregions,gooddist] = select_matchedregions(rrdist,ratio,lowerbound)
% Input:  rrdist: region-to-region distance vector
%         ratio:  ratio of max allowed distance to minimum distance
%         lowerbound: lower bound on minimum distance
% Output: goodregions: binary vector of good matching regions

%goodregions = true(1,length(rrdist));
goodregions = (rrdist <= ratio*min(rrdist));
if min(rrdist) > lowerbound,
    goodregions = false(1,length(rrdist));
end;
gooddist = rrdist(goodregions);