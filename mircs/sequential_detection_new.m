echo off;
if (~exist('toStart','var'))
    initpath;
    config;
    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
    imageSet = imageData.test;
    face_comp = [imageSet.faceLandmarks.c];
    cur_t = imageSet.labels;
    fb = FbMake(2,3,1);
    fb = squeeze(fb(:,:,1));
    dTheta = 10;
    thetaRange = 0:dTheta:180-dTheta;
    b = zeros(dsize(fb,1:2));
    doGabor1 = true;
    if (doGabor1)
        b(ceil(end/2),1:end) = 1;
        fb2 = zeros([size(fb) length(thetaRange)]);
        a = zeros(7);
        a(3,4) = 1;
        a(5,4) = -1;
        for k = 1:length(thetaRange)
            q = imrotate(b,(k-1)*dTheta,'bicubic','crop');
            fb2(:,:,k) = FbApply2d(q,imrotate(fb(:,:,1)',(k-1)*dTheta,'bicubic','crop'),'same',0);
            fb2(:,:,k) = fb2(:,:,k)/sum(sum(abs(fb2(:,:,k))));
        end
    end
       
    montage2(fb2)
    %     initUnaryModels
    iv = 1:length(cur_t);
    allScores = -inf(size(cur_t));
    toSkip = false(size(cur_t));
    frs = {};
    pss = {};
        
    propss = {};
    Ds = {};
    angless ={};
    
    hand_scores = {};
end

addpath('/home/amirro/code/3rdparty/sliding_segments');
conf.demo_mode = true;


% get the different properties...
% I_subs = getSubMouthImages(conf,imageSet);

%%
close all;
debug_ = true;

ff = find(cur_t);
% 505, the Obama image
% for k = ff([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54])'

% for q = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ] % 526
% iv = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ];
iv = 1:1000
for q = 1:length(iv)
    q
    k = iv(q);
    imageInd = k;
    if (~debug_)
        if (~isinf(allScores(k)))
            continue;
        end
    end
    if (toSkip(k))
        continue;
    end
    
    currentID = imageSet.imageIDs{imageInd};
    if(~cur_t(k))
                                        continue;
    end
    curTitle = '';
    %      clf ;pause;
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);
    %     clear regions;
    %     [regions,regionOvp,G] = getRegions(conf,currentID,false);
    if (debug_)
        clf;
        subplot(2,2,1);
        imagesc(I); axis image; hold on;
        plotBoxes2(faceBoxShifted([2 1 4 3]));
        plotBoxes2(lipRectShifted([2 1 4 3]),'m');
    end
    box_c = round(boxCenters(lipRectShifted));
    
    % get the radius using the face box.
    [r c] = BoxSize(faceBoxShifted);
    boxRad = (r+c)/2;
    
    bbox = [box_c(1)-r/4,...
        box_c(2),...
        box_c(1)+r/4,...
        box_c(2)+boxRad/2];
    bbox = round(bbox);
    if (debug_)
        plotBoxes2(bbox([2 1 4 3]),'g');
    end
    
    if (any(~inImageBounds(size(I),box2Pts(bbox))))
        if (~debug_)
            allScores(k) = -10^6;
        end
        continue;
    end
    I_sub = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I_sub = imresize(I_sub,[50 50],'bilinear');
    I_sub = rgb2gray(I_sub);
    II = imresize(I_sub,2,'bicubic');
    if (debug_)
        subplot(2,2,2); imagesc(II); axis image; colormap gray;
    end
    [lineness,symmetry,M,orientation] = getLineProps(II);
    edgeMap = edge(II,'canny');
    [edgelist edgeim] = edgelink(edgeMap, []);
    seglist = lineseg(edgelist,3);
    if (debug_)
        subplot(2,2,3);
        imagesc(edgeMap);colormap gray;axis image;hold on;
    end
    segs = seglist2segs(seglist);
    segs = segs(:,[2 1 4 3]); % make order x,y,x,y
    segs = fixSegs(segs); % make sure that the upper y point is first.
    vecs = segs2vecs(segs);
    d = atand(vecs(:,2)./vecs(:,1));
    startPts = segs(:,1:2);
    sel_x = ismember(startPts(:,1),30:70);
    sel_y = ismember(startPts(:,2),1:30);
    sel_ = sel_x & sel_y;
    sel_ = true(size(sel_));
    if (debug_)
        quiver(segs(sel_,1),segs(sel_,2),vecs(sel_,1),vecs(sel_,2),0,'r','LineWidth',2);
    end
    pause; continue;
    %     segs_ = segs(sel_,:);
    % 1. segs with strong responses...
    
    [Z,allPts]= paintLines(zeros(size(II)),segs(:,[2 1 4 3]));
    
    % compute the different attributes for each segment...
    Z = imdilate(Z,ones(3));
    rprops = regionprops(Z,'PixelIdxList');
    props = zeros(size(segs,1),7);
    props(:,1) = d;
    for r = 1:length(rprops)
        p = rprops(r).PixelIdxList;
        props(r,2) = mean(symmetry(p));
        props(r,3) = mean(lineness(p));
        props(r,4) = size(allPts{r},1);
        props(r,7) = mean(M(p));
    end
    props(:,5:6) = startPts;
    
    % check if there is a parallel line...
    [X,norms] = normalize_vec(vecs');
    cos_angles = X'*X;
    % remove self-angle
    cos_angles = cos_angles.*(1-eye(size(cos_angles)));
    maxAngle = 20; % maximal angle between adjacent segments.
    angles = acosd(cos_angles);
    [ii,jj] = find(angles <= maxAngle);
    %[ii,jj] = find(abs(cos_angles) >= cosd(maxAngle)); % ii,jj are possible pairs of segments.
    t = ii<jj;
    ii = ii(t); jj = jj(t);
    % find the maximal distance between pairs of lines
    means = (segs(:,1:2)+segs(:,3:4))/2;
    D = l2(means,means).^.5;
    
    
    Ds{k} = D;
    angless{k} = angles;
    propss{k} = props;
    
    
    curScores = getScores(D,angles,props);
    
    [s,is]  = sort(curScores,'descend');
    for kk = 1:min(1,length(is))
        props(is(kk),:)'
        clf; imagesc(II); axis image; colormap gray; hold on;
        s(kk)
        [nn,mm] = find(paintLines(zeros(size(II)),segs([is(kk)],[2 1 4 3])));
        plot(mm,nn,'r.');
%         pause;
    end



if (~debug_)
    toSkip(k) = true;
end
% %
% %     for kk = 1:length(ii)
% %         v1 = ii(kk); v2 = jj(kk);
% %         if (D(v1,v2) < 10)
% %             clf; imagesc(II); axis image; colormap gray; hold on;
% %             [nn,mm] = find(paintLines(zeros(size(II)),segs([v1 v2],[2 1 4 3])));
% %             plot(mm,nn,'r.');
% %             pause;
% %         end
% %     end
%h =  drawedgelist(segs(ii(kk),:), rowscols, lw, col, figno)

%     z = paintRegionProps(Z,rprops,sum(props(:,2:3),2));
%     imagesc(z); colormap jet
%
% 2. parallel segs


%     imshow(edgeim,[])
if (debug_)
    pause;
end
continue;


figure,imagesc(M); hold on; quiver(-sin_,cos_);

binSize = 16;
nBins = 9;
[M,O] = gradientMag( im2single(II),0,0,0,1 ); softBin = -1; useHog = 2;
figure,imagesc(M)
figure,imagesc(O)
H = gradientHist(M,O,binSize,nBins,softBin,useHog,.2);

V = hogDraw(H.^2,15,1);
figure,imagesc(V)
figure,imagesc(II); colormap gray;

%      [M,O] = gradientMag( im2single(II),0,0,0,1 ); softBin = -1; useHog = 2;
%or1 = or*pi/180;
or1 = or;
H = gradientHist(im2single(pc),O,binSize,nBins,softBin,useHog,.2);

V = hogDraw(H.^2,15,1);
figure,imagesc(V)
%     figure,imagesc(II)

imshow(or,[])
figure,imshow(II)
figure,imshow(im)

% %
pause; continue;
%
%
%     imshow(ft,[])
%     im = dispfeat(ft,pc);
%     figure,imshow(im)
if (doCap)
    for qq = 1:length(thetaRange)
        if (thetaRange(qq) < 145 && thetaRange(qq) > 45)
            s = (theta <= thetaRange(qq)+wTheta) & (theta>=thetaRange(qq)-wTheta);
            %             s = imdilate(s,ones(1,9));
            %             clf; imagesc(s); pause;
            fr2(:,:,qq) = fr(:,:,qq).*s.*radCap;
            %             z(qq) = max(radon(fr2(:,:,q),thetaRange(qq)));
        else
            fr2(:,:,qq) = 0;
        end
        %             fr2(:,:,q) = s;
    end
end



frs{k} = fr2;
pss{k} = ps;
%     frs{k} = bsxfun(@times,fr2,phaseSym);
%     size(fr)
c_ = bsxfun(@times,fr2,ps);
[m,im] = max(c_(:));

if (~debug_)
    allScores(k) = m;
end
[ii,jj,kk] = ind2sub(size(fr2),im);
if (debug_)
    subplot(2,2,2); imagesc(edge(im2double(I_sub),'canny')); axis image
    
    subplot(2,2,4); montage2(  bsxfun(@times,fr2,ps));
    
    %     [z,iz] = max(z);
    
    subplot(2,2,3); %title(num2str([z iz]));%max(fr2(:))));
    %title(num2str(max(fr2(:))));
    title(num2str(m));
    %title(num2str(max(frs{k}(:))));
    %         title(num2str(max(max(max(bsxfun(@times,fr2,ps))))));
    sz2 =size(I_sub);
    hold on;
    cd_ = maxRad*cosd(180-thetaRange(kk));
    sd_ = maxRad*sind(180-thetaRange(kk));
    quiver(jj-cd_,ii-sd_,...
        cd_,sd_,0,'g','LineWidth',3);
    % %         quiver(sz2(2)/2,1,...
    % %             maxRad*cosd(180-thetaRange(kk)),maxRad*sind(180-thetaRange(kk)),0,'g');
    %         saveas(gcf,['/home/amirro/notes/images/drink_mirc/straw_new/' sprintf('%03.0f.jpg',q)]);
    pause;
    %         pause(.001);
    %         if (q==74)
    %             break
    %         end
    %
end


%     handsFile = fullfile(conf.handsDir,strrep(currentID,'.jpg','.mat'));
%     L_hands = load(handsFile);
%     bboxes = L_hands.shape.boxes;
%     b_scores = bboxes(:,6); % remove all but top 1000
%     [b,ib] = sort(b_scores,'descend');
%     bboxes = bboxes(1:ib(min(length(ib),1000)),:);
%     bb = nms(bboxes,.3);
% %     hold on;
% %     plotBoxes2(bboxes(bb,[2 1 4 3]));
%     map = computeHeatMap(I,bboxes(bb,[1:4 6]),'max');
%
%     hand_scores{k} = mean(map(bbox(2):bbox(4),bbox(1):bbox(3)));

continue;
L_feats = load(fullfile('~/storage/bow_s40_feats/',strrep(currentID,'.jpg','.mat')));
feats = (vl_homkermap(L_feats.feats, 1, 'kchi2', 'gamma', 1));

% get a different region subset for each region type.

rs = zeros(length(partNames),size(feats,2));
for iPart = 1:length(partNames)
    [res_pred, res] = partModels(iPart).models.test(feats);
    rs(iPart,:) = row(res);
end
rs(isnan(rs)) = -inf;
subsets = suppresRegions(regionOvp,1,rs,I,regions);

% unite all subsets
sel_ = unique([subsets{:}]);
rs = rs(:,sel_);
regions = regions(sel_);
G = G(sel_,sel_);
selBox = lipRectShifted;
faceBox = faceBoxShifted;
regionConfs = struct('score',{});
for ii = 1:length(partNames)
    
    regionConfs(ii).score = rs(ii,:);
end
f = faceBoxShifted;
cupBox = [f(1) f(4) f(3) f(4)+f(4)-f(2)];
cupBox = inflatebbox(cupBox,[1 1],'both');
hold on; plotBoxes2(cupBox([2 1 4 3]),'g');


% %         pause;continue
% % %
%
%
%     displayRegions(I,rsegions,regionConfs(3).score,0,1);
%
%     continue;
% 1. straw candidates: long object intersecting the lip area.
L_regions = load(fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat')));
props = L_regions.props(sel_);

% find all regions overlapping with lips area.
%     [lip_ovp1,lip_int1] = boxRegionOverlap(lipRectShifted,regions,[]);

regionBoxes = cat(1,props.BoundingBox);
regionBoxes(:,3:4) = regionBoxes(:,3:4)+regionBoxes(:,1:2);
selBox = cupBox;
[sel_ovp,sel_int,sel_areas] = boxRegionOverlap(selBox,regions,[],regionBoxes);

[face_ovp,face_int,face_areas] = boxRegionOverlap(faceBoxShifted,regions,[],regionBoxes);
% find the best face region...

[r,ir] = sort(face_ovp,'descend');

clf;
%     imagesc(blendRegion(I,regions{ir(1)},-1)); axis image;
b1 = blendRegion(I,regions{ir(1)},-1);
L = load(fullfile('~/storage/boxes_s40',strrep(currentID,'.jpg','.mat')));
regions_ss = double(L.boxes(:,[1 2 3 4]));


bbovp = boxesOverlap(faceBoxShifted,regions_ss);
[r,ir] = sort(bbovp,'descend');

imagesc(blendRegion(b1,computeHeatMap(I,[regions_ss(ir(1),:) 1]),-1,[0 0 1])); axis image;

hold on; plotBoxes2(regions_ss(ir(1:min(5,length(ir))),[2 1 4 3]),'m-.');

%displayRegions(I,regions_ss,bbovp,5);

hold on;plotBoxes2(faceBoxShifted([2 1 4 3]),'g-.','LineWidth',2);

% find regions overlapping with the face, and within them, find a
% better face region.

%     new_bbovp = boxRegionOverlap(regions_ss(ir(1),:),[ggg,regions],[],[new_bb;regionBoxes]);
%     pause
%     displayRegions(I,[ggg,regions],new_bbovp,0,5)
%
pause;
continue;
[~,~,s] = BoxSize(round(faceBoxShifted));

sel_inside = sel_int./sel_areas;
lambda = 0;
cup_score = (regionConfs(2).score)-lambda*face_int/s; % don't want segments in face. % + lambda*(sel_inside > .5);
% also, don't want the area to be larger than the face :
cup_score = cup_score;%((face_areas/s) > 0);
cup_score(sel_ovp==0) = -inf;
pause;
displayRegions(I,regions,cup_score,0,1);
continue;
%     selBox = inflatebbox(lipRectShifted,2,'both');
% find regions intersecting with lip area
%[lip_ovp,lip_int,lip_areas] = boxRegionOverlap(selBox,regions,[],regionBoxes);
% now, remove regions which start too high, e.g, intersect the region
% above.
antiSelBox = [1,1,size(I,2),selBox(2)];
[anti_ovp] = boxRegionOverlap(antiSelBox,regions,[],regionBoxes);

%lip_score = exp(-[props.MinorAxisLength]/5) + [props.Eccentricity]+10*[props.MajorAxisLength]/mean(dsize(I,1:2));
lip_score = ([props.MinorAxisLength] < 10) + [props.Eccentricity]+10*[props.MajorAxisLength]/mean(dsize(I,1:2));
lip_score = lip_score + 10*regionResponses;
lip_score(~(lip_ovp > 0 & anti_ovp == 0 & lip_areas < 1000)) = 0;

if (debug_)
    f =  find(lip_score);
    [q,iq] = sort(lip_score(f),'descend');
    showSel_ = f(iq(1:min(3,length(iq))));
    regionSubset = fillRegionGaps(regions(showSel_));
    displayRegions(I,...
        regionSubset, q);
    
end
continue;
[parts,allRegions,scores] = followSegments3(conf,regions,G,regionConfs,I,selBox,faceBox,regionOvp,[],[]);

allScores(k) = allRegions{1}(2);
%         [],[]);%relativeShape,relativeShapes_);
%     Z = zeros(dsize(I));
%     for pp = 1:length(parts{1})
%         Z(regions{allRegions{1}(pp)reg = pp;
%     end

%     pause;
%
end

save segData.mat propss Ds angless;

%%
% ids = {};
scores = zeros(size(propss));
for k = 1:length(propss)
    %     ids{k} = k*ones(size(propss{k},1),1);
    
    curProps = propss{k};
    if (isempty(curProps))
        continue;
    end
    curD = Ds{k};
    curAngles = angless{k};
    
    
    curScores = getScores(curD,curAngles,curProps);
    
    scores(k) = max(curScores);
end

new_score = scores+ismember(face_comp,6:11);
new_score = new_score + (imageSet.faceScores>-.8);

[prec rec aps] = calc_aps2(new_score',cur_t);
% new_score = hh;
[v,iv] = sort(new_score,'descend');
%%
% plot(1:length(v),v,'b');
% hold on; plot(1:length(v),v.*cur_t(iv)','r+');
% theList = false(size(iv));
for k = 1:length(iv)
    iv(k)
    v(k)
    %         if (~cur_t(iv(k)))
    %             continue;
    %         end
    currentID = imageSet.imageIDs{iv(k)};
    I = getImage(conf,currentID);
    clf; subplot(1,2,1); imagesc(I); hold on; axis image;
    pause;
end


