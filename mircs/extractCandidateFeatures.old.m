function R = extractCandidateFeatures(conf,currentID,faceBoxShifted,lipRectShifted,faceLandmarks,debug_,clusters_trained)
weightVector = [1 10 10 0*-.01 10 3 1 1 10 1 0 0 0 0 0];
R = [];
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
personBounds = [xmin ymin xmax ymax];
facePts = faceLandmarks.xy;
if (isempty(facePts))
    return;
end
facePts = boxCenters(facePts);

box_c = round(boxCenters(lipRectShifted));
sz = faceBoxShifted(3:4)-faceBoxShifted(1:2);
bbox = round(inflatebbox([box_c box_c],floor(sz/1.5),'both',true));

if (any(~inImageBounds(size(I),box2Pts(bbox))))
    return;
end

I_sub_color = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
I_sub = rgb2gray(I_sub_color);

ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
ucm = ucm(ymin:ymax,xmin:xmax);
subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
E = subUCM;

xy = faceLandmarks.xy;
xy_c = boxCenters(xy);

chull = convhull(xy_c);
% find the occluder!! :-)
c_poly = xy_c(chull,:);
c_poly = bsxfun(@minus,c_poly,bbox(1:2));
face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(E,1),size(E,2));
bb = bwperim(face_mask);
bd = bwdist(bb);

[ii,jj,vv] = find(subUCM);
if (length(ii) < 5)
    return;
end
xy_edge = [jj ii];
boundaries = bwboundaries(face_mask);
xy_boundaries = fliplr(boundaries{1});

D = l2(xy_boundaries,xy_edge);
vv = repmat(vv',size(xy_boundaries,1),1);

% balance between the distance and strength to the boundary
sig_ = 1000;
cost_fun = 1*exp(-D/sig_)+vv;
tooFar = D>size(E,1)*.3;
cost_fun(tooFar) = exp(-D(tooFar)/sig_);
[~,imm] = max(cost_fun,[],2);
vecs = xy_edge(imm,:)-xy_boundaries;

E = E>.2;
if (~any(E(:)))
    return
end
E = bwmorph(E,'thin',Inf);
if (nnz(E) < 5)
    return
end
[seglist,edgelist] = processEdges(E);

if (debug_)
    drawedgelist(edgelist,size(E),2,'rand');
end

if (debug_)
    %         clc;
end

regions_sub = combine_regions_new(subUCM,.1);
regions_sub = fillRegionGaps(regions_sub);
areas = cellfun(@nnz,regions_sub);
regions_sub(areas/numel(E)>.6) = [];

% perims = cellfun2(@bwperim,regions_sub);
boundaries = cellfun2(@bwboundaries,regions_sub);

% remove boundary points common with the edge of the image.

% take only one of each.
boundaries = cellfun2(@(x) x{1},boundaries);


lines = lineseg(boundaries,2);
% clf; imagesc(I_sub_color); axis image;
% drawedgelist(lines,size(E),2,'rand');return;

% contour_descriptor = HSO([132 132],lines(1),10,15,2);

[candidates,inds] = splitByDirection(lines);
inds = [inds{:}];

I = getImage(conf,currentID);

% make sure left point is first.
candidates = fixSegLists(candidates);
candidates = seglist2edgelist(candidates);
lengths = cellfun(@length,candidates);
[M,O] = gradientMag(im2single(I_sub));

candidates = candidates(lengths > 2);
inds = inds(lengths > 2);

if (isempty(candidates))
    return
end
% make sure that left-right distance is large enough (in terms of
% search window)

% extract some features from these candidates
pFirst = cell2mat(cellfun(@(x) x(1,:),candidates,'UniformOutput',false)');
pLast =  cell2mat(cellfun(@(x) x(end,:),candidates,'UniformOutput',false)');
pMean = cell2mat(cellfun(@(x) mean(x),candidates,'UniformOutput',false)');

horzDist = (pLast(:,2)-pFirst(:,2))./size(I_sub,2);

isConvex = false(size(candidates));
u = pLast-pFirst;

% make sure that the base of the contour isn't too vertical

for ic = 1:length(candidates)
    v = bsxfun(@minus,candidates{ic}(2:end-1,:),pFirst(ic,:));
    curCross = u(ic,2)*v(:,1)-u(ic,1)*v(:,2);
    isConvex(ic) = ~any(curCross>0);
end

% isConvex = true(size(isConvex)); %% TODO!!

if (~any(isConvex))
    return
end

horzDist = horzDist(isConvex);
pMean = pMean(isConvex,:);

candidates = candidates(isConvex);
inds = inds(isConvex);
verticality = abs(u(isConvex,1)./u(isConvex,2));
if (isempty(candidates))
    return
end
contourLengths = cellfun(@(x) sum(sum(diff(x).^2,2).^.5),candidates);
candidateImages = cellfun2(@(x) imdilate(paintLines(false(size(E)),seglist2segs({x})),ones(3)),...
    candidates);

bboxes = cellfun2(@(x) pts2Box(fliplr(ind2sub2(size(E),find(x)))),candidateImages);
bboxes = cat(1,bboxes{:});

% for kkk = 1:length(candidates)
%     clf; imagesc(candidateImages{kkk}); axis image; hold on;
%     plotBoxes2(bboxes(kkk,[2 1 4 3]),'g');
%     pause;
% end

ims = cellfun2(@(x) I_sub + imdilate(paintLines(false(size(E)),seglist2segs({x})),ones(3)),...
    candidates);

ucm_faceRemoved = subUCM;
ucm_faceRemoved(bd<=3) = -subUCM(bd<=3);
ucmStrengths = cellfun(@(x) mean(ucm_faceRemoved(x & subUCM > 0)),candidateImages);
%     ucmStrengths = cellfun(@(x) mean(subUCM(x & subUCM > 0)),candidateImages);

% make sure that the region spanned by this does not overlap the face
% by too much.

skinprob = computeSkinProbability(double(im2uint8(I_sub_color)));
normaliseskinprob = normalise(skinprob);

strel_ = [ones(1,5),zeros(1,3),zeros(1,5)]';
ims_up = cellfun2(@(x) imdilate(paintLines(false(size(E)),seglist2segs({x})),strel_),...
    candidates);
ims_down = cellfun2(@(x) imdilate(paintLines(false(size(E)),seglist2segs({x})),flipud(strel_)),...
    candidates);
ims_ = cat(3,ims_up{:})-cat(3,ims_down{:});

% find the interesection points (if any) of the candidates with the
% given face outline.

c_poly = xy_c(chull,:);
c_poly = bsxfun(@minus,c_poly,bbox(1:2));

% turn poly into a sequence of line segments
face_segs = [c_poly(1:end-1,:),c_poly(2:end,:)];


nIntersections = zeros(size(candidates));

for iCandidate = 1:length(candidates)
    out = lineSegmentIntersect(seglist2segs(candidates(iCandidate)),face_segs(:,[2 1 4 3]));
    nIntersections(iCandidate) = nnz(out.intAdjacencyMatrix);
end

bw = poly2mask(xy_c(chull,1),xy_c(chull,2),size(I,1),size(I,2));

bw = imerode(bw(bbox(2):bbox(4),bbox(1):bbox(3)),ones(3));
insides = cellfun(@(x) nnz(bw.*x),candidateImages);
insides = insides/size(E,1);

skinTransition = zeros(size(candidates));
for iC = 1:length(candidates)
    curIm = ims_(:,:,iC);
    skinTransition(iC) = mean(normaliseskinprob(curIm~=0).*curIm(curIm~=0));
end

rois = cellfun2(@(x) poly2mask(x(:,2),x(:,1),size(E,1),size(E,2)),candidates);

[ovps,ints,areas] = boxRegionOverlap(face_mask,rois);
xy_l = box2Pts(lipRectShifted);
lipMask = poly2mask(xy_l(:,1),xy_l(:,2),size(I,1),size(I,2));
lipMask = lipMask(bbox(2):bbox(4),bbox(1):bbox(3));
lipMask = exp(-bwdist(lipMask).^2/100);
inLips = cell2mat(cellfun2(@(x) max(max(x.*lipMask)),rois));

R = [contourLengths/size(E,1);...
    ucmStrengths;...
    row(pMean(:,1))/size(E,1);...
    verticality';...
    skinTransition;...
    insides;...
    nIntersections;...
    areas./nnz(face_mask);...
    inLips;...
    horzDist';...
    areas;...
    bboxes'/size(E,1)];

currentScores = R'*weightVector';
currentScores(isnan(currentScores)) = -10;
[s,is] = sort(currentScores,'descend');
if (debug_)
    clf;
    subplot(2,3,1);
    imagesc(I); axis image; hold on;
    plotBoxes2(faceBoxShifted([2 1 4 3]));
    plotBoxes2(lipRectShifted([2 1 4 3]),'m');
    plotBoxes2(bbox([2 1 4 3]),'g');
    plot(facePts(:,1),facePts(:,2),'r.');
    subplot(2,3,2);
    imagesc(I_sub_color); axis image; colormap gray;
    if (~isempty(ims))
        subplot(2,3,4); imagesc(subUCM);axis image
        hold on; plot(c_poly(:,1),c_poly(:,2),'g-+');
        plot(xy_boundaries(:,1),xy_boundaries(:,2),'g','LineWidth',2);
        hold on; quiver(xy_boundaries(:,1),xy_boundaries(:,2),vecs(:,1),vecs(:,2),0,'r');
        subplot(2,3,3),montage2(cat(3,ims{is}))
        subplot(2,3,5); imagesc(normaliseskinprob);axis image
    end
end

%
%
