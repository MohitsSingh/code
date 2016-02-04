function res = processLinesToCandidates(I,segs)
res = {};
% find parallel lines...

nSegs = size(segs,1);
vecs = segs2vecs(segs);
[X,norms] = normalize_vec(vecs');

minNorm = 3;
segs(norms<minNorm,:) = [];
nSegs = size(segs,1);
vecs = segs2vecs(segs);
[X,norms] = normalize_vec(vecs');


% remove too short norms
cos_angles = X'*X;
% remove self-angle
cos_angles = cos_angles.*(1-eye(size(cos_angles)));
maxAngle = 15; % maximal angle between adjacent segments.
angles = real(acosd(cos_angles));
% if segments are quite short (relative to their distance),
%consider wider angles.
means = (segs(:,1:2)+segs(:,3:4))/2;
D = l2(means,means).^.5;
D(eye(size(D))>0) = inf;


% make sure the distance isn't too small...
lines_ = [segs(:,1:2),vecs];
projections = zeros(nSegs,nSegs,2);
intersections = false(nSegs,nSegs);
positions = zeros(nSegs,nSegs,2);

for k = 1:nSegs
    dists = distancePointLine(segs(k,1:2),lines_);
    projections(k,:,1) = dists;
    dists = distancePointLine(segs(k,3:4),lines_);
    projections(k,:,2) = dists;
    curInt = intersectEdges(segs(k,:),segs);
    
    [dists,poss] = distancePointEdge(segs(k,1:2),segs);
    positions(k,:,1) = poss;
    [dists,poss] = distancePointEdge(segs(k,3:4),segs);
    positions(k,:,2) = poss;    
    intersections(k,:) = all(~isinf(curInt) & ~isnan(curInt),2);
end

%goodAngles = angles < 20;
goodAngles = angles <= maxAngle | angles >= 180-maxAngle;

%goodPositions = positions > 0 & positions < 1;
%goodPositions = any(goodPositions,3);

goodPositions = (positions(:,:,1) > 0 & positions(:,:,1) < 1) |...
    (positions(:,:,2) > 0 & positions(:,:,2) < 1);
% goodPositions = goodPositions | goodPositions';

goodCandidates = goodAngles & ~intersections & min(projections,[],3) > 1 & D < 15 & ...
    goodPositions;

% goodCandidates = goodAngles & ~intersections & min(projections,[],3) > 1 & D < 10;
% goodCandidates = ~intersections & min(projections,[],3) >0 & D < 20;
% x2(I); plotSegs(segs);

% segs = selectseg(segs);

% 
% goodAngles = angles <= maxAngle | angles >= 180-maxAngle;
% goodCandidates = goodAngles & ~intersections & min(projections,[],3) > 1 & D < 20;
% figure(1); x2(I); hold on;
% [ss,iss] = selectseg(segs);

% x2(I); plotSegs(segs(any(goodCandidates,1) | any(goodCandidates,2)',:),'g-')
[xx,yy] = meshgrid(1:size(D,1),1:size(D,2));
goodCandidates(yy >= xx) = false; % don't need the symmetric case.

[ii,jj] = find(goodCandidates);
for t = 1:length(ii)
    curPts = [segs(ii(t),1:2);segs(ii(t),3:4);segs(jj(t),1:2);segs(jj(t),3:4)];
    curPts = curPts(convhull(curPts),:);
    res{t} = poly2mask2(curPts,size2(I));
end

%clf;figure(1);pause; displayRegions(I,res,[],.1);

debugging = false;
if (debugging)
    
    [m,im] = sort(D(:),'ascend');
    [ii,jj] = ind2sub(size2(D),im);
    for t = 1:length(ii)
        if (~goodCandidates(im(t))),continue,end
        %     t
        mm = im(t);
        [im(t) ii(t) jj(t)]
        disp(['projection: ' num2str(squeeze(projections(ii(t),jj(t),:))')]);
        disp(['distance: ' num2str(m(t))]);        
        clf;
        imagesc2(I);plotSegs(segs(ii(t),:),'g-','LineWidth',2)
        plotSegs(segs(jj(t),:),'g-','LineWidth',2)
        pause(.1)
    end
    
end