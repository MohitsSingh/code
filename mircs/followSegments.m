function endPT = followSegments(L,startPt,vec,im,regions,strawBox,realBoundaries,ucm)
rprops = regionprops(L,'PixelIdxList','Orientation',...
    'Eccentricity','MajorAxisLength','MinorAxisLength','PixelList',...
    'Centroid','Area');
RGB = label2rgb(L, 'spring', 'c', 'shuffle');

I = cropper(im,round(strawBox));
roi = directionalROI(im,startPt,vec);

startPt = ceil(startPt);
Z = false(size(L));
Z(startPt(2),startPt(1)) = true;
f = unique(L(Z));
f = f(f>0);

orientations = cat(1,rprops.Orientation);
ori_vecs = [-cosd(orientations) sind(orientations)];

orientation = rprops(f).Orientation;
maxSegments = 5;
nSegs = 1;

if (orientation < 0)
    orientation = orientation + 180;% point "down"
end
direction = [-cosd(orientation),sind(orientation)];
vec = rprops(f).MajorAxisLength*direction;

endPT = startPt + vec;

Am = boundarylen(L,numel(unique(L)));

% find the UCM strength between the segments...
% Aucm = ucmStrengh(L,Am,ucm);

% H = hierarchyStrength(L,regions)

centroids = cat(1,rprops.Centroid);

% find the distances between centroids...
D = l2(centroids,centroids);

% imagesc(L);
imagesc(im.*repmat(realBoundaries,[1 1 3]));
hold on;
gplot(Am.*(D.^.5<100),centroids,'g');

%     return;

killBorders = false;
graphData = constructGraph(im,zeros(size(L)),L,killBorders);

% imagesc(im.*repmat(realBoundaries,[1 1 3]));
% hold on;
% gplot(graphData.pairwise{1}>=.001,centroids,'g');


%
% find the agreement in direction...
% the orientation is more trusted as a function of the eccentricity.
eccentricities = [rprops.Eccentricity];
minorAxisLengths = [rprops.MinorAxisLength];
ori_vecs_ = ori_vecs;
% ori_vecs_ = ori_vecs.*repmat(eccentricities',1,2);
% dirAgreements = ori_vecs_*ori_vecs_';

% binaryPotentials = graphData.pairwise{1}.*abs(dirAgreements);
binaryPotentials = graphData.pairwise{1};
% curDirection = [0 0];
direction = [0 0];
neighborQueue = [];


% get the region graph...

visited = false(size(rprops));
visited(f) = true;

while(true)
%     clf;imagesc(L);
%     hold on;
%     quiver(startPt(1),startPt(2),vec(1),vec(2),'g','LineWidth',3);
%     % continue to the next segment if the aspect ratio, with, angle are
    % similar enough to this one.
    
    % find neighboring segments.
    curNeighbors = find(binaryPotentials(f,:));
    
    rowSel = zeros(size(Am));
    rowSel(f,:) = 1;
    
    imagesc(im.*repmat(realBoundaries,[1 1 3]));
 
    hold on;
    gplot(binaryPotentials.*rowSel,centroids,'g');
    hold on;
    plot(rprops(f).Centroid(1),rprops(f).Centroid(2),'m*');
    quiver(startPt(1),startPt(2),vec(1),vec(2),'r','LineWidth',3);
    %gplot(Am.*(D.^.5<9999).*rowSel.*(graphData.pairwise > 1.8),centroids,'g');
%     gplot(Am.*(graphData.pairwise{1} > .1),centroids,'g');
%     return
    curPotentials = binaryPotentials(f,curNeighbors)+eccentricities(curNeighbors)-...
        minorAxisLengths(curNeighbors);
        
    localDirection = bsxfun(@minus,centroids(curNeighbors,:),centroids(f,:));
    localDirection = normalize_vec(localDirection')'
    relativeVecs = (localDirection*direction') > sind(50);
    originalDirection = (centroids(curNeighbors,:)*direction') > sind(80);
        if (nSegs>1)
            [bestNeighbor,iBest] = max(curPotentials(:)+1000*relativeVecs);
        else
            [bestNeighbor,iBest] = max(curPotentials(:));
        end
    pause(.1)
    f = curNeighbors(iBest);
    visited(f) = true;
    binaryPotentials(:,f) = false;
%     figure,imagesc(L==f)
          nSegs = nSegs+1;
    if (nSegs > maxSegments)
        disp('reached maximal number of segments, aborting');
        break;
    end
    
    direction = localDirection(iBest,:);
%     break;
% pause;
% imagesc(im.*repmat(ismember(L,find(visited)),[1 1 3]));
% pause;
    continue;
    
%     Z_sel = false(size(L));
%     Z_sel(nextPoint(2),nextPoint(1)) = 1;
%     Z_sel = imdilate(Z_sel,ones(3));
%     curNeighbors = setdiff(unique(L(Z_sel(:))),f);
    
    
    
    % (i)check type of junction;
    % (ii) check appearance of current neighbor...
    
    %neighborQueue = [curNeighbors,neighborQueue];
    T_minorAxisRatio = .9;
    found = 0;
    for iNeighbor = 1:length(curNeighbors)
        n = curNeighbors(iNeighbor);
        minorAxisRatio = rprops(f).MinorAxisLength / rprops(n).MinorAxisLength;
        minorAxisRatio = min(minorAxisRatio,1/minorAxisRatio);
        
        if (minorAxisRatio < T_minorAxisRatio)
            continue;
        end
        
        curDot = dot(ori_vecs(f,:),ori_vecs(n,:))
        if (curDot < sind(70))
            continue;
        end
        found = n;
        break;
    end
    
    if (~found)
        break;
    end
    
    f = found;
    direction = ori_vecs(f,:);
    vec = rprops(f).MajorAxisLength*direction;
    startPt = nextPoint;
    
%     plot(nextPoint(1),nextPoint(2),'r*');
    
    nSegs = nSegs+1;
    if (nSegs > maxSegments)
        disp('reached maximal number of segments, aborting');
        break;
    end
    inBounds = inImageBounds(size(L),nextPoint);
    if (~inBounds)
        disp('reached image boundary, aborting');
        break;
    end
    % follow the next segment matching this one
end

imagesc(im.*repmat(ismember(L,find(visited)),[1 1 3]));

end