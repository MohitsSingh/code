function curFeats = extractCandidateFeatures4(conf,imageSet,salData,k,debug_)
%Rs = extractCandidateFeatures3(conf,imageSet,salData,k)
if (nargin < 5)
    debug_ = false;
end
curFeats = struct('bbox',{},'bc',{},'horz_extent',{},'y_top',{},'ucmStrength',{},'isConvex',{},...
    'salStrength',{});

currentID = imageSet.imageIDs{k};
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
ucm = ucm(ymin:ymax,xmin:xmax); %#ok<NODEF>
bbox = round(imageSet.faceBoxes(k,1:4));
lipBox = imageSet.lipBoxes(k,1:4)-bbox([1 2 1 2]);
bbox = clip_to_image(bbox,I);
tic
subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
if (isempty(subUCM))
    return;
end
I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
bCenter = round(boxCenters(lipBox));
z = zeros(dsize(I,1:2));
z(bCenter(2),bCenter(1)) = 1;
%s = exp(-bwdist(z.^2)/(.1*size(I,1)));
dd = bwdist(z)<.3*size(I,1); % clip potential edges to this range.
imageSet.faceBoxes(k,3:4)-imageSet.faceBoxes(k,1:2);
% sort ucm values, lineseg each one.
% u = unique(subUCM);
salImage = salData{k};
salImage = imresize(salImage,size(subUCM),'bilinear');
E = subUCM.*(subUCM>.1);
subUCM = E;
if (nnz(E) < 10)
    return;
end
%E = addBorder(E,3,0);
% E(E<.6) = 0;

[seglist,edgelist] = processEdges(E);
segs = seglist2segs(seglist);
[z,z_pts] = paintLines(zeros(size(E)),segs);

d = 3;

x = {};
y = {};
theta = {};
for k = 1:length(z_pts)            
    x{k} = z_pts{k}(1:d:end,1);
    y{k} = z_pts{k}(1:d:end,2);
    curSeg = segs2vecs(segs(k,:));
    theta{k} = repmat(atan2(curSeg(1),curSeg(2)),length(x{k}),1);
end

x = cat(1,x{:});
y = cat(1,y{:});

theta = 180*cat(1,theta{:})/pi;

inds = sub2ind(size(E),y,x);
sel_ = dd(inds);

x = x(sel_); y = y(sel_); theta = theta(sel_);

% if (debug_)
%     figure,imagesc(z); hold on;
%     drawedgelist(seglist);
%     plot(x,y,'r+');
%     quiver(x,y,cosd(theta),sind(theta),'g');
% end

szRange = ([.2 .4 ])*size(E,1);
alphaRange = [90 112 135 235 248 270];
% alphaRange = [-90];
allPts = zeros(4,2,length(szRange)*length(alphaRange));
k = 0;

candidates = {};
for ix = 1:length(x)
    curPt = [x(ix) y(ix)]';
    %     for iTheta = 1:length(thetaRange)
    curTheta = theta(ix);
    %curTheta = thetaRange(iTheta);
    r = rotationMatrix(pi*curTheta/180);
    u = [cosd(curTheta);sind(curTheta)];
    for iSz = 1:length(szRange)
        curSize = szRange(iSz);
        for iAlpha = 1:length(alphaRange)
            curAlpha = alphaRange(iAlpha);
            % construct a candidate u-shape.
            v1 = r*[-cosd(curAlpha);sind(curAlpha)];
            v2 = r*[cosd(curAlpha);sind(curAlpha)];
            pt1 = curPt+u*curSize/2;
            pt2 = curPt-u*curSize/2;
            pt1_1 = pt1+2*v1*curSize/2;
            pt2_1 = pt2+2*v2*curSize/2;
            k = k+1;
            allPts(:,:,k) = ([pt2_1,pt2,pt1,pt1_1]');
            
            if (all(inImageBounds(size(E),allPts(:,:,k))))            
                candidates{end+1} = fliplr(round((allPts(:,:,k))));
            end
            %                 allPts{end+1} = flipud([pt2_1,pt2,pt1,pt1_1]');
            %                 
            if (0 && debug_ &&  (toc > .01))
                tic
                clf;
                imagesc(E);
                hold on;
                curPts = allPts(:,:,k);                
                plot(curPt(1),curPt(2),'g*');
                plot(x(ix),y(ix),'rs');
                quiver(x(ix),y(ix),5*cosd(theta(ix)),5*sind(theta(ix)),0,'g');
                drawedgelist({fliplr(curPts)},size(E),2,'r')
%                 plot(
                xlim([1 size(E,2)]);
                ylim([1 size(E,1)]);
                
                
%                 pause(.01)
                drawnow;
            end
            %                 tic;
            %                 end
        end
        %         end
    end
end


% for k = 1:length(candidates)
%     candidates{k} = seglist2segs(candidates(k));
% end


for k = 1:length(candidates)
    segs = candidates{k};
    segs = [segs(1:end-1,:) segs(2:end,:)];
    candidates{k} = segs;
end
% 1. make segment adjacency graph. 
% 2. find the angle between every couple of segments.
% 3. find sequences of segments with same turning direction.

% figure,imagesc(E);
% drawedgelist(seglist,size(E),2,'rand');

% candidates = findConvexArcs3(seglist,E,edgelist);

% where is the mouth box? sample around the mouth. 




% also add curves for each individual edge inside the ucm.
% u = unique(subUCM);
% u(u==0) = [];
% newCandidates = {};
% for k = 1:length(u)
%     subUCM_u = subUCM==u(k);
%     if (nnz(subUCM_u) < 5)
%         continue;
%     end
%     [seglist_u,~] = processEdges(subUCM_u);
%     
%     [candidates_u,inds] = splitByDirection(seglist_u);
%     newCandidates = [newCandidates,candidates_u];
% end

% candidates = [candidates,newCandidates];

if (isempty(candidates))
    return;
end

% candidates = fixSegLists(candidates);
% get the following features:
% center, width, height, color, saliency

bboxes = zeros(length(candidates),4);
y_tops = zeros(length(candidates),1);
ucmStrengths = zeros(length(candidates),1);
salStrengths = zeros(length(candidates),1);
ims = cellfun2(@(x)(paintLines(zeros(size(E)),x)>0),candidates);
ims_show = cellfun2(@(x).7*rgb2gray(I)+...
    (paintLines(zeros(size(E)),x)>0),candidates);
%mImage(ims_show); pause;
% clf; montage2(cat(3,ims_show{:}));
%  pause; return;

% calculate angle from top of image, check convexity
topCenter = [size(E,2)/2,1];
isConvex = false(size(candidates));
for iCandidate = 1:length(candidates)
    %         iCandidate
    pts = candidates{iCandidate};
    pts = [pts(1:end,1:2);pts(end,3:4)];
    pts = fliplr(pts); % x,y
    d = bsxfun(@minus,pts,topCenter);
    tg = atan2(d(:,2),d(:,1));
    x = pts(:,1);
    y = pts(:,2);
    [s,is] = sort(tg,'ascend');
    if (is(1) > is(end))
        inc = -1;
    else
        inc = 1;
    end
    x = x(is(1):inc:is(end));
    y = y(is(1):inc:is(end));
    
    pts = [x y];
    diffs = diff(pts);
    crosses = zeros(size(diffs,1)-1,1);
    for id = 1:length(crosses)
        crosses(id) = diffs(id,2)*diffs(id+1,1)-diffs(id,1)*diffs(id+1,2);
    end
    
    isConvex(iCandidate) = ~any(crosses<0);
end

%%%%%%%%
% ims = ims(isConvex);
% candidates = candidates(isConvex);

for iCandidate = 1:length(candidates)
    lineImage = ims{iCandidate};
    [y,x] = find(lineImage);
    bboxes(iCandidate,:) = pts2Box([x y]);
    %         chull = convhull(x,y);
    ucmStrengths(iCandidate) = sum(subUCM(imdilate(lineImage,ones(5)) & subUCM > 0))/nnz(lineImage);
    salStrengths(iCandidate) = mean(salImage(imdilate(lineImage,ones(5))));
    % find the x,y which span the x extend, from the top. this is the
    % "top" of the cup.
    [y,is] = sort(y,'ascend');
    x = x(is);
    xmin = min(x);
    xmax = max(x);
    x_left = find(x==xmin,1,'first');
    x_right = find(x==xmax,1,'first');
    y_tops(iCandidate) = max(y(x_left),y(x_right));
end
bboxes(:,[1 3]) = bboxes(:,[1 3])/size(E,2);
bboxes(:,[2 4]) = bboxes(:,[2 4])/size(E,1);
bc = boxCenters(bboxes);
c_score = exp(-(bc(:,1)-.5).^2*10);
horz_extent = bboxes(:,3)-bboxes(:,1);
curScore = c_score+horz_extent+ucmStrengths;
%     showSorted(ims,curScore);
y_tops = y_tops/size(E,1);

curFeats(1).bbox = bboxes;
curFeats(1).bc = bc;
curFeats(1).horz_extent = horz_extent;
curFeats(1).y_top = y_tops;
curFeats(1).ucmStrength = ucmStrengths;
curFeats(1).isConvex = isConvex;
curFeats(1).salStrength = salStrengths;

if (debug_)
    close all;
    showSorted(ims_show,ucmStrengths,5);
    pause;
%     clf;   subplot(1,2,1); imagesc(I); axis image; colormap('default');
%     subplot(1,2,2); imagesc(subUCM); axis image;
end
