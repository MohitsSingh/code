function follow_geometry_2(I,lines_,ellipses_,imageSet,imageID,face_poly,mouth_poly)
% a built decision process hypothesizing and checking different object.
% start with a cup for example. I should have rules. rules always
% conditional; may be conditioned on global coordinates system.

% rules are of the form (may think later of others):
% x,y meet (or are close enough)
% x,y have local angle theta
% x intersect y
% x is contained in region y
% x above, below, left, right certain locations
% x continued (extrapolated for line, completed for ellipse) reaches point y
% x,y, parallel
% x meets y in t junction

q = find(cellfun(@any,strfind(imageSet.imageIDs,imageID)));

% extract shape parameters / features.
global pTypes;
pTypes.TYPE_LINE = 1;
pTypes.TYPE_ELLIPSE = 2;
pTypes.TYPE_POINT = 3;
pTypes.TYPE_POLYGON = 4;
pTypes.special.face = 5000;
pTypes.special.mouth = 50001;
% add all primitives in two directions to keep things simple.

% ellipses_ = [ellipses_;flipEllipses(ellipses_)];
ellipseFeats = getEllipseFeatures(ellipses_);
de = [ellipseFeats.isDegenerate];
de = de | (ellipses_(:,3) > .5*norm(dsize(I,1:2)))';
% add degenerate ellipses as lines to list of lines,
lines_ = [lines_;[cat(1,ellipseFeats(de).startPoint), cat(1,ellipseFeats(de).endPoint)]];
% lines_ = [lines_;flipLines(lines_)];
% ellipseFeats(de) = [];% (remove from ellipses?)
lineFeats = getLineFeatures(I,lines_);

posemap = 90:-15:-90;
face_pose = posemap(imageSet.faceLandmarks(q).c);
if (nargin < 6)
    face_box = imageSet.faceBoxes(q,:);
    % for now, hand code the rules.
    % get absolute locations of: face boundary, face keypoints, mouth location.
    
    [xy_c,mouth_poly,face_poly] = getShiftedLandmarks(imageSet.faceLandmarks(q));
    
    clf,imagesc(I); axis image; hold on;
    plot(xy_c(:,1),xy_c(:,2),'md');
    plot(mouth_poly(:,1),mouth_poly(:,2),'r-','LineWidth',3);
    [x_poly,y_poly] = poly2cw(face_poly(:,1),face_poly(:,2));
    [x_poly,y_poly] = inflatePolygon(x_poly,y_poly,1.1);
    face_poly_inflated = [x_poly,y_poly];
    plot(x_poly,y_poly,'m--');
else
    face_box = pts2Box(face_poly);
    [x_poly,y_poly] = poly2cw(face_poly(:,1),face_poly(:,2));
    [x_poly,y_poly] = inflatePolygon(x_poly,y_poly,1.1);
    face_poly_inflated = [x_poly,y_poly];
end

% remove features not intersecting the vicinity of the face region.
near_face_box = inflatebbox(face_box,4,'both',false);
i1 = intersectBox(lineFeats,near_face_box);
i2 = intersectBox(ellipseFeats,near_face_box);
lineFeats = lineFeats(i1);
ellipseFeats = ellipseFeats(i2);

% clf; imagesc(I); axis image; hold on;
% plotStuff(lineFeats,'r');
% plotStuff(ellipseFeats,'g');


% after having removed most candidates, calculate some more features.
ellipseFeats = getEllipseFeatures2(ellipseFeats,mouth_poly,face_poly,face_box);
lineFeats = getLineFeatures2(lineFeats,mouth_poly,face_poly,face_box);

%  add mouth, face region as special primitives.
polygonFeats = getPolygonFeatures({mouth_poly,face_poly});
polygonFeats(1).special = pTypes.special.mouth;
polygonFeats(2).special = pTypes.special.face;

primitives = [mat2cell2(row(ellipseFeats),[1 length(ellipseFeats)]),...
    mat2cell2(row(lineFeats),[1 length(lineFeats)])];
%
debug_findPrimitive(primitives,[119 102]); % line
debug_findPrimitive(primitives,[126.4 106.2]); % ellipse
% plotStuff(ellipseFeats);
% lineFeats = suppressDuplicates(lineFeats);

% find adjacency graph for primitives.
[G,IG] = findAdjacency(primitives);

% NH = makeNeighborhood(G,IG);
NH = [];


globalData.mouth_poly = polygonFeats(1);
globalData.face_poly = polygonFeats(2);
globalData.face_box = face_box;
globalData.face_pose = face_pose;

% define a state machine...
% face_box =  inflatebbox(imageSet.faceBoxes(q,:),[2 2],'both',false);
bottle_ellipses = find(findBottleEllipses(ellipseFeats,globalData)); % bottom of bottle
cup_ellipses = find(findCupEllipses(ellipseFeats));
checkCups(lineFeats,ellipseFeats,cup_ellipses);

% line_pairs = getParallelLines(primitives,G,IG);
% return;
startNodes_cup = cup_ellipses;
% startNodes_1 = [];
% startNodes_1 = find(findCandidates_1(ellipseFeats,mouth_poly,face_poly,face_box));
startNodes_straw = find(findCandidates_2(lineFeats,mouth_poly,face_poly,face_box));
startNodes_bottle = bottle_ellipses;
startNodes_straw = startNodes_straw + length(ellipseFeats);
debugInfo.I = I;

states.state_initial = 1; % initial
states.state_first_ellipse = 2;
states.state_ellipse_perp_line = 3;
states.state_ellipse_perp_line_2 = 4;
states.state_straw_1 = 5;
states.state_straw_to_ellipse = 6;
states.state_first_bottle_ellipse = 7;
states.state_bottle_line_1 = 8;

states.state_accept = 1000;
nextState = [];
nextStateParams = [];


% runStateMachine(primitives,NH,G,IG,startNodes_cup,states,states.state_first_ellipse, globalData,debugInfo);
% runStateMachine(primitives,NH,G,IG,startNodes_straw,states,states.state_straw_1, globalData,debugInfo);
runStateMachine(primitives,NH,G,IG,startNodes_bottle,states,states.state_first_bottle_ellipse, globalData,debugInfo);


function runStateMachine(primitives,NH,G,IG,startNodes,states,startState,globalData,debugInfo)

if (isempty(startNodes))
    warning('state machine: no candidate ellipses found. aborting');
    return;
end


global pTypes;
% G is the adjacency graph.

if (isempty(startNodes))
    warning('state machine: no candidate ellipses found. aborting');
    return;
end

S = initState(startNodes,startState);
I = debugInfo.I;
baseDistance = 15;
distDelta = 5;
while ~S.state.isempty()
    [currentState,currentNode,currentParams,currentVis] = getStateAndNode(S);
    currentNode
    gg = G(currentNode,:);
    if (currentParams.depth > 5)
        disp('reached maximum depth');
        continue;
    end
    gg(gg > (baseDistance + currentParams.depth*distDelta)) = 0;
    N = find(gg);
    %some visualizations: find neighbors of current state (usually helpful).
    currentVis = currentVis{1};
    clf; imagesc(I); axis image; hold on;
    for k = 1:length(currentVis)
        p_ = primitives{currentVis(k)};
        plot(p_.xy(:,1),p_.xy(:,2),'G','LineWidth',2);
    end
    
    p = primitives{currentNode};
    ig = IG(currentNode,N);
    for iN = 1:length(N)
        curNeighbor = N(iN);
        % don't allow parents to be neighbors :-/
        if (currentParams.parent == curNeighbor)
            continue;
        end
        
        
        if (ismember(curNeighbor, currentVis))
            %             disp('skipping, already visited');
            continue;
        end
        
        pN = primitives{curNeighbor};
        f = getPairwiseFeatures(p,pN,ig(iN));
        %   (state,stateParams,p1,p2,f,rule,states)
        % apply the transition rule.
        % TODO : As a result, a new compound primitive can be created,
        % in which case we need to update the adjacency matrix.
        [nextState,stateParams] = applyRule(currentState,currentParams,p,pN,f,[],states,primitives,globalData);
        if (isempty(nextState))
            continue;
        end
        stateParams.parent = currentNode;
        if (nextState==states.state_accept)
            clc
            disp('accepting');
            clf; imagesc(I); axis image; hold on;
            for k = 1:length(currentVis)
                p_ = primitives{currentVis(k)};
                plot(p_.xy(:,1),p_.xy(:,2),'G','LineWidth',2);
            end
            p_ = primitives{curNeighbor};
            plot(p_.xy(:,1),p_.xy(:,2),'m','LineWidth',2);
            pause;
        end
        stateParams.depth = currentParams.depth+1;
        addStateAndNode(S,nextState,curNeighbor,stateParams,[currentVis curNeighbor])
    end
end

function E = findCandidates_1(ellipseFeats,mouth_poly,face_poly,face_box)

% slightly enlarge the face polygon
[x_poly,y_poly] = poly2cw(face_poly(:,1),face_poly(:,2));
[x_poly,y_poly] = inflatePolygon(x_poly,y_poly,1.1);

ellipse_centers = cat(1,ellipseFeats.center);
ellipse_to_mouth = bsxfun(@minus,mean(mouth_poly),ellipse_centers);
%first, consider only ellipses that are near the mouth area.
% also, the nearest point to the mouth must be quite near.

nearFace = fevalArrays(ellipseFeats,@(x) any(inBox(face_box,x.xy(:,1),x.xy(:,2))));
% want the rim quite close to the mouth center
%inFaceRegion = fevalArrays(ellipseFeats,@(x)any(inpolygon(x.xy(:,1),x.xy(:,2),x_poly,y_poly)));

nearMouth = fevalArrays(ellipseFeats,@(x)min(l2(mean(mouth_poly,1),x.xy))).^.5;

% remove the ellipses not "pointing" towards the mouth:
% for frontal faces, this means that ellipses below the mouth, should point
% towards it but those above the mouth should point away from it.
ellipse_boxes = cat(1,ellipseFeats.bbox);
ellipse_n = cat(1,ellipseFeats.middleVector);

for q = 1:length(ellipseFeats)
    ee = ellipseFeats(q);
    plot(ee.xy(:,1),ee.xy(:,2));
end

above = ellipse_boxes(:,4) <= min(mouth_poly(:,2));
goods = sum(ellipse_to_mouth.*ellipse_n,2);
goods(above) = -goods(above);
req = goods > 0 & nearFace(:) & (nearMouth <= 20)';

plotBoxes(ellipse_boxes(req,:),'r','LineWidth',2)

E = req;
function E = findCupEllipses(ellipseFeats)

ellipse_n = cat(1,ellipseFeats.middleVector);
ellipse_to_face = cat(1,ellipseFeats.toFace);
ellipse_to_mouth = cat(1,ellipseFeats.toMouth);
ellipse_to_center = cat(1,ellipseFeats.toFaceCenter);
inFace = cat(1,ellipseFeats.inFace);
goods = sum(ellipse_n.*ellipse_to_face,2) > 0;
goods = goods & (ellipse_to_center > .3 & ellipse_to_center < 1.0);
% goods = goods & sum(ellipse_to_mouth.^2,2).^.5 <= 20;
goods = goods & inFace > .5;
% also want the middle vector to be roughly up (for frontal faces).
ellipse_n_n = normalize_vec(ellipse_n,2);
goods = goods & abs(ellipse_n_n*[0 1]') > .6;


for q = 1:length(ellipseFeats)
    if (goods(q))
        ee = ellipseFeats(q);
        plot(ee.xy(:,1),ee.xy(:,2),'r-','LineWidth',2);
    end
end
pause
disp('press any key to continue');
%
% above = ellipse_boxes(:,4) <= min(mouth_poly(:,2));
% goods = sum(ellipse_to_mouth.*ellipse_n,2);
% goods(above) = -goods(above);
% req = goods > 0 & nearFace(:) & (nearMouth <= 20)';

% plotBoxes(ellipse_boxes(goods,:),'r','LineWidth',2)

E = goods;

function E = findBottleEllipses(ellipseFeats,globalData)

ellipse_n = cat(1,ellipseFeats.middleVector);
ellipse_to_face = cat(1,ellipseFeats.toFace);
ellipse_to_mouth = cat(1,ellipseFeats.toMouth);
ellipse_to_center = cat(1,ellipseFeats.toFaceCenter);
inFace = cat(1,ellipseFeats.inFace);
ellipse_n_n = normalize_vec(ellipse_n,2);
ellipse_to_mouth_n = normalize_vec(ellipse_to_mouth,2);
goods = (sum(ellipse_n_n.*ellipse_to_mouth_n,2)) > .9;
goods = goods & (inFace < .1); % want bottle ellipses to be out of face!

face_pose = globalData.face_pose;
if (face_pose < -10) % left of face...
    goods = goods & (ellipse_to_mouth_n*([-1 0]') > 0);
else (face_pose > 10)% right of face...
    goods = goods & (ellipse_to_mouth_n*([1 0]') < 0);
end % otherwise, no strong preference.

goods = goods & (ellipse_to_center); % ?

for q = 1:length(ellipseFeats)
    
    if (goods(q))
        ee = ellipseFeats(q);
        plot(ee.xy(:,1),ee.xy(:,2),'m-','LineWidth',2);
    end
end
% pause
% disp('press any key to continue');
% %
% above = ellipse_boxes(:,4) <= min(mouth_poly(:,2));
% goods = sum(ellipse_to_mouth.*ellipse_n,2);
% goods(above) = -goods(above);
% req = goods > 0 & nearFace(:) & (nearMouth <= 20)';

% plotBoxes(ellipse_boxes(goods,:),'r','LineWidth',2)

E = goods;

function E = findCandidates_2(lineFeats,mouth_poly,face_poly,face_box)

% find lines whos continuation might intersect the mouth polygon;

segs = [cat(1,lineFeats.startPoint) cat(1,lineFeats.endPoint)];
% one of the ends should be strictly below the mouth.

% find the distance between center of mouth and each of the lines.
mouth_center = mean(mouth_poly,1);
E = distancePointLine(mouth_center, [segs(:,1:2),segs(:,3:4)-segs(:,1:2)]);

E = E < 10;

mouth_box = pts2Box(mouth_poly);
ymin = min(segs(:,2),segs(:,4));
E = E & ymin >= mouth_box(2);

test_point = [82 100];
a = bsxfun(@minus,segs(:,3:4),test_point);
FF = find(sum(a.^2,2).^.5 < 5);
d1 = col(min(l2(mouth_center,segs(:,1:2)),l2(mouth_center,segs(:,3:4))).^.5);
% make sure that the starting point is not too far from the mouth.
E = E & d1 <= 50;

plotStuff(lineFeats(E),'y');


function feats = getPolygonFeatures(polygons)
global pTypes;
feats = struct('params',{},'xy',{},'center',{},'bbox',{},'startPoint',{},'endPoint',{});
for k = 1:length(polygons)
    a = polygons{k};
    feats(k).params = a;
    xy = [a(:,1) a(:,2)];
    feats(k).xy = xy;
    feats(k).bbox = pts2Box(xy);
    feats(k).center = mean(a,1);
    feats(k).startPoint = feats(k).center;
    feats(k).endPoint= feats(k).center;
    feats(k).type = 'polygon';
    feats(k).typeID = pTypes.TYPE_POLYGON;
    feats(k).special = 0;
end

function ellipseFeats = getEllipseFeatures2(ellipseFeats,mouth_poly,face_poly,face_box)
ellipse_centers = cat(1,ellipseFeats.center);
mouthCenter = mean(mouth_poly,1);
face_center = boxCenters(face_box);
ellipse_to_face = fevalArrays(ellipseFeats,@(x) mean(bsxfun(@minus,face_center,x.xy),1));
ellipse_to_face = squeeze(ellipse_to_face)';
ellipse_to_mouth = fevalArrays(ellipseFeats,@(x) mean(bsxfun(@minus,x.xy,mouthCenter),1));
ellipse_to_mouth = squeeze(ellipse_to_mouth)';
ellipse_to_center = fevalArrays(ellipseFeats,@(x) mean(x.xy(:,2)-face_box(2))/(face_box(4)-face_box(2)));
ellipse_bbox = squeeze(fevalArrays(ellipseFeats,@(x) pts2Box(x.xy)))';
ellipse_face_ovp = boxesOverlap(ellipse_bbox,face_box);
x_poly = face_poly(:,1);
y_poly = face_poly(:,2);
[x_poly,y_poly] = poly2cw(x_poly,y_poly);
[x_poly,y_poly] = inflatePolygon(x_poly,y_poly,1.3);
pts_in_face = fevalArrays(ellipseFeats,@(x)mean(inpolygon(x.xy(:,1),x.xy(:,2),x_poly,y_poly)));
for k = 1:length(ellipseFeats)
    ellipseFeats(k).toFaceCenter = ellipse_to_center(k);
    ellipseFeats(k).toFace = ellipse_to_face(k,:);
    ellipseFeats(k).toMouth = ellipse_to_mouth(k,:);
    ellipseFeats(k).bbox = ellipse_bbox(k,:);
    ellipseFeats(k).face_box_ovp = ellipse_face_ovp(k);
    ellipseFeats(k).inFace = pts_in_face(k);
    
end

function lineFeats = getLineFeatures2(lineFeats,mouth_poly,face_poly,face_box)
mouthCenter = mean(mouth_poly,1);
face_center = boxCenters(face_box);
line_to_face = fevalArrays(lineFeats,@(x) mean(bsxfun(@minus,face_center,x.xy),1));
line_to_face = squeeze(line_to_face)';
line_to_mouth = fevalArrays(lineFeats,@(x) mean(bsxfun(@minus,x.xy,mouthCenter),1));
line_to_mouth = squeeze(line_to_mouth)';
line_to_center = fevalArrays(lineFeats,@(x) mean(x.xy(:,2)-face_box(2))/(face_box(4)-face_box(2)));
line_bbox = squeeze(fevalArrays(lineFeats,@(x) pts2Box(x.xy)))';
line_face_ovp = boxesOverlap(line_bbox,face_box);
x_poly = face_poly(:,1);
y_poly = face_poly(:,2);
[x_poly,y_poly] = poly2cw(x_poly,y_poly);
pts_in_face = fevalArrays(lineFeats,@(x)mean(inpolygon(x.xy(:,1),x.xy(:,2),x_poly,y_poly)));
for k = 1:length(lineFeats)
    lineFeats(k).toFaceCenter = line_to_center(k);
    lineFeats(k).toFace = line_to_face(k,:);
    lineFeats(k).toMouth = line_to_mouth(k,:);
    lineFeats(k).bbox = line_bbox(k,:);
    lineFeats(k).face_box_ovp = line_face_ovp(k);
    lineFeats(k).inFace = pts_in_face(k);
    
end

function [G,IG] = findAdjacency(primitives)

p1 = zeros(length(primitives),2);
p2 = zeros(length(primitives),2);

for k = 1:length(primitives)
    p = primitives{k};
    p1(k,:) = p.startPoint;
    p2(k,:) = p.endPoint;
end

[G,IG] = min(cat(3,l2(p1,p1),l2(p1,p2),l2(p2,p1),l2(p2,p2)),[],3);
G = G.^.5; % get actual distances.
% sparsify
% G = G.*(G < 30);
G(eye(size(G,1))>0) = 0; % remove self-adjacencies.

function addStateAndNode(S,state,node,prms,vis)
if (nargin < 4)
    prms = [];
end
if (nargin < 5)
    vis = [];
end
S.state.push([state node]);
S.params.push(prms);
S.vis.push({vis});
function [state node params vis] = getStateAndNode(S)
M = S.state.pop();
state = M(1);
node = M(2);
params = S.params.pop();
vis = S.vis.pop();

function S = initState(startNodes,startState)
S.state = CStack;
S.params = CStack;
S.vis = CStack;
for k = 1:length(startNodes)
    addStateAndNode(S,startState,startNodes(k),struct('parent',0,'depth',0),startNodes(k));
end

function f = getPairwiseFeatures(p,pN,g12)
global pTypes;
% check the nature of the adjacency.
f = struct('flipStart',false,'flipEnd',false,'turnDirection',{});
f(1).flipStart = false;
switch g12
    case 1
        f.flipStart = true;
        p11 = p.endPoint; p12 = p.startPoint;
        p21 = pN.startPoint; p22 = pN.endPoint;
    case 2
        f.flipStart = true; f.flipEnd = true;
        p11 = p.endPoint; p12 = p.startPoint;
        p21 = pN.endPoint; p22 = pN.startPoint;
    case 3
        p11 = p.startPoint; p12 = p.endPoint;
        p21 = pN.startPoint; p22 = pN.endPoint;
    case 4
        f.flipEnd = true;
        p11 = p.startPoint; p12 = p.endPoint;
        p21 = pN.endPoint; p22 = pN.startPoint;
end

line_S = createLine(p11,p12);
line_T = createLine(p21,p22);
% get line-line features

turnDirection = 180*lineAngle(line_S,line_T)/pi;

f(1).turnDirection = turnDirection;
f(1).line_S = line_S;
f(1).line_T = line_T;


function [nextState,nextStateParams] = applyRule(state,stateParams,p1,p2,f,rule,states,primitives,globalData)
global pTypes;
nextState = [];
nextStateParams = [];

%(S,state,node,prms,vis)
plot(p2.xy(:,1),p2.xy(:,2),'r');
plot(p2.startPoint(1),p2.startPoint(2),'m+');
plot(p2.endPoint(1),p2.endPoint(2),'m+');
plot(p1.startPoint(1),p1.startPoint(2),'r+');
plot(p1.endPoint(1),p1.endPoint(2),'r+');
if (isfield(stateParams,'firstTurn'))
    nextStateParams.firstTurn = stateParams.firstTurn;
    nextStateParams.firstLine = stateParams.firstLine;
    nextStateParams.secondLine = stateParams.secondLine;
end
switch (state)
    case states.state_first_bottle_ellipse
        if (p2.typeID == pTypes.TYPE_LINE)
            % want a perperdicular line, towards the face.
            ff = f.turnDirection;
            dd = 25;
            if (abs(ff- 270)< dd || abs(ff-90) < dd)
                line_T = f.line_T;
                mouthCenter = globalData.mouth_poly.center;
                n1 = normalize_vec(mouthCenter-line_T(1:2),2);
                n2 = normalize_vec(line_T(3:4),2);
                if (n2*n1' > .7)
                    nextState = states.state_bottle_line_1;
                    % want to continue as long as we're getting closer to
                    % face.
                    
                end
            end
        end
    case states.state_bottle_line_1
        ff = f.turnDirection;
        dd = 20;
        if (abs(ff-180) < dd || ff < dd);
            line_T = f.line_T;
            mouthCenter = globalData.mouth_poly.center;
            %
            n1 = normalize_vec(mouthCenter-line_T(1:2),2);
            n2 = normalize_vec(line_T(3:4),2);
            if (n2*n1' > .7)
                
                if (isContinuationOf(f.line_S,f.line_T))
                    % check proximity to mouth...
                    
                    if (inBox(globalData.face_box,f.line_T(1:2)+f.line_T(3:4)))
%                     if (norm(f.line_T(1:2)+f.line_T(3:4)-...
%                             globalData.mouth_poly.center) < 20)
                        nextState = states.state_accept;
                    else
                        nextState = states.state_bottle_line_1;
                    end
                end
                % want to continue as long as we're getting closer to
                % face.
                
            end
        end
        
    case states.state_straw_1
        %         disp ('looking for parallel line to straw');
        if (p2.typeID == pTypes.TYPE_LINE)
            % check parallelism
            line_from = createEdge(p1.startPoint,p1.endPoint);
            %lineAngle
            
            dd = 25;
            ff = f.turnDirection;
            
            if (ff > 360-dd || ff < dd || abs(ff-180) < dd)
                [dist pos1] = distancePointEdge(p2.startPoint,line_from);
                [dist pos2] = distancePointEdge(p2.endPoint,line_from);
                
                %                 pos1_ = min(pos1,pos2);
                %                 pos2_ = pos1+pos2-pos1_; % == max(pos1,pos2)
                
                if (abs(pos2-pos1) > .6) % parralel, and "near" each other
                    % the lower point should end with an ellipse. (later:
                    % maybe with an occluding hand.
                    % so we're reduced to looking for a cup, just as in
                    % near the mouth case.
                    
                    % * if it's long enough - finished.
                    bSize = globalData.face_box(3)-globalData.face_box(1);
                    if (p1.absLength/bSize > .7)
                        nextState = states.state_accept;
                    else
                        
                        % make a new "straw" primitive
                        startPoint = p1.startPoint; endPoint = p1.endPoint;
                        if (pos1 > pos2) % flipped
                            startPoint = startPoint + p2.endPoint;
                            endPoint = endPoint + p2.startPoint;
                        else
                            startPoint = startPoint + p2.startPoint;
                            endPoint = endPoint + p2.endPoint;
                        end
                        startPoint = startPoint/2;
                        endPoint = endPoint/2;
                        
                        plotPoint(startPoint,'g*','start');
                        plotPoint(endPoint,'g*','end');
                        
                        d1 = distancePoints(startPoint,globalData.mouth_poly.center);
                        d2 = distancePoints(endPoint,globalData.mouth_poly.center);
                        if (d1 < d2)
                            nextStateParams.straw_seg = [startPoint endPoint];
                        else
                            nextStateParams.straw_seg = [endPoint startPoint];
                        end
                        
                        nextState = states.state_straw_to_ellipse;
                    end
                end
            end
        end
    case states.state_first_ellipse
        %         disp('looking for line perp to ellipse');
        %%%if (p2.typeID == pTypes.TYPE_LINE) % can be line or ellipse!
        %             if (f.flipStart)
        %                 ellipseTurnDirection = 180*angle3Points(p1.curveCenter,p11,p12)/pi;
        
        if (f.flipStart && abs(f.turnDirection - 270) < 30) || ...
                (~f.flipStart && abs(f.turnDirection - 90) < 30)
            %                 disp('line found');
            nextState = states.state_ellipse_perp_line;
            nextStateParams.turnDirection = f.turnDirection;
            nextStateParams.flipStatus = [f.flipStart f.flipEnd];
            nextStateParams.firstTurn = f.turnDirection;
            nextStateParams.firstLine = f.line_S;
            nextStateParams.secondLine = f.line_T;
            %                 nextStateParams.from = p1;
        end
        %%%  end
    case states.state_ellipse_perp_line
        %         disp('state: looking for perp 2 or ellipse');
        if (p2.typeID == pTypes.TYPE_LINE) %% look for line with same direction
            T = f.turnDirection;
            % check source ellipse turn direction.
            
            % the second line cannot be closer to the starting ellipse
            % than the furthest point in the first line.
            parentPts = primitives{stateParams.parent}.xy;
            
            %d1 = max(max(l2(p1.xy,parentPts)));
            %d2 = min(min(l2(p2.xy,parentPts)));
            
            
            b = inSameDirection(primitives{stateParams.parent},p1,p2);
            
            if (b && (T < 20 || T > 340))
                %                 disp('got perp line');
                % option 1: add a search for another line
                nextState = states.state_ellipse_perp_line_2;
                nextStateParams.flipEnd = f.flipEnd;
                % option 2: add a search for an opposite line.
                
            end
        elseif (p2.typeID == pTypes.TYPE_ELLIPSE) % look for ellipse at bottom of line...
            T = f.turnDirection;
            parentPts = primitives{stateParams.parent}.xy;
            
            [b] = inSameDirection(primitives{stateParams.parent},...
                p1,p2);
            
            b = b & abs(f.turnDirection - stateParams.firstTurn) < 30;
            
            %             d1 = max(max(l2(p1.xy,parentPts)));
            %             d2 = min(min(l2(p2.xy,parentPts)));
            % %
            % find about the connection type
            
            
            prevLine = stateParams.secondLine;
            p11 = prevLine(1:2);
            p12 = prevLine(1:2)+prevLine(3:4);
            p21 = f.line_T(1:2);
            
            r = norm(p21-p12) < norm(p21-p11);
            
            
            
            
            % if (norm(p1.toFace) < norm(p2.toFace))
            if (r && b && ((abs(f.turnDirection - 90) < 20) || (abs(f.turnDirection - 270) < 20)))
                nextState = states.state_accept;
            end
        end
    case states.state_ellipse_perp_line_2
        
        %         disp('state: looking for closing line or ellipse');
        if (p2.typeID == pTypes.TYPE_ELLIPSE)
            
            % make sure that ellipse is further from mouth than line.
            
            %             parentPts = primitives{stateParams.parent}.xy;
            %             d1 = max(max(l2(p1.xy,parentPts)));
            %             d2 = min(min(l2(p2.xy,parentPts)));
            
            % if ()
            if (norm(p1.toFace) < norm(p2.toFace))
                % check the angle...
                if abs(f.turnDirection - 270) < 20 || abs(f.turnDirection - 90) < 20
                    %                     disp('got it');
                    nextState = states.state_accept;
                end
            end
        end
    case states.state_straw_to_ellipse
        %         disp('state: looking for closing line or ellipse, for straw');
        if (p2.typeID == pTypes.TYPE_ELLIPSE)
            % turn direction should be ~90 degrees.
            p11 = stateParams.straw_seg(1:2);
            p12 = stateParams.straw_seg(3:4);
            % want the ellipse to point "downwards" w.r.t direction from
            % mouth
            t = sum(-p2.middleVector.*[p12-p11]);
            if (t > 0) % good, can continue - as if we're near the mouth.
                nextState = states.state_first_ellipse;
            end
        end
end


function lines_ = flipLines(lines_)
lines_ = lines_(:,[3 4 1 2]);

function ellipses_= flipEllipses(ellipses_)
ellipses_= ellipses_(:,[1 2 3 4 5 7 6]);

function t = intersectBox(feats,bbox)
t = false(size(feats));
for k = 1:length(t)
    t(k) = any(inBox(bbox,feats(k).xy));
end
%
% function pairs = getParallelLines(lineFeats)
% % find pairs which are not only roughly parallel, but are "across" each
% % other, meaning that their mutual projections are roughly on the lines
% % as well.
% S = cat(1,lineFeats.startPoint);
% T = cat(1,lineFeats.endPoint);
% segs = [S T];
% vecs = segs2vecs(segs);
% [X,norms] = normalize_vec(vecs');
% cos_angles = X'*X;
% % remove self-angle
% cos_angles = cos_angles.*(1-eye(size(cos_angles)));
% maxAngle = 5; % maximal angle between adjacent segments.
% [ii,jj] = find(abs(cos_angles) >= cosd(maxAngle)); % ii,jj are possible pairs of segments.
%
% allLines = createLine(S,T);
% G = zeros(size(allLines,1));
% for i = 1:size(G,1)
%
% end
%
% a = 1;

function [G,Gx,Gy] = checkCups(lineFeats,ellipseFeats,cup_ellipses)
% 1. find opposing ellipse / line.

p1 = cat(1,ellipseFeats.startPoint);
p2 = cat(1,ellipseFeats.endPoint);

Gx = zeros(length(ellipseFeats));
Gy = zeros(length(ellipseFeats));

ellipseEdges = createEdge(p1,p2);
cupEllipses = ellipseFeats(cup_ellipses);
for k = 1:length(cupEllipses)
    %     curLine = [cupEllipses(k).startPoint cupEllipses(k).endPoint];
    %     clf;hold on;
    %     plot(cupEllipses(k).xy(:,1),cupEllipses(k).xy(:,2));
    %
    %     plot(cupEllipses(k).center(1),cupEllipses(k).center(2),'g+')
    %     quiver(cupEllipses(k).center(1),cupEllipses(k).center(2),...
    %         cupEllipses(k).middleVector(1),cupEllipses(k).middleVector(2),'r');
    % intersectLineEdge (line, edge)
    middleLine = createLine(cupEllipses(k).center(1),cupEllipses(k).center(2),...
        cupEllipses(k).middleVector(1),cupEllipses(k).middleVector(2));
    %     drawLine(middleLine,'color','g');
    t = intersectLineEdge(middleLine,ellipseEdges);
    
    
    Gx(cup_ellipses(k),:) = t(:,1);
    Gy(cup_ellipses(k),:) = t(:,2);
    
    %
    %     for kk = 1:size(t,1)
    %         if (~isnan(t(kk,1)))
    %             plot(ellipseFeats(kk).xy(:,1),ellipseFeats(kk).xy(:,2),'b');
    %         end
    %     end
    %
    %
    %     ellipse_n = cupEllipses(k).middleVector
    %
    
    
end

G = ~isnan(Gx);




function debug_findPrimitive(primitives,p)

% sp_diff = cellfun(@(x) sum((x.startPoint-p).^2,2),primitives).^.5;
% ep_diff = cellfun(@(x) sum((x.endPoint-p).^2,2),primitives).^.5;
sp_diff = cellfun(@(x) sum((x.xy(1,:)-p).^2,2),primitives).^.5;
ep_diff = cellfun(@(x) sum((x.xy(end,:)-p).^2,2),primitives).^.5;

[m_start,sp_m] = min(sp_diff);
[m_end,ep_m] = min(ep_diff);

if (m_start < m_end)
    disp(['nearest element to point is ' num2str(sp_m) ' with start-point distance ' num2str(m_start)]);
else
    disp(['nearest element to point is ' num2str(ep_m) ' with end-point distance ' num2str(m_end)]);
end


function b = inSameDirection(p1,p2,p3)
seg1 = [p1.startPoint p1.endPoint];
d1 = l2(p1.xy,p2.xy);
[m12,im] = min(d1(:));
[im_12,jm_12] = ind2sub(size(d1),im);
d2 = l2(p2.xy,p3.xy);
[m23,im] = min(d2(:));
[im_23,jm_23] = ind2sub(size(d2),im);

% plot(p1.xy(im_12,1),p1.xy(im_12,2),'r+');
% plot(p2.xy(jm_12,1),p2.xy(jm_12,2),'ro');
%
% plot(p2.xy(im_23,1),p2.xy(im_23,2),'m+');
% plot(p3.xy(jm_23,1),p3.xy(jm_23,2),'mo');
%

% n1 : from the nearest point in p1 to p2, the mean direction to p2
% n2 : from the nearest point in p2 to p3, the mean direction to p3

n1 = mean(bsxfun(@minus,p2.xy,p1.xy(im_12,:)),1);
n2 = mean(bsxfun(@minus,p3.xy,p2.xy(im_23,:)),1);

%     n1 = p2.xy(jm_12,:)-p1.xy(im_12,:);
%     n2 = p3.xy(jm_23,:)-p2.xy(im_23,:);
b = n1*n2' > 0;

function b = isParallel(line1,line2)
b = false;
error('not implemented');

function b = ellipse_in_front_of(E1,E2)
b = false;
error('not implemented');

function b = isContinuationOf(line1,line2)
% check if line2 is a continuation of line 1
% edge1 = createEdge(line1);
% edge2 = createEdge(line2);
% assume point are ordered so p21 is near p12.
% n1 = normalize_vec(line1.p2-line1.p1,2);
% n2 = normalize_vec(line2.p2-line2.p1,2);

T = lineAngle(line1,line2);
d = 10;
b = T > 360 - d || T < d;
% make sure that lines are not to far apart on the orthogonal axis.
d = distancePointLine([line2(1:2);line2(3:4)+line2(1:2)],line1);
b = b && max(d) < 5;





% end
% f
%function check
