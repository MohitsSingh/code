function [pairScores,lineScores,pairExplanations,lineExplanations,pairPolys] = getStrawScores(conf,segs,props,mouth_box,M,mouth_data,imgData,debug_)
if (nargin < 7)
    debug_ = false;
end
nSegs = size(segs,1);
low_score = -100;
mouth_poly = mouth_data.mouth_poly;
mouth_width = mouth_data.mouth_width;
mouth_pt1 = mouth_data.pt_1;
mouth_pt2 = mouth_data.pt_2;
vecs = segs2vecs(segs);
[X,norms] = normalize_vec(vecs');
props(:,4) = norms;
cos_angles = X'*X;
% remove self-angle
cos_angles = cos_angles.*(1-eye(size(cos_angles)));
maxAngle = 15; % maximal angle between adjacent segments.
angles = real(acosd(cos_angles));
% if segments are quite short (relative to their distance),
%consider wider angles.
means = (segs(:,1:2)+segs(:,3:4))/2;
D = l2(means,means).^.5;
pairExplanations = cell(size(D));
lineExplanations = cell(size(D,1),1);
tooWide = (D./mouth_width)>.4;
tooThin = (D./mouth_width) < .1;

% calculate projection distances
lines_ = [segs(:,1:2),vecs];
projections = zeros(nSegs,nSegs,2);
intersections = false(nSegs,nSegs);
for k = 1:nSegs
    dists = distancePointLine(segs(k,1:2),lines_);
    projections(k,:,1) = dists;
    dists = distancePointLine(segs(k,3:4),lines_);
    projections(k,:,2) = dists;
    curInt = intersectEdges(segs(k,:),segs);
    intersections(k,:) = all(~isinf(curInt) & ~isnan(curInt),2);
    %intersections(k,:,:) = ;
end

isDup = mean(projections,3) < 2.5;


% find intersections....


[l_1,l_2] = meshgrid(norms);
maxShortAngle = 30;
mean_l = (l_1+l_2)/2;
% mean_l(2,4)
shorts = (D>=mean_l/2);%figure,imagesc2(shorts)
% D(2,4)

angleCandidates = (angles <= maxAngle) | (shorts & angles <= maxShortAngle);

[ii,jj] = find(angleCandidates);
% find the maximal distance between pairs of lines
lineScores = getScores(props);   % score for single lines.
maxDist = 8*conf.straw.dim/50;
pairScores = double(D<=maxDist);
%pairScores = pairScores.double(angles <= 15);
pairScores = pairScores-(max(D,maxDist)-maxDist).^2;
%[14,12]
% check coverings...
% angle of straw should correspond to it's starting point.
d = props(:,1);
[d1,d2] = meshgrid(d);

% should roughly point to center of mouth.
mouth_center = conf.straw.dim/2;
mouth_center = [mouth_center mouth_center];
mean_angle = abs(d);
vec_angle = [cosd(d),sind(d)];
start_pts = props(:,5:6);
dist_1 = bsxfun(@minus,segs(:,1:2),mouth_center);
[~,dist_1] = normalize_vec(dist_1,2);
dist_2 = bsxfun(@minus,segs(:,3:4),mouth_center);
[~,dist_2] = normalize_vec(dist_2,2);
[~,near_mouth] = min([dist_1,dist_2],[],2);


% rearrange each segment so first point is closer to mouth.
segs_1 = zeros(size(segs));
for k = 1:nSegs
    if (near_mouth(k)==1)
        segs_1(k,:) = segs(k,:);
    else
        segs_1(k,:) = segs(k,[3 4 1 2]);
    end
end

% now measure the projection of start-points and end points of each segment
% on mouth_line.

mouth_line = createEdge(mouth_pt1,mouth_pt2);
[dist1 pos1,pos1_notrunc] = distancePointEdge(segs_1(:,1:2),mouth_line);
[dist2 pos2] = distancePointEdge(segs_1(:,3:4),mouth_line);

badConfiguration = abs(pos1-.5)>.25 & (abs(pos1-.5)>abs(pos2-.5));

lineScores(badConfiguration) = low_score;
lineExplanations = appendString(lineExplanations,col(badConfiguration),[],'bad configuration',debug_);

% tooExtreme = pos1 < .1 | pos1 > .9;
% lineScores(tooExtreme) = low_score;
% lineExplanations = appendString(lineExplanations,col(tooExtreme),[],'too extreme',debug_);


% find if intersection with edge is on edge.
intersectsMouth = intersectLineEdge(lines_,repmat(mouth_line,nSegs,1));
intersectsMouth_b = all(~isinf(intersectsMouth) & ~isnan(intersectsMouth),2);

[v1,v2] = meshgrid(intersectsMouth_b,intersectsMouth_b);
pairNotIntersectsMouth = ~v1 & ~v2;
pairScores(pairNotIntersectsMouth) = low_score;
pairExplanations = appendString(pairExplanations,pairNotIntersectsMouth,[],'fall outside mouth',debug_);

% [ii,jj] = find(v1 | v2);


% [b1,b2] = meshgrid(intersectsMouth_b);

% [proj1,proj2] = meshgrid(pos1); % projection positions.
% intersectsMouth(~intersectsMouth_b,:) = pos1(~intersectsMouth_b);
[~, pos_of_intersection] = distancePointEdge(intersectsMouth,mouth_line);
pos_of_intersection(isnan(pos_of_intersection)) = pos1(isnan(pos_of_intersection));
[p_1,p_2] = meshgrid(pos_of_intersection);
int_width = abs(p_1-p_2);

% [~,int_width] = normalize_vec(intersectsMouth-intersectsMouth,2);
int_width_large = int_width > .4;% & ~isnan(int_width);
% int_width_large(v1 & v2) = int_width/mouth_width>.4 & ~isnan(int_width) & ~isinf(int_width);
pairScores(int_width_large) = low_score;
pairExplanations = appendString(pairExplanations,int_width_large,[],'span too much of mouth',debug_);


% fallOutsideMouth = ~intersectsMouth; %pos1 < 0 | pos1 >= 1;
% lineScores(fallOutsideMouth) = low_score;
% lineExplanations = appendString(lineExplanations,col(fallOutsideMouth),[],'fall outside mouth',debug_);

% for each pair of segments, the points near the mouth
% should be at least as close as those far away from it, but check this
% only for angles which deviate too much from parallel

wrongShapes = false(size(pairScores));

for iPair = 1:length(ii)
    i1= ii(iPair); j1 = jj(iPair);
    %     if (angles(i1,j1) > 5)
    seg1 = segs(i1,:);
    seg2 = segs(j1,:);
    n1 = near_mouth(i1)-1;
    n2 = near_mouth(j1)-1;
    
    d_near = norm(seg1(2*n1 + [1:2])-seg2(2*n2 + [1:2]));
    n1 = 1-n1;n2 = 1-n2;
    d_far = norm(seg1(2*n1 + [1:2])-seg2(2*n2 + [1:2]));
    % %     clf; imagesc2(M);
    % %     drawedgelist(segs2seglist(seg1([2 1 4 3])),size2(M),2,[1 0 0]);
    % %     drawedgelist(segs2seglist(seg2([2 1 4 3])),size2(M),2,[1 0 0]);
    
    if (d_near-d_far) >= max(.2*d_near,3)
        wrongShapes(i1,j1) = true;
        % % % % %         pairScores(i1,j1) = low_score;
        % % % % %         pairExplanations = appendString(pairExplanations,i1,j1,'wrong shape',debug_);
        %         explanations{i1,j1} = ['wrongly shaped, ',explanations{i1,j1}];
    end
end

fars = props(:,9) > conf.straw.dim/10;
[v1,v2] = meshgrid(fars,fars);
fars = v1 & v2;
far_and_bad_shape = fars & wrongShapes & angles > 10; % check shape only for big angles.
pairScores(far_and_bad_shape) = low_score;
pairExplanations = appendString(pairExplanations,far_and_bad_shape,[],'far and badly shaped',debug_);

% subplot(2,3,2);showCoords(mouth_poly_2,'Color','g');

% calculate angle using further point
vec_to_center = bsxfun(@minus,segs(:,1:2),mouth_center);
vec_to_center2 = bsxfun(@minus,segs(:,3:4),mouth_center);
[~,n1] = normalize_vec(vec_to_center,2);
[~,n2] = normalize_vec(vec_to_center2,2);
for q = 1:length(n1)
    if (n2(q)>n1(q))
        vec_to_center(q,:) = vec_to_center2(q,:);
    end
end

% vec to center should also agree with look direction of face.
posemap =  90:-15:-90;
face_look_direction = posemap(imgData.faceLandmarks.c);
%
% figure(2); imagesc2(M); quiver(means(:,1),means(:,2),vec_to_center(:,1),vec_to_center(:,2));
% quiver(segs(:,1),segs(:,2),vecs(:,1),vecs(:,2),0,'r','LineWidth',2);

[vec_to_center,~] = normalize_vec(vec_to_center,2);

face_vec = [sind(face_look_direction) cosd(face_look_direction)];
dots_look_direction = sum(bsxfun(@times,vec_to_center,face_vec),2);

dots = sum(vec_to_center.*vec_angle,2);
lineScores = lineScores + dots.^2;
d_dots = acosd(abs(dots));
% lineScores(d_dots>30) = low_score;
startPt_x = props(:,5);
v = (startPt_x-conf.straw.dim/2);
% pairScores(angles > 20)= low_score;
covers_ = zeros(size(pairScores));
[ii,jj] = find(pairScores > low_score);
% [ii,jj] = find(lineScores);

% projections = zeros([size(pairScores)]);

for k = 1:length(ii)
    u = ii(k); v = jj(k);
    p1.startPoint = segs(u,1:2);p1.endPoint = segs(u,3:4);
    p2.startPoint = segs(v,1:2);p2.endPoint = segs(v,3:4);
    %     line_ = createLine(p2.startPoint,p2.endPoint);
    
    %     hold on; drawEdge(segs(u,:));
    %     drawEdge(segs(v,:));
    covers_(u,v) = covers(p1,p2);
end
% pairScores(covers_==0) = low_score;
% pairExplanations = appendString(pairExplanations,covers_==0,[],'no cover',debug_);
minCover = min(covers_,covers_');
maxCover = max(covers_,covers_');
maxCover_t = maxCover<=.45;
pairScores(maxCover_t) = low_score;%TODO
pairExplanations = appendString(pairExplanations,maxCover_t,[],'max cover too small',debug_);

top_y = props(:,8);
bottom_y = props(:,11);
mouth_center_y = mean(mouth_box([2 4]),2);

mouth_top_y = min(mouth_poly(:,2));
lineScores(top_y < mouth_top_y) = low_score;
lineExplanations = appendString(lineExplanations,col(top_y < mouth_top_y),[],'top y above mouth center',debug_);
lineScores(bottom_y < mouth_center_y) = low_score;
lineExplanations = appendString(lineExplanations,col(bottom_y < mouth_center_y),[],'bottom y above mouth center',debug_);

% also make sure that for each straw at least one of the ends is entirely
% below the lips



lineScores(dots_look_direction < 0) = low_score;
lineExplanations = appendString(lineExplanations,col(dots_look_direction < 0),[],'direction away with face',debug_);
%if (abs(face_look_direction) > 30) % since even 30 could mean frontal
angles_from_look_direction = acosd(dots_look_direction);
lineScores(abs(angles_from_look_direction)> 60) = low_score;
lineExplanations = appendString(lineExplanations,col(abs(angles_from_look_direction)> 45),[],'direction from face > 45',debug_);
%end

pairScores(~angleCandidates) = low_score;
pairExplanations = appendString(pairExplanations,~angleCandidates,[],'bad angles',debug_);

[d_dots1,d_dots2] = meshgrid(d_dots);
d_dots = (d_dots1+d_dots2)/2;

pairScores(d_dots>30) = low_score;
pairExplanations = appendString(pairExplanations,d_dots>30,[],'dots>30',debug_);
pairScores(tooWide) = low_score;
pairExplanations = appendString(pairExplanations,tooWide,[],'too wide',debug_);
pairScores(tooThin) = low_score;
pairExplanations = appendString(pairExplanations,tooThin,[],'too thin',debug_);
pairScores(isDup) = low_score;
pairExplanations = appendString(pairExplanations,isDup,[],'dup line',debug_);
pairScores(intersections) = low_score;
pairExplanations = appendString(pairExplanations,isDup,[],'lines intersect',debug_);

% lineScores = lineScores + norms'/10;
% lineScores = lineScores-abs(abs(d)-90)/10;
% lineScores(abs(d) < 40) = 0;
lineScores(norms<=5) = low_score;
lineExplanations = appendString(lineExplanations,col(norms<=5),[],'too short',debug_);

% check if entire straw is inside face...
tt = props(:,12)>=.8;
lineScores(tt) = lineScores(tt)-1;
lineExplanations = appendString(lineExplanations,col(tt),[],'straw inside face',debug_);

tt = props(:,13)>=.5;
lineScores(tt) = low_score;
lineExplanations = appendString(lineExplanations,col(tt),[],'part of face perim',debug_);

short_norms = norms <= .3*mouth_width;

% min mouth dist: props(:,9)
tt = short_norms &  props(:,9)' > 3;
lineScores(tt) = low_score;
lineExplanations = appendString(lineExplanations,col(tt),[],'short and far from mouth',debug_);

% for each "straw" find if it crosses the mouth or not.
pairPolys = {};

[ii,jj] = find(pairScores > 0);
for iPair = 1:length(ii)
    %         iPair
    %     i1 = 84;j1 = 104;
    i1 = ii(iPair);j1 = jj(iPair);
    seg1 = segs(i1,:);
    seg2 = segs(j1,:);
    
    if (min(lineScores(i1),lineScores(j1)) < 0)
        continue;  % no need to check this pair
    end
    
    %
    %         debug_ = false;    
    centerPoint = mean(mouth_poly,1);        
            scale = size(M,1)/5;
    %scale = [scale,D(i1,j1)];
    scale = [scale,2];
    %     scale = [scale,scale/2];
    reverse = true;
    [xx_orig,u_major,u_minor,xy] =  segPairToPolygon(seg1,seg2,centerPoint,scale,false,[],false);
    [xx,u_major,u_minor,xy] =       segPairToPolygon(seg1,seg2,centerPoint,scale,false,[],true);
    pairPolys{i1,j1,1} = xx_orig;
    pairPolys{i1,j1,2} = xx;
    %         xx_t = (xx(1,:)+xx(2,:))/2;
    
    % measure the gpb inside this region
    %         xx_orig_t = xx_orig*conf.straw.dim/s1;
%             clf,imagesc2(M);plotPolygons(xx_orig,'r-+','LineWidth',2);
%            plotPolygons(xx,'g-+','LineWidth',2);pause;
    
    %
    %[xx,u_major,u_minor,xy] = segPairToPolygon(seg1,seg2,centerPoint,scale,false);
    %     [seg1';seg2']
    %         clf,imagesc2(M);plotPolygons(xx_orig,'r-+','LineWidth',2);
    %         plotPolygons(xx,'m--','LineWidth',2);
    %
    %         plot(xx_t(1),xx_t(2),'md','MarkerFaceColor','y','MarkerEdgeColor','r','MarkerSize',11)'
    %         plotPolygons(xy,'g--','LineWidth',2);
    %         pause
    %
    
    
end

% longs = norms>=size(M,1)/5;
% lineScores(longs) = lineScores(longs)+2;
end
