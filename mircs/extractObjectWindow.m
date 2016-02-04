function [obj_sub,inferredRegion,orig_poly,target_rect_size] = extractObjectWindow(conf,curImageData,segData,debug_)
obj_sub = [];

[I,I_rect] = getImage(conf,curImageData);
[M,landmarks,face_box,face_poly,mouth_box,mouth_poly] = getSubImage(conf,curImageData,conf.straw.extent,true);
s1 = size(M,1);
sz = conf.straw.dim;
M = imResample(M,[sz sz],'bilinear');

[s,is] = sort(segData.totalScores(:),'descend');
if (isempty(s))% || s(1) < 0)
%     drawnow;pause;
    return;
end
mouth_poly = fix_mouth_poly(mouth_poly);
totalScores = segData.totalScores;

% get the general direction of the straw and the scale of face to
% extract a region for training.
%     if (0)
qq = is(1);
ij = ind2sub2(size(totalScores),1:length(totalScores(:)));
ii = ij(qq,1); jj = ij(qq,2);

seg1 = segData.segs(ii,:);seg2 = segData.segs(jj,:);
mouth_center = mean(mouth_poly,1);
faceScale = face_box(4)-face_box(2);

seg1 = seg1/(conf.straw.dim/s1)+face_box([1 2 1 2]);
seg2 = seg2/(conf.straw.dim/s1)+face_box([1 2 1 2]);



asp_ratio = [.7 .3];

[inferredRegion,u_major,u_minor,orig_poly] = segPairToPolygon(seg1,seg2,mouth_center,[faceScale faceScale].*asp_ratio,debug_,I);

k = convhull(orig_poly); orig_poly = orig_poly(k,:);
% t1 = orig_poly(:,1);t2 = orig_poly(:,2);
% [t1 t2] = poly2cw(t1,t2); orig_poly = [t1 t2];

% % inferredRegion = bsxfun(@minus,inferredRegion,u_major*faceScale/5);
% %TODO....

% shift everything a bit up so we include the upper lip as well

if (debug_)
    plotBoxes(face_box,'m--');
    plotPolygons(inferredRegion,'g--','LineWidth',2);
    plot(mouth_center(1),mouth_center(2),'md','MarkerSize',11,'MarkerFaceColor','g');
    
    subplot(2,2,1); 
    imagesc2(I);
    drawedgelist(segs2seglist([seg1([2 1 4 3]);seg2([2 1 4 3])]),size2(I),2,[0 1 0]);
end
r1 = asp_ratio(2); r2 = asp_ratio(1);
ss = 200;
target_rect_size = ss*[r1 r2];
[obj_sub] = rectifyWindow(I,inferredRegion,target_rect_size);


obj_sub = clip_to_bounds(obj_sub);
if (debug_)
    subplot(2,2,2); imagesc2(obj_sub);    
    drawnow;
end