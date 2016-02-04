function seg_scores = get_putative_occluders(conf,curImageData, L,im)
posemap =  90:-15:-90;

%[L.occlusionPatterns.min_dist_to_mouth]

curPose = posemap(curImageData.faceLandmarks.c);


% fprintf('pose: %d, score: %01.3f\n',curPose,curImageData.faceScore);
needStrictOcclusion = abs(curPose) <= 30;
seg_scores = [L.rprops.dot];
seg_in_face = [L.occlusionPatterns.seg_in_face];
mouth_in_seg = [L.occlusionPatterns.mouth_in_seg];
face_in_seg = [L.occlusionPatterns.face_in_seg];
dists_to_mouth = [L.occlusionPatterns.min_dist_to_mouth];
solidities = [L.rprops.Solidity];

% get the convex deficiencies
%         seg_scores([L.occlusionPatterns.seg_in_face]<.1) = 0;
%         [L.occlusionPatterns.seg_in_face]
faceBox = region2Box(L.occlusionPattern.face_mask);
% check the UCM score of the portion of the segment which actually borders
% with the non-occluded part of the face. If this is a weak border, then
% it's probably not such a strong occlusion.
[ucm,gPb_thin] = loadUCM(conf,curImageData.imageID); %#ok<*STOUT>
face_mask_smaller = imerode(L.occlusionPattern.face_mask,ones(5));
ucm_strength = zeros(1,length(L.rprops));
gpb_strength = zeros(1,length(L.rprops));
ucm = imdilate(ucm,ones(3));
for q = 1:length(L.rprops)
    curBorders = bwperim(L.occludingRegions{q});
    curBorders = curBorders & face_mask_smaller;
    
    % find the intersection of the UCM values with the current borders
    ucm_strength(q) = mean(ucm(curBorders));
    gpb_strength(q) = mean(gPb_thin(curBorders));
end

faceScale = faceBox(4)-faceBox(2);

% small hack to fix areas
% for k = 1:length(L.occludingRegions)
%     L.rprops(k).Area = nnz(L.occludingRegions{k});
% end
areaRatio = [L.rprops.Area]./nnz(L.occlusionPattern.face_mask);

d1 = areaRatio > .3;
d2 = areaRatio <  1.5;
seg_scores = seg_scores + double(d1 & d2);
seg_scores = seg_scores+double(solidities-.75);
seg_scores = seg_scores.*(solidities > .5);

%         clf; imagesc(L.occlusionPattern.mouth_mask+L.occlusionPattern.face_mask);


% lipScore = -inf; 
% if (~isempty(curImageData.lipBoxes))
%     [lipScore,iv] = max(curImageData.lipBoxes(:,5));
% end
if (isfield(curImageData,'lipScore'))
    lipScore = curImageData.lipScore;
else
    lipScore = 0;
end

if (lipScore >= 20)
    %seg_scores(:) = 0;
    seg_scores = seg_scores-lipScore/10;
end

if (needStrictOcclusion)
    %seg_scores = seg_scores.*([L.occlusionPatterns.seg_in_mouth]>0);
    %seg_scores = seg_scores.*(mouth_in_seg>=.1)
    seg_scores(mouth_in_seg==0) = seg_scores(mouth_in_seg==0)-.3;
    
    %     ovp = row(regionsOverlap2(L.occludingRegions,curImageData.actionRoi));
    % curImageData.lipScore
%     if (curImageData.lipScore >=0)
%         seg_scores(:) =-3;
%     end
    % if the mouth isn't occluded and the segment isn't in the face, yet
    % another penalty, unless the face is turning to the object.
    %     seg_scores(ovp==0 & face_in_seg < .1) = seg_scores(ovp==0 & face_in_seg < .1)-.5;
    %     seg_scores(mouth_in_seg==0 & face_in_seg < .1) = seg_scores(mouth_in_seg==0 & face_in_seg < .1) - .5;
    %             double([L.occlusionPatterns.strictlyOccluding]);
else
    
    % if the face is looking sideways, require that the angle of
    % the object is similar to that of the face
    oris = [L.rprops.Orientation];
    centroids = cat(1,L.rprops.Centroid);
    yy = sind(oris); xx = cosd(oris);
    %     clf; imagesc(im);axis image; hold on;
    %     quiver(centroids(:,1),centroids(:,2),xx',yy');
    ppp = curPose;
    vec = [sind(ppp),cosd(ppp)];
    %
    %     seg_scores = seg_scores-2*(1-abs(vec*[xx;yy]));
    %             seg_scores = seg_scores.*([L.occlusionPatterns.min_dist_to_mouth] == 0); %TODO: work on this: if the mouth is
    % slightly internal to the face, although object touches face,
    % it may fail. A more robust thing to do is view the angle of
    % the object w.r.t the face, and if it's long enough in that
    % angle....
end
%         figure,displayRegions(im,L.occlusionPattern.mouth_mask)
%         pp = [L.occlusionPatterns.angular_coverage_min_f];
pp = [L.occlusionPatterns.angular_coverage_min_f];
% figure(2); imagesc(pp); set(gca,'YTick',1:size(pp,1));
% length of touching w.r.t face perimeter

pp(isnan(pp)) =inf;
%penalty = sum(~isnan(pp))/size(pp,1);
% coverage penalty: penalize segments surrounding the face too much

% apply the penalty only for segments which seem outside the
% face region

dists = [L.occlusionPatterns.dist_coverage];


% another penalty - if not nearest the mouth.
[dd,idd] = sort(unique(dists_to_mouth),'ascend');

if (idd >= 3)
    low_rankers = dists_to_mouth > dd(3);
    seg_scores(low_rankers) = seg_scores(low_rankers)-1;
    
end
% and another - if there is another object which is significantly more occluding than
% this one.

% % [dd,idd] = sort(unique(seg_in_face),'ascend');
% % 
% % if (idd >= 3)
% %     low_rankers = dists_to_mouth > dd(3);
% %     seg_scores(low_rankers) = seg_scores(low_rankers)-1;
% %     
% % end


% dd1 = median(dists_to_mouth);
% low_rankers = idd(min(3,length(idd)):end);
% low_rankers = dists_to_mouth > dd1;

dists_to_mouth_n = dists_to_mouth/faceScale;
far_from_mouth = dists_to_mouth_n>.1;
seg_scores(far_from_mouth) = seg_scores(far_from_mouth)-.2;
seg_scores([L.occlusionPatterns.seg_in_face]>.4) = -2;
seg_scores(face_in_seg > .35) = -2;
% seg_scores(seg_in_face > .35) = -2;
from_gpb = [L.occlusionPatterns.region_type]==1;
% seg_scores(~from_gpb) = seg_scores(~from_gpb)-.1;
% seg_scores(~from_gpb) = -100;

% dont want it to be too long / thin

maj_min = [L.rprops.MajorAxisLength]./[L.rprops.MinorAxisLength];
% seg_scores(maj_min > 3) = seg_scores(maj_min > 3)-1;
% seg_scores(maj_min > 3) = 0;

% seg_scores(~d1) = -10;
seg_scores(areaRatio < .1) = seg_scores(areaRatio < .1)-.5;
seg_scores = seg_scores;
seg_scores(ucm_strength<.1) = seg_scores(ucm_strength<.1)-.5;
seg_scores = seg_scores+ucm_strength;
seg_scores(isnan(seg_scores)) = -inf;
[u,iu] = sort(seg_scores,'descend');

% coveragePenalty(iu)
% distPenalty(iu)