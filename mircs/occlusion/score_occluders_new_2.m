function seg_scores = score_occluders_new_2(I,occlusionPattern, faceLandmarks, regions,occlusionPatterns)
posemap =  90:-15:-90;

% curPose = posemap(faceLandmarks.c);
%[rprops,x] = extractShapeFeatures(fillRegionGaps(regions));
[rprops,x] = extractShapeFeatures(regions);
% needStrictOcclusion = abs(curPose) <= 30;
seg_scores = zeros(size(regions));
seg_in_face = [occlusionPatterns.seg_in_face];
mouth_in_seg = [occlusionPatterns.mouth_in_seg];
face_in_seg = [occlusionPatterns.face_in_seg];
dists_to_mouth = [occlusionPatterns.min_dist_to_mouth];
solidities = [rprops.Solidity];

faceBox = (occlusionPattern.faceBox);
faceScale = faceBox(4)-faceBox(2);

areaRatio = [rprops.Area]./nnz(occlusionPattern.face_mask);
seg_scores = row(seg_scores);
seg_scores = seg_scores+double(solidities-.75);
seg_scores = seg_scores.*(solidities > .5);
% if (isfield(curImageData,'lipScore'))
%     lipScore = curImageData.lipScore;
% else
%     lipScore = 0;
% end
% if (lipScore >= 20)
%     seg_scores = seg_scores-lipScore/10;
% end
% if (needStrictOcclusion)
%     seg_scores(mouth_in_seg==0) = seg_scores(mouth_in_seg==0)-.3;
% end
pp = [occlusionPatterns.angular_coverage_min_f];
% figure(2); imagesc(pp); set(gca,'YTick',1:size(pp,1));
% length of touching w.r.t face perimeter

pp(isnan(pp)) =inf;
%penalty = sum(~isnan(pp))/size(pp,1);
% coverage penalty: penalize segments surrounding the face too much

coveragePenalty = sum(pp<=4)/size(pp,1);
% apply the penalty only for segments which seem outside the
% face region
coveragePenalty(seg_in_face>.2) = 0;

dists = [occlusionPatterns.dist_coverage];

%
if (size(faceLandmarks.xy,1)==68)
    bb_1 = setdiff(1:size(dists,1),11:18);
    bb_2 = setdiff(1:size(dists,1),[1:7 27]); % TODO - what if the cup/bottle is from below? 
    
    distPenalty1 = sum(dists(bb_1,:) <= max(5,faceScale*.1));
    distPenalty2 = sum(dists(bb_2,:) <= max(5,faceScale*.1));
    required_1 = 12:15; required_2 = 3:6;
    vicinityPenalty_1 = sum(dists(required_1,:) > max(5,faceScale*.1));
    vicinityPenalty_2 = sum(dists(required_2,:) > max(5,faceScale*.1));
    vicinityPenalty = [vicinityPenalty_1;vicinityPenalty_2];
    [distPenalty,iP] = min([distPenalty1;distPenalty2]);
    vicinityPenalty = vicinityPenalty(sub2ind(size(vicinityPenalty),iP,1:length(vicinityPenalty_1)));
    vicinityPenalty(vicinityPenalty<4) = 0;
    forbiddenPoints = 18:27;
else
    bb = setdiff(1:size(dists,1),2:11);
    %forbiddenPoints = 18:27;
    distPenalty = sum(dists(bb,:) <= max(5,faceScale*.1));
    %requiredPoints = 6:10;
    requiredPoints = 7:9;
    vicinityPenalty = sum(dists(requiredPoints,:) > max(5,faceScale*.1));
    % at least two points should be covered.
%     vicinityPenalty(vicinityPenalty<4) = 0;
end

%
%
%
distPenalty(face_in_seg>.2) = distPenalty(face_in_seg>.2)*.5;
% in case the segment is well inside the face, we expect it to cover the
% mouth.
distPenalty_orig = distPenalty;
distPenalty(seg_in_face>.2 & dists_to_mouth==0 & distPenalty_orig<2.5) = 0;
%
% distPenalty(seg_in_face>.2 & dists_to_mouth==0) = distPenalty(seg_in_face>.2 & dists_to_mouth==0)*.5;


% but in any case too many of the keypoints are covered, set the
% score to 0
totalCoverage = sum(dists <= faceScale*.1)/size(dists,1);
distPenalty(totalCoverage>=.4) = +2;

% forbiddenCoverage

seg_scores = seg_scores-coveragePenalty;%-angularPenalty;
seg_scores = seg_scores-distPenalty;
seg_scores = seg_scores-vicinityPenalty;
% seg_scores(seg_in_face<.1) = seg_scores(seg_in_face<.1)-penalty;

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
% seg_scores([occlusionPatterns.seg_in_face]>.4) = -2;
seg_scores(face_in_seg > .35) = -2;
% seg_scores(seg_in_face > .35) = -2;
% from_gpb = [occlusionPatterns.region_type]==1;
% seg_scores(~from_gpb) = seg_scores(~from_gpb)-.1;
% seg_scores(~from_gpb) = -100;

% dont want it to be too long / thin

% maj_min = [rprops.MajorAxisLength]./[rprops.MinorAxisLength];
% seg_scores(maj_min > 3) = seg_scores(maj_min > 3)-1;
% seg_scores(maj_min > 3) = 0;

% seg_scores(~d1) = -10;
% seg_scores(areaRatio < .1) = seg_scores(areaRatio < .1)-.5;
% [u,iu] = sort(seg_scores,'descend');

% coveragePenalty(iu)
% distPenalty(iu)