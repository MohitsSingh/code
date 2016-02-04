function [newDets,dets,allScores] = combineDetections(dets)

% rearrange the detections by image order.
nImages = size(dets(1).cluster_locs(:,1),1);
allScores = zeros(nImages,length(dets));
for k = 1:length(dets)
    dets(k) =arrangeDet(dets(k),'index');
    allScores(:,k) = dets(k).cluster_locs(:,12);
end

% allScores = allScores.*(allScores>0);

[~,iChoice] = max(allScores,[],2);
newDets = dets(1);
newDets.cluster_locs = zeros(nImages,13);
for k = 1:nImages
    newDets.cluster_locs(k,:) = dets(iChoice(k)).cluster_locs(k,:);
end
newDets = arrangeDet(newDets,'index');
% [~,is] = sort(newDets.cluster_locs(:,12),'descend');
% newDets.cluster_locs = newDets.cluster_locs(is,:);
end