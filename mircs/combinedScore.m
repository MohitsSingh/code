function s = combinedScore(conf,im,curImageData,model)
%1. remove all responses which are not in occlusion regions:

occludingRegions = getOccludingCandidates(im,curImageData);
s = zeros(size2(im));
displayRegions(im,occludingRegions);return;

return;
if (isempty(occludingRegions))
    s = zeros(size2(im)); return;
end
regionBoxes = cellfun2(@(x) pts2Box(fliplr(ind2sub2(dsize(im,1:2),find(x)))), occludingRegions);
regionBoxes = cat(1,regionBoxes{:});
responses = curImageData.templateResponses;
responses = responses(nms(responses,.9),:);
[ overlaps ] = boxesOverlap( regionBoxes,responses);
responses(:,end) = responses(:,end)+1.1;
[ovp,iovp] = max(overlaps,[],1);
% ovp(ovp<.5) = 0;
%responses(:,end) = responses(:,end).*ovp(:);
responses(ovp == 0,:) = [];
s = computeHeatMap(im,responses,'max');




