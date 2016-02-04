function kps = getKPCoordinates_zhu(lmData,ress)
kps = NaN(length(lmData),68,2);
for t = 1:length(lmData)
    polys = lmData(t).polys;
    curPts = cellfun2(@(x) mean(x,1),polys);
    curPts = cat(1,curPts{:});
    curPts =  bsxfun(@minus,curPts,ress(t,1:2));
    kps(t,1:size(curPts,1),:) = curPts;
end
end
