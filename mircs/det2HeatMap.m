function Z = det2HeatMap(im,dets)
Z = {}
for k = 1:length(dets)
    M = -inf(dsize(im,1:2));
    bc = boxCenters(dets(k).cluster_locs(:,1:4));
    vals =dets(k).cluster_locs(:,12);
    scales = dets(k).cluster_locs(:,8);
    inds = sub2ind2(size(M),fliplr(round(bc)));
    for z = 1:length(inds)
        M(inds(z)) = max(M(inds(z)),vals(z));
    end
    M(isinf(M)) = min(M(~isinf(M)));
    Z{k} = imfilter(M,fspecial('gauss',36,3),'replicate');
end
end