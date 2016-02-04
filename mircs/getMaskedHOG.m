function feats = getMaskedHOG(conf,imgs,rr)
feats = {};
for k = 1:length(imgs)
    k
    curImg = im2single(imgs{k});
    for kk = 1:length(rr{k})
        mask = rr{k}{kk};
        [ii,jj] = find(mask);
        xmin = min(jj);
        xmax = max(jj);
        ymin = min(ii);
        ymax = max(ii);
        I = imresize(curImg(ymin:ymax,xmin:xmax,:),[48 48]);
        X = vl_hog(I,conf.features.vlfeat.cellsize,'NumOrientations',9);
        feats{k,kk} = X(:);
        %imageSetFeatures2 
    end
    
end

end
