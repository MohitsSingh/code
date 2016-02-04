function allProperties = getProperties(imageList)
allProperties = cell(size(imageList));
% for iImage =96
parfor iImage = 1:length(imageList)
    iImage
    im = imageList{iImage};
    [Iseg labels map gaps E] = vl_quickseg(im, .5,2, 10);
    faceSeg =  segmentFace_stochastic(im,1);
    rprops = regionprops(labels,faceSeg,'Eccentricity','PixelList','Orientation','PixelIdxList','Area','Solidity','BoundingBox',...
        'MinorAxisLength','MaxIntensity','Image','MajorAxisLength');
    e1 = [rprops.Eccentricity];
    e2 = [rprops.Orientation];
    e3 = zeros(size(rprops));
    for k = 1:length(e3)
        e3(k) = max(rprops(k).PixelList(:,2));
    end
    e4 = [rprops.Area];
    e5 = [rprops.Solidity];
    e6 = [rprops.MinorAxisLength];
    e7 = zeros(1,length(rprops));
    for k = 1:length(e7)
        xy = rprops(k).PixelList;
        [ymin,iminy] = min(xy(:,2));
        xmin = xy(iminy,1);
        e7(k) = xmin >= 22 && xmin <= 50 && ymin >= 40 && ymin <=70;
    end
    e8 = inf(size(e7));
    for iCandidate = 1:length(e8)
        [x,y] = find(labels==iCandidate);
        if (length(x) >= 3)
            [c,gof] = fit(y,x,'poly1');
            e8(iCandidate) = gof.rmse;
        end
    end
    
    f_small = imresize(faceSeg,1,'bilinear');
    [gx,gy] = gradient(f_small);
    e9 = zeros(size(e8));
    for iCandidate = 1:length(e9);
        gx_ = gx(rprops((iCandidate)).PixelIdxList);
        gy_ = gy(rprops((iCandidate)).PixelIdxList);
        ori = rprops((iCandidate)).Orientation;
        e9((iCandidate)) = sum(abs([gx_ gy_]*[cos(ori) sin(ori)]'));
    end
    curProps = [e1(:) e2(:) e3(:) e4(:) e5(:) e6(:) e7(:) e8(:) e9(:)];
    allProperties{iImage} = curProps;
end