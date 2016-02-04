%%%% Experiment 24 %%%%
%% Jan 15/2014
%% apply a detection on each region of the GPB, rotated according to it's major axis,\
%% where the major axis may assume several options.
if (0)
    initpath;
    config;
    init_hog_detectors;
    refine_hog_detectors;
    [test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
    [train_ids,train_labels]= getImageSet(conf,'train');
    f = find(test_labels);
    conf.get_full_image = true;
    currentID = train_ids{1};
    %currentID = 'drinking_027.jpg';
    I = getImage(conf,currentID);
    regions = getRegions(conf,currentID,true);
    seg_shape = @(x) regionprops(x,'Eccentricity','Orientation','Area','MajorAxisLength','MinorAxisLength');
    shapes = cellfun2(seg_shape,regions);
end
oris = cellfun(@(x) x.Orientation, shapes);

plot(oris)

% rotate the image according to this orientation...
edges = -105:15:105;
[n,bin] = histc(oris,edges);
Z_total = -inf(size2(I));
for iBin = 2:length(edges)-1
    curTheta = edges(iBin)
    %         displayRegions(I,regions(bin==iBin),[],0);
    dTheta = 10;
    %thetaRange = 90+[curTheta-dTheta : dTheta : curTheta+dTheta];
    thetaRange = [90+curTheta,270+curTheta];
    curClassifier = classifiers(10);
    %         all_bbs = detect_rotated(I,curClassifier,cell_size,features,detection,threshold,thetaRange,true);
    detection.max_scale = .2;
    detection.min_scale = 0;
    dets = detect_rotated2(I,classifiers,cell_size,features,detection,threshold,thetaRange);
    polys = {dets.polys};
    scores = [dets.scores];
    [s,is] = sort(scores,'descend');
    is = is(1:min(100,length(is)));
    scores = scores(is);
    polys = polys(is);
    
    H = computeHeatMap_poly(I,polys,scores,'max');
    polyRegions = cellfun2(@(x) poly2mask2(x,size2(I)),polys);
    curRegions = regions(bin==iBin);
    ovp = regionsOverlap(curRegions,polyRegions);
    
    % sum each region with the scores from it's most overlapping polygon
    [ovp,imax] = max(ovp,[],2);
    
    Z = -inf(size2(I));
    for k = 1:length(curRegions)
        Z = max(Z,curRegions{k}*scores(imax(k))*ovp(k));
    end
    Z_total = max(Z,Z_total);
    
    clf; subplot(1,3,1); imagesc(I); axis image;
    subplot(1,3,2); imagesc(Z); axis image;
    subplot(1,3,3); imagesc(Z_total); axis image;
    % sum score in segments...
    
    pause;
end


