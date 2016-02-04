function detect_hog_parallel(baseDir,imageNames,indRange,outDir)

cd('~/code/mircs');
initpath;
config;
addpath('~/code/utils');
addpath('/home/amirro/code/3rdparty/uri');
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
addpath('/home/amirro/code/3rdparty/exemplarsvm/features/');
init_hog_detectors;
refine_hog_detectors;
detection.min_scale = 0;
for k = 1:length(indRange)
    currentID = imageNames(indRange(k)).name;
    imagePath = fullfile(baseDir,imageNames(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping ... \n');
        continue;
    end
    
    workFileName = strrep(resFileName,'.mat','.wrk');
    if (exist(workFileName,'file'))
        fprintf('work file name already exists! skipping...');
        continue;
    end
    
    fid = fopen(workFileName,'w');
    fclose(fid);
    threshold = 0;
    I = im2double(imread(imagePath));
    
    
    % add overlap of each polygon with each region...
        
    regions = getRegions(conf,currentID,true);
    seg_shape = @(x) regionprops(x,'Eccentricity','Orientation','Area','MajorAxisLength','MinorAxisLength');
    shapes = cellfun2(seg_shape,regions);
    oris = cellfun(@(x) x.Orientation, shapes);
    % rotate the image according to this orientation...
    edges = -105:15:105;
    [n,bin] = histc(oris,edges);
    % Z_total = -inf(size2(I));
    
    
    regionRes = struct('regionID',{},'hogScore',{},'theta',{},'ovpScore','class');
    t = 0;
    for iBin = 2:length(edges)-1
        curTheta = edges(iBin)
        %         displayRegions(I,regions(bin==iBin),[],0);
        thetaRange = [90+curTheta,270+curTheta];
        curClassifier = classifiers(10);
        detection.max_scale = .3;
        detection.min_scale = 0;
        dets = detect_rotated2(I,classifiers,cell_size,features,detection,threshold,thetaRange);
        polys = {dets.polys};
        scores = [dets.scores];
        thetas = [dets.theta];
        classes = [dets.class];
        [s,is] = sort(scores,'descend');
        is = is(1:min(100,length(is)));
        scores = scores(is);
        polys = polys(is);
        thetas = thetas(is);
        %     H = computeHeatMap_poly(I,polys,scores,'max');
        polyRegions = cellfun2(@(x) poly2mask2(x,size2(I)),polys);
        regionInds = find(bin==iBin);
        curRegions = regions(bin==iBin);
        ovp = regionsOverlap(curRegions,polyRegions);
        % sum each region with the scores from it's most overlapping polygon
        [ovp,imax] = max(ovp,[],2); % find best polygon for each region
        
        %struct('regionID',{},'hogScore',{},'theta',{},'ovpScore','class');
        
        for iRegion = 1:length(regionInds)
            t = t+1;
            regionRes(t).regionID = regionInds(iRegion);
            regionRes(t).hogScore = scores(imax(iRegion));
            regionRes(t).theta = thetas(imax(iRegion));
            regionRes(t).ovpScore = ovp(iRegion);
            regionRes(t).class = classes(imax(iRegion));
        end
        
        %     Z = -inf(size2(I));
        %     for k = 1:length(curRegions)
        %         Z = max(Z,curRegions{k}*scores(imax(k))*ovp(k));
        %     end
        %     Z_total = max(Z,Z_total);
        %
        %     clf; subplot(1,3,1); imagesc(I); axis image;
        %     subplot(1,3,2); imagesc(Z); axis image;
        %     subplot(1,3,3); imagesc(Z_total); axis image;
        %     % sum score in segments...
        
%         pause;
    end
    
    thetaRange = 0:10:350;
    res = detect_rotated2(I,classifiers,cell_size,...
        features,detection,threshold,thetaRange);
    
    for kk = 1:length(res) % double->uint16
        res(kk).polys = uint16(res(kk).polys);
        res(kk).scores = single(res(kk).scores);
    end
    
    
    
    fprintf('done with image %s!\n',filename);
    save(resFileName,'res','regionRes');
    delete(workFileName);
end

fprintf('finished all images in current batch!\n\n\n');
end