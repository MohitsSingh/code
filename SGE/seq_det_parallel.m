function seq_det_parallel(baseDir,d,indRange,outDir)

cd /home/amirro/code/mircs;
echo off;
if (~exist('toStart','var'))
    initpath;
    config;
    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
end
conf.demo_mode = false;
imageSet = imageData.test;
cur_t = imageSet.labels;

conf.get_full_image = true;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);

conf.featConf = init_features(conf);

%%
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if results for image %s exist...',filename);
    if (exist(resFileName,'file'))
        fprintf('results exist. skipping\n');
        continue;
    else
        fprintf('calculating...');
    end
    
    res = struct('parts',{},'regions',{},'scores',{});
    res(1).scores = -inf;
    
    imageInd = find(cellfun(@any,strfind(imageSet.imageIDs,currentID)));
    if (~any(imageInd))
        save(resFileName,'res');
        continue;
    end
    
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);
    
    conf.get_full_image = true;
    [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
    
    gpbFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','.mat'));
    ucmFile = strrep(gpbFile,'.mat','_ucm.mat');
    if (exist(ucmFile,'file'))
        load(ucmFile);
    end
    [regions,regionOvp,G] = getRegions(conf,currentID);
    [regionConfs] = applyModel(conf,currentID);%,partModels);
    %     dpmResPath = fullfile(conf.dpmDir,strrep(currentID,'.jpg','.mat'));
    %     load(dpmResPath);
    %     [regionConfs,modelResults] = normalizeScores(normalizers,regionConfs,modelResults);
    T_ovp = 1; % don't show overlapping regions...
    region_sel = suppresRegions(regionOvp, T_ovp); % help! we're being oppressed! :-)
    regions = regions(region_sel);
    regionOvp = regionOvp(region_sel,region_sel);
    origBoundaries = ucm<=.1;
    segLabels = bwlabel(origBoundaries);
    segLabels = imdilate(segLabels,[1 1 0]);
    segLabels = imdilate(segLabels,[1 1 0]');
    S = medfilt2(segLabels); % close the small holes...
    
    segLabels(segLabels==0) = S(segLabels==0);
    
    conf.get_full_image = true;
    
    
    %     segImage = paintSeg(I,segLabels);
    %     subplot(1,2,1); imagesc(I); axis image;
    %     subplot(1,2,2); imagesc(segImage); axis image;
    %     hold on;plotBoxes2([ymin xmin ymax xmax],'g');
    %     lipRectShifted = lipRectShifted + [xmin ymin xmin ymin];
    %     plotBoxes2(lipRectShifted([2 1 4 3]));
    %     [~,name,~] = fileparts(currentID);
    %     outDir = '/home/amirro/storage/res_s40';
    %     [~,filename,~] = fileparts(currentID);
    %     resFileName = fullfile(outDir,[filename '.mat']);
    %
    %     if (exist(resFileName,'file'))
    %         load(resFileName);
    %     end
    for iModel = 1:length(regionConfs)
        regionConfs(iModel).score = regionConfs(iModel).score(region_sel);
        nans = isnan(regionConfs(iModel).score);
        regionConfs(iModel).score(nans) = -inf;
    end
    
    %     selBox = faceBoxShifted;
    selBox = lipRectShifted;
    %     load(fullfile('~/storage/relativeFeats_s40',strrep(currentID,'.jpg','.mat')));
    %     relativeShapes = relativeShapes(:,region_sel);
    
    %     relativeShapes_ = region2EdgeSubset(G,relativeShapes,region_sel);
    
    
    G = G(region_sel,region_sel);
    
    faceBox = faceBoxShifted;
    
    [res.parts,res.allRegions,res.scores] = followSegments2(conf,regions,G,regionConfs,I,selBox,faceBox,regionOvp,[],[]);
    save(resFileName,'res');
    
end


end