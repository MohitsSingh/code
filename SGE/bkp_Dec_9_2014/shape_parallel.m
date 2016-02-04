function shape_parallel(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;
config;

for k = 1:length(indRange)
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        disp('already exists! skipping ... \n');
        continue;
    end
    currentID = d(indRange(k)).name;
    regions = getRegions(conf,currentID);
    conf.get_full_image = true;
    [shapeFeats] = getShapeFeatures(conf,getImage(conf,currentID),regions);
    
    save(resFileName,'shapeFeats');
    
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

