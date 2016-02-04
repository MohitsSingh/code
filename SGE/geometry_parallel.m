function geometry_parallel(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;
config;
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    
    imagePath = fullfile(baseDir,currentID);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping ... \n');
        continue;
    end
    currentID = d(indRange(k)).name;
    
    regionsPath = fullfile('~/storage/gpb_s40',[filename '_regions.mat']);
    [regions] = getRegions(conf,currentID,true);
    propertyList = {'MajorAxisLength','MinorAxisLength',...
        'Solidity','BoundingBox','Orientation','Eccentricity','Area','Centroid'};
    props = cellfun(@(x) regionprops(x,propertyList),regions,'UniformOutput',false);
    props = cat(1,props{:});
    save(resFileName,'props');
    
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

