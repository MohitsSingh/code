function bowFeats_parallel( baseDir,d,indRange,outDir,tofix )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
cd ~/code/mircs;
initpath;
config;
conf.featConf = init_features(conf,1024);
% learn the part models...
extractors = initializeFeatureExtractors(conf);
extractors{1}.doPostProcess = false;
for k = 1:length(indRange)
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    %     for iFeatType = 1:length(featConf)
    %         suffix = featConf(iFeatType).suffix;
    resFileName = fullfile(outDir,[filename '.mat']);
    %     fprintf('\nchecking if filename %s exists :.... ',resFileName);
    if (exist(resFileName,'file'))
%         fprintf('YES:.... \n');
        continue;
        %
    end
    fprintf('calculating for%s :....\n ',resFileName);
    feats = extractors{1}.extractFeatures(d(indRange(k)).name);
    save(resFileName,'feats');
    
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

