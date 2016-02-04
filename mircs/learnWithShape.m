function objModel = learnWithShape(conf,groundTruth,objName)
objName = 'cup';
gtParts = {groundTruth.name};
isObj = cellfun(@any,strfind(gtParts,objName));
groundTruth = groundTruth(isObj);
sourceImages = {groundTruth.sourceImage};
imgDir = fullfile('~/storage/Drinking_Images/Test',['Test' objName]);
ensuredir(imgDir);
conf.get_full_image = true;
curDir = pwd;
kasSource = '~/code/3rdparty/kas_sources_V1.04/';
for k = 1:length(groundTruth)
    imgPath = fullfile(imgDir,groundTruth(k).sourceImage);
    
    % write the ground truth....
%     groundTruth(k).
    
    if (~exist(imgPath,'file'))
        I = getImage(conf,groundTruth(k).sourceImage);
        imwrite(I,imgPath);
    end
    
    gtPath = strrep(imgPath,'.jpg',['_' objName '.groundtruth']);
        
    if (~exist(gtPath,'file'))        
        sameImage = find(cellfun(@any,strfind(sourceImages,groundTruth(k).sourceImage)));
        gtBB = zeros(length(sameImage),4);
        for iImg = 1:length(sameImage)
            gtBB(iImg,:) = round(pts2Box([groundTruth(sameImage(iImg)).polygon.x,groundTruth(sameImage(iImg)).polygon.y]));
        end
        
        save(gtPath,'gtBB');
    end
end

cd(kasSource);
addpath(genpath(pwd));
setenv('BERKELEY_UNITEX_PATH',pwd);
detect_kas_exec_interface(imgDir,'jpg',2);


 % compute the contour segment network...

disp('done writing images');


end