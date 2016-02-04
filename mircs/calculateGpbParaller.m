function calculateGpbParaller(baseDir,d,indRange)

handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));


for k = 1:length(indRange)
    filename = fullfile(baseDir,d(indRange(k)).name);
    resFileName = strrep(filename,'.tif','.mat');
    fprintf('checking if filename %s exists :.... \n',resFileName);
    
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping! \n');
        continue;
    end
    
    fprintf('calculating gPb....');
    [gPb_orient, gPb_thin, textons] = globalPb(imresize(imread(filename),2,'bilinear'));
    fprintf('done! Saving result to file...\n');
    
    save(resFileName,'gPb_orient', 'gPb_thin', 'textons');
    fprintf('finised!\n');
end
fprintf('\n\n\nFINISHED\n\n\n!\n');
end