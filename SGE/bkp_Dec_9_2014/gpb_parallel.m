function gpb_parallel(baseDir,d,indRange,outDir)

handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));
addpath('/home/amirro/code/utils');
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
addpath('/home/amirro/code/mircs/utils/');
doFix = 1;
for k = 1:length(indRange)
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [pathstr,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    curPath_ucm =  strrep(resFileName,'.mat','_ucm.mat');
    curPath_regions = strrep(resFileName,'.mat','_regions.mat');
    
    fprintf('*****>>>>>>>>>>>>>CURRENT FILE: %s\n:', resFileName);
    %     fprintf('calculating gpb...');
    
    needGPB = false;
    needUCM = false;
    needRegions = false;
    if (~doFix)
        if (exist(curPath_regions,'file'))
            fprintf('regions exist! \n');
            clear G;
            load(curPath_regions);
            if ~exist('G','var')
                fprintf('calculating adjacency graph...\n');
                G = regionAdjacencyGraph(regions);
                save(curPath_regions,'regions','regionOvp','G');
            end
            continue;
        else
            needRegions = true;
        end
    else
        needGPB = true;
    end
    if (exist(curPath_ucm,'file'))
        fprintf('ucm exists!\n');
    else
        needUCM = true;
    end
    
    if(exist(resFileName,'file'))
        fprintf('gpb exists!\n');
    else
        needGPB = true;
    end
    
    if (needGPB)
        fprintf('calculating gPb....');
        [gPb_orient, gPb_thin, textons] = globalPb(imread(imagePath));
        gPb_orient = single(gPb_orient);
        gPb_thin = single(gPb_thin);
        textons = single(textons);
        fprintf('done! Saving result to file...\n');
        %save(resFileName,'gPb_orient', 'gPb_thin', 'textons');
        save(resFileName,'gPb_thin');
        if (doFix)
            continue;
        end
    end
    
    if (needUCM)
        fprintf('calculating ucm....');
        load(resFileName);
        ucm = contours2ucm(double(gPb_orient));
        save(curPath_ucm,'ucm');
    end
    
    if (needRegions)
        fprintf('calculating regions....');
        load(curPath_ucm);
        regions  = combine_regions_new(ucm,.1);
        regionOvp = regionsOverlap(regions);
        G = regionAdjacencyGraph(regions);
        save(curPath_regions,'regions','regionOvp','G');
    end
    
    %     delete(resFileName);
    
    fprintf('done with : %s\n:', resFileName);
end
fprintf('\n\n ***** finished all files in batch ****\n\n\n\n');
end