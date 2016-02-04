function allres = gpb_parallel_new(imagePaths,inds,jobID,outDir,extraInfo,job_suffix)
cd ~/code/mircs/;
conf = extraInfo.conf;
path(extraInfo.path);
newImageData = extraInfo.newImageData;

if (~isfield(extraInfo,'runMode'))
    runMode = 'full';
else
    runMode = extraInfo.runMode;
end

if (~isfield(extraInfo,'minFaceScore'))
    minFaceScore = -.6;
else
    minFaceScore = extraInfo.minFaceScore;
end
allres = {};
conf.get_full_image = true;
handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));
addpath('/home/amirro/code/utils');
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
addpath('/home/amirro/code/mircs/utils/');
doFix = false;
for k = 1:length(inds)
    k
    curImageData = newImageData(inds(k));
    if (curImageData.faceScore >= minFaceScore)
        resPath = fullfile(outDir,strrep(curImageData.imageID,'.jpg',['_' job_suffix '.mat']));
        %imagePath = fullfile(baseDir,d(indRange(k)).name);        
        [pathstr,filename,~] = fileparts(curImageData.imageID);
        resFileName = fullfile(outDir,[filename '.mat']);
        curPath_ucm =  strrep(resFileName,'.mat','_ucm.mat');
        curPath_regions = strrep(resFileName,'.mat','_regions.mat');
        fprintf('*****>>>>>>>>>>>>>CURRENT FILE: %s\n:', resFileName);
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
%                 continue;
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
        
        
        
        I = getImage(conf,curImageData.imageID);
        if (strcmp(extraInfo.runMode,'sub'))
           [I,~,face_box,face_poly] = getSubImage(conf,curImageData,2,false);
%            I = imresize(I,[48 NaN],'bilinear');
        end
        
        if (~isempty(extraInfo.absScale))
            I = imresize(I,[2*extraInfo.absScale NaN],'bilinear');
        end
        
        if (needGPB)
            fprintf('calculating gPb....');
            [gPb_orient, gPb_thin, textons] = globalPb(I);
            gPb_orient = single(gPb_orient);
            gPb_thin = single(gPb_thin);
            textons = single(textons);
            fprintf('done! Saving result to file...\n');
            %save(resFileName,'gPb_orient', 'gPb_thin', 'textons');
            save(resFileName,'gPb_thin','gPb_orient');
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
%         save(resPath,'res');
    end    
    
end
%     end
allres{k} = [];
fprintf('done with image %s!\n',curImageData.imageID);

fprintf('***********************************\n************FINISHED***************\n***********************************\n');

res = allres;
resFileName = fullfile(outDir,sprintf('job_%05.0f_%s_agg.mat',jobID,job_suffix));
save(resFileName,'res','inds','jobID');