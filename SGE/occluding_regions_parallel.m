function occluding_regions_parallel(baseDir,d,indRange,outDir)


cd ~/code/mircs;
initpath;
config;
load ~/storage/misc/imageData_new;
% now we have weight vectors for all of the above...
addpath('/home/amirro/code/3rdparty/PolygonClipper');
addpath('/home/amirro/code/3rdparty/sliding_segments');
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
addpath(genpath('~/code/3rdparty/geom2d'));
params.inflationFactor = 1.5;
params.regionExpansion = 1;
params.ucmThresh = .1;
params.fullRegions = true;
conf.get_full_image = true;

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
    
     fprintf('calculating for: %s!\n',filename);
    
    imageIndex = findImageIndex(newImageData,currentID);
    curImageData = newImageData(imageIndex);
       
    occlusionPattern = getOcclusionData(conf,curImageData);
    curImageData.occlusionPattern = occlusionPattern;
    [im,I_rect] = getImage(conf,curImageData.imageID);
    %if (curImageData.faceScore == -1000)
    if (curImageData.faceScore <-.9) %TODO...
        occlusionPattern = [];
        occludingRegions = [];
        occlusionPatterns = []; 
        rprops = [];
    else
        [xy_c,mouth_poly,face_poly] = getShiftedLandmarks(curImageData.faceLandmarks,-I_rect);
        curImageData.mouth_poly = mouth_poly;
        curImageData.face_poly = face_poly;
        [occludingRegions,occlusionPatterns,rprops] = getOccludingCandidates(im,curImageData);
    end
        
    save(resFileName,'occlusionPattern','occludingRegions','occlusionPatterns','rprops');
    %     [curRegions,groups,ref_box,face_mask,mouth_mask,I_sub,~,region_sel] = extractRegions2(conf,imgData,params); %#ok<ASGLU>
    %     %     save(resFileName,'curRegions','groups','ref_box','face_mask','mouth_mask','I_sub','params','region_sel');
    %
    %     if (~isempty(curRegions))
    %         curRegions = curRegions(region_sel);
    %         [~,curRegions] = expandRegions(curRegions,[],groups);
    %         N = numel(curRegions{1});
    %         areas = cellfun(@nnz,curRegions);
    %         curRegions((areas/N) > .5) = [];
    %         if (isempty(curRegions))
    %             continue;
    %         end
    %         curRegions = fillRegionGaps(curRegions);
    %         curRegions = col(removeDuplicateRegions(curRegions));
    %
    %         [curRegionFeats] = extractRegionFeatures(conf,curRegions,imgData,false,-.6,true);
    %     else
    %         curRegionFeats = [];
    %     end
    %
    %     save(resFileName,'curRegions','groups','ref_box','face_mask','mouth_mask','I_sub','params','region_sel','curRegionFeats');
    %
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

