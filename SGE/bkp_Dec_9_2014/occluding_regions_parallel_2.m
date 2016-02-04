function occluding_regions_parallel_2(baseDir,d,indRange,outDir)


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
    I = imread(imagePath);
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
    if (curImageData.faceScore < -.6)
        continue;
    end
    f_1 = j2m('~/storage/gpb_s40_face_2',curImageData);
    L1 = load(f_1);
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1.5,true);
    face_mask = poly2mask2(face_poly,size2(I));
    faceScale = face_box(4)-face_box(2);
    
    face_mask = poly2mask2(face_poly,size2(I));
    faceScale = face_box(4)-face_box(2);                    
    face_mask = refineOutline_3(I,face_mask,ceil(faceScale/20)); %TODO -
    
    regions2 = L1.res.regions;
    perims = cellfun2(@(x) nnz(bwperim(x)),regions2);
    U = addBorder(false(size2(regions2{1})),1,1);
    % kill regions with too much border
    border_lengths = cellfun2(@(x) nnz(x & U),regions2);
    perims = cat(2,perims{:}); border_lengths = cat(2,border_lengths{:});
    regions2(border_lengths./perims > .2) = [];
    regions2 = cellfun2(@(x) imResample(single(x),size2(M),'nearest'),regions2);
    regions2 = regions2(cellfun(@(x) nnz(x)>0,regions2));
    regions2 = removeDuplicateRegions(regions2);
    regions2 = shiftRegions(regions2,round(face_box),I);
    % add more regions....
    [regions,regionOvp,G] = getRegions(conf,curImageData.imageID);
    regions2 = [regions2,regions];    
    roi = round(inflatebbox(face_box,[1.5 1.5],'both',false));
    roi = clip_to_image(roi,I);
    curImageData.face_mask = face_mask;
    occlusionPattern = getOcclusionData(conf,curImageData,roi,regions2);
    curImageData.occlusionPattern = occlusionPattern;
    curImageData.mouth_poly = mouth_poly;
    curImageData.face_poly = face_poly;
    [occludingRegions,occlusionPatterns,region_scores] = getOccludingCandidates_3(conf,I,curImageData);        
    save(resFileName,'occludingRegions','occlusionPatterns','region_scores');   
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

