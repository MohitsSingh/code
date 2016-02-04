%%% check my occlusion handling on COFW


initpath; config;
dataDir = '/home/amirro/code/3rdparty/rcpr_v1/data';
train_data_path = fullfile(dataDir,'COFW_train');
train_data = load(train_data_path);

for k = 1:length(train_data)
    k = 201;
    curIm = train_data.IsTr{201};
    
    curBB = train_data.bboxesTr(k,:);curBB(3:4) = curBB(3:4)+curBB(1:2);
    curBB = inflatebbox(curBB,1.3,'both',false);
    
    clf; imagesc2(train_data.IsTr{201});colormap gray;
    plotBoxes(curBB);
    
    I = cropper(curIm,curBB);
    [gPb_orient, gPb_thin, textons] = globalPb(I);
    gPb_orient = single(gPb_orient);
    gPb_thin = single(gPb_thin);
    textons = single(textons);
    ucm = contours2ucm(double(gPb_orient));
    regions  = combine_regions_new(ucm,.1);
    regionOvp = regionsOverlap(regions);
    G = regionAdjacencyGraph(regions);
    
    seg_data.gPb_orient = gPb_orient;
    seg_data.gPb_thin = gPb_thin;
    seg_data.textons = textons;
    seg_data.ucm = ucm;
    seg_data.regions = regions;
    seg_data.regionOvp = regionOvp;
    seg_data.G = G;
    
    displayRegions(I,regions);
    I1 = im2double(cat(3,I,I,I));
    landmarks = extractLandmarks(conf,I1);
    landmarks = landmarks([landmarks.isvalid]);
    [faceLandmarks] = landmarks2struct_3(landmarks);n = 0;
    for t = 1:length(faceLandmarks)
        n = n+1;
        imageData.faceLandmarks(n) = faceLandmarks(t);
    end
    for k = 1:length(faces)
        
        imageData.segData = seg_data;imageData.imageID = I1;
    [occlusionPatterns,regions,face_mask,mouth_mask] = getOcclusionPattern_2(conf,imageData,faceLandmarks(k));
%     occlusionPattern.mouth_poly = mouth_poly;
%     occlusionPattern.face_poly = face_poly;
    occlusionPattern.occlusionPatterns = occlusionPatterns;
    occlusionPattern.regions = regions;
    occlusionPattern.face_mask = face_mask;
    occlusionPattern.mouth_mask = mouth_mask;
    occlusionPattern.faceBox = faceLandmarks(k).faceBox;
    clear occlusionData;
    [occlusionData.occludingRegions,occlusionData.occlusionPatterns,occlusionData.rprops] = getOccludingCandidates_2(I,occlusionPattern);
    imageData.occlusionData(k) = occlusionData;
end
end


