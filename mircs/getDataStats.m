function dataStats = getDataStats(conf,fra_db,test_subset,testData)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[r,ir] = sort(res.curScores,'descend');
test_inds = testData.inds;
test_regions = testData.regions;
labels = testData.labels;
unique_inds = unique(test_inds);

classCount = zeros([max([fra_db.classID])],1);
posCount = zeros(size(classCount));
%%
for p = 13:length(unique_inds)
%     p
    curInd = unique_inds(p);    
    g = test_inds==curInd;    
    imgData = fra_db(curInd);
%     
    curRegions = test_regions(g);
    curOvps= testData.ovps(g);
    curLabels = 2*(labels(g)==imgData.classID)-1;
    curClassID = imgData.classID;
    classCount(curClassID) =classCount(curClassID)+1;
    if (any(curLabels==1))
        posCount(curClassID) = posCount(curClassID)+1;
    else
        fprintf('no positive candidates. image: %d,%s , class %d\n',curInd,imgData.imageID,curClassID);
        
        break;
        [I_sub,faceBox,mouthBox,face_poly,I] = getSubImage2(conf,imgData,false);
        displayRegions(I_sub,curRegions,curOvps)
         %thisTestData = collectSamples2(conf, imgData,1,params);
        clf; imagesc2(I_sub);
        plotPolygons(face_poly);
    end
end
   %% 
image_seen = false(size(fra_db));
for it = 1:length(r)
    t = ir(it);
    curInd = test_inds(t);
    if image_seen(curInd),continue,end
    image_seen(curInd) = true;
    imgData = fra_db(curInd);
    [I_sub,faceBox,mouthBox,I] = getSubImage2(conf,imgData,~params.testMode);
    %clf;imagesc2(I); plot_dlib_landmarks(imgData.Landmarks_dlib);
    clf;imagesc2(I);
    plotPolygons(imgData.landmarks.xy,'g.','LineWidth',2);
    %     dpc;continue
    f = test_inds==curInd;
    curRegions = test_regions(f);
    [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,imgData.isTrain);
    %     [mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
    heatMap = computeHeatMap_regions(I_sub,curRegions,res.curScores(f),'max');
    %     heatMap(heatMap<0) = min(heatMap(heatMap(:)>0));
    clf;
    curLandmarks = bsxfun(@minus,curLandmarks,mouthBox(1:2));
    subplot(1,3,1); imagesc2(I_sub);% plot_dlib_landmarks(curLandmarks);
    plotPolygons(curLandmarks,'go','MarkerSize',3,'LineWidth',3);
    subplot(1,3,2);
    imagesc2(sc(cat(3,heatMap,I_sub),'prob_jet'));
    % dpc;continue;
    subplot(1,3,3);
    curScores = res.curScores(f);
    displayRegions(I_sub,curRegions,curScores,'maxRegions',3);
end

end

