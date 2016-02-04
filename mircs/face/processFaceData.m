function faceLandmarks = processFaceData(conf,ids,faceLandmarks)

% faceLandmarks = struct('s',{},'c',{},'xy',{},'level',{},'lipBox',{},'faceBox',{})
% faceLandmarks(1).s = [];
% faceLandmarks(1).valid = true;
% faceLandmarks = repmat(faceLandmarks,length(landmarks),1);
count_ = 0;
nFaceNotDetected = 0;
for q = 1:length(faceLandmarks)
    bs = faceLandmarks(q).bs;
    
    % choose box with maximal area inside person box, this is probably the
    % correct face. 
    
    if (isempty(bs))
        nFaceNotDetected = nFaceNotDetected + 1;
        continue;
    end
    
    boxes = zeros(length(bs),4);
    for k = 1:length(bs)
        boxes(k,:) = pts2Box(bs(k).xy);
    end
    
%     conf.get_full_image = true;
%     clf,imshow(getImage(conf,ids{q})); hold on;
%     plotBoxes2(faceLandmarks(q).personBox(:,[2 1 4 3]),'g','LineWidth',2);
%     plotBoxes2(boxes(:,[2 1 4 3]),'m','LineWidth',1);
%     drawnow
    
    bInt = BoxIntersection(boxes,faceLandmarks(q).personBox);
%     plotBoxes2(boxes(:,[2 1 4 3]),'r--','LineWidth',1);
    [~,~,a] = BoxSize(bInt);
    [~,~,b] = BoxSize(boxes);
    [m,im] = max(a./b);
%     q
    if (m < .8)        
            conf.get_full_image = true;
    clf,imshow(getImage(conf,ids{q})); hold on;
    plotBoxes2(faceLandmarks(q).personBox(:,[2 1 4 3]),'g','LineWidth',2);
    plotBoxes2(boxes(:,[2 1 4 3]),'m','LineWidth',1);
    conf.get_full_image = false;
    
    
    
%     detect_landmarks(conf,{getImage(conf,ids{q})},4,true);
    conf.get_full_image = true;
pause

        continue;
    end
%     
    bs = bs(im);

    if (size(bs.xy,1) == 68)
        lipCoords = boxCenters(bs.xy(33:51,:));
        lipBox = pts2Box(lipCoords);
    elseif (size(bs.xy,1) == 39)
% % %         conf.get_full_image = true;
% % %         clf,imshow(getImage(conf,ids{q}));
% % %         hold on;
% % %         plot(bs.xy(:,1),bs.xy(:,2),'g.');
% % %         
% % %         for kk = 1:size(bs.xy,1)
% % %             text(bs.xy(kk,1),bs.xy(kk,2),num2str(kk),'color','y');
% % %         end
% % %           
% % %         
% % %         pause;
%         lipCoords = [];
%         lipBox = [];
%         faceLandmarks(q).valid = false;
        
        lipCoords = boxCenters(bs.xy([20:22 28 29],:));
        lipBox = pts2Box(lipCoords);
        
    else
        error('!!!')
    end
    count_ = count_+1;
    faceBox = pts2Box(boxCenters(bs(1).xy));
%     faceLandmarks(q) = bs(1);

    faceBox = faceBox-repmat(faceLandmarks(q).personBox(1:2),1,2);
    lipBox = lipBox-faceLandmarks(q).personBox;

    faceLandmarks(q).lipBox = lipBox;
    faceLandmarks(q).faceBox = faceBox;
    faceLandmarks(q).s = bs.s;
        
end

nFaceNotDetected
% for images where the lips were not detected,
% take the average box.
count_
allBoxes_missing = cat(1,faceLandmarks.lipBox);
faceBoxes_missing = cat(1,faceLandmarks.faceBox);
meanBox = mean(allBoxes_missing,1);
meanFaceBox = mean(faceBoxes_missing,1);
minScore = min([faceLandmarks.s]);
for k = 1:length(faceLandmarks)
    if (isempty(faceLandmarks(k).lipBox))
%         k
        faceLandmarks(k).lipBox = meanBox;
        faceLandmarks(k).faceBox = meanFaceBox;
    end
    
    if (isempty(faceLandmarks(k).s))
        faceLandmarks(k).s = -1000*minScore;
        
    end
end

allBoxes_complete = cat(1,faceLandmarks.lipBox);
faceBoxes_complete = cat(1,faceLandmarks.faceBox);