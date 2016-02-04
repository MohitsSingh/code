function [lipImages,faceScores] = getLipImages(conf,m,landmarks,sz,inflateFactor,ttt)
[faceLandmarks,allBoxes] = landmarks2struct(landmarks);
faceScores = [faceLandmarks.s];
% 
% for k = 1:length(m)
%     k
% %     if (ttt(k))
% clf
%     imshow(m{k});
%     hold on;
%     plotBoxes2(allBoxes(k,[2 1 4 3])/2,'g','LineWidth',2);
%     pause;
% %     end
% end

% get the avg. box size..
bc = boxCenters(allBoxes);

meanWidth = mean(allBoxes(:,3)-allBoxes(:,1));
meanHeight = mean(allBoxes(:,4)-allBoxes(:,2));
allBoxes2 = [bc-repmat([meanWidth meanHeight],size(bc,1),1)/2,...
    bc+repmat([meanWidth meanHeight],size(bc,1),1)/2];

% allBoxes2(:,3)-allBoxes2(:,1)
lipImages = multiCrop(conf,m,round(inflatebbox(allBoxes2/ttt,inflateFactor)),sz);
end