function [allR] = drawBoxes(I,bbs,vals,mode)
%DRAWBOXES Summary of this function goes here
% mode == 1--> max, otherwise-->add
%   Detailed explanation goes here
% if (mode == 1)
%     R = zeros(ones(size(I,1),size(I,2));
% else
% 
% if (~isempty(vals))
%     warning('vals is ignored in drawBoxes, please put vals in bbs(:,12)');
% end
if (nargin < 4)
    mode = 1;
end
origSize = dsize(I,1:2);
% subplot(1,2,1); imshow(I);
if (size(bbs,2) < 13) % add angles
    bbs_ = ones(size(bbs,1),13);
    bbs_(:,1:4) = bbs;
    bbs_(:,13) = 0;
    bbs = bbs_;
%     bbs = [bbs,zeros(size(bbs,1),1)];
end
allR = {};
angles = unique(bbs(:,13));
imageDiag = mean(dsize(I,1:2));
minActualScale = 80/imageDiag;
for iAngle = 1:length(angles)
    curAngle = angles(iAngle);
    if (mode == 1)
        R = -inf(size(I,1),size(I,2));
    else
        R  =zeros(size(I,1),size(I,2));
    end
    
    II = imrotate(I,curAngle,'bilinear','loose');
    R = imrotate(R,curAngle,'bilinear','loose');
    if (mode==1)
        R = -inf(size(R));
    end
    curBBS = bbs(bbs(:,13)==curAngle,:);
    curBBS = curBBS(curBBS(:,8)>=minActualScale,:);
    % end
    vals = curBBS(:,12);
    %         if (isempty(vals))
    %             vals = ones(size(curBBS,1),1);
    %         end
    for k = 1:size(curBBS,1)
        b = round(curBBS(k,1:4));
        
        b(1) = max(b(1),1);
        b(2) = max(b(2),1);
        b(3) = min(b(3),size(R,2));
        b(4) = min(b(4),size(R,1));
        if (mode == 1)
            R(b(2):b(4),b(1):b(3)) = max(R(b(2):b(4),b(1):b(3)),vals(k));
        else
            R(b(2):b(4),b(1):b(3)) = R(b(2):b(4),b(1):b(3)) + vals(k);
        end
    end
    % crop out the original image out of R.
%     clf;
%     subplot(2,2,3);imshow(II);
%     subplot(2,2,4);imshow(R,[]);
    
    sizeDiff = size(R)-origSize;
    R = imrotate(R,-curAngle,'bilinear','crop');
    %         II = imrotate(II,-curAngle,'bilinear','crop');
    topLeft = floor(sizeDiff/2)+1;
    bottomRight = topLeft+origSize-1;
    R = R(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2));
    II = II(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2),:);
    %         figure(3);imshow(R);
    allR{iAngle} = R;
    
%             subplot(2,2,1); imshow(imrotate(I,0,'bilinear','loose'));
%             subplot(2,2,2);imshow(R,[]);title(num2str(curAngle));
%     
%     pause;
    if (iAngle==3)
        blurg = 1;
    end
end

% R(R==-1000) = min(R(R>-1000));

%         figure,imshow(R,[]);
end

