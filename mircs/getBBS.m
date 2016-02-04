function bbs = getBBS(imageIDs,boxes,frameBoxes,subScale)
for k = 1:length(imageIDs)
    
    bbox = frameBoxes(k,1:4);
    subBox = boxes(k,1:4)-bbox([1 2 1 2]);    
    if (~isempty(subScale))  % normalize the sub-box relative to the given frame.
        bc = boxCenters(subBox);
        dd = (bbox(3)-bbox(1));
        subBox = inflatebbox([bc bc],[subScale subScale]*dd,'both',true);
        subBox = subBox + bbox([1 2 1 2]);
    end    
    bbs(k,:) = subBox;
    
    
%     currentID = imageIDs{k};
%     [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
%     clf;
%     imagesc(I);axis image;
%     hold on;
%     plotBoxes2(bbs(k,[2 1 4 3]),'g');
%     pause;
end