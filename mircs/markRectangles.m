function rects = markRectangles(I,roi_box,cls,reqAnno,numToSelect)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
rects = {};
for t = 1:numToSelect
    clf;
    imagesc2(I);plotBoxes(roi_box,'r--','LineWidth',3);
    bb_inflated = inflatebbox(roi_box,3,'both',false);
    bb_inflated = BoxIntersection(bb_inflated,[1 1 fliplr(size2(I))]);
    zoomToBox(bb_inflated);
    title(sprintf('please mark rectangle for %s of class %s',reqAnno,cls));    
    bb = getrect();
    bb(3:4) = bb(3:4)+bb(1:2);
%     bb = getSingleRect(true);
    rects{t} = bb;
    if t==numToSelect,break,end
    title('hit left button to continue, other to finish');
    plotBoxes(cat(1,rects{:}));
    plotBoxes(roi_box,'r--','LineWidth',3);
    [x,y,b] = ginput(1);
    if b~=1
        break
    end
end
end

