function pts = boxesToEdges(boxes,edgeMap)
    T = .1;
    pts = zeros(size(boxes,1),2);
    for k = 1:size(boxes,1)
        curBox = round(clip_to_image(boxes(k,:),edgeMap));
        b = edgeMap(curBox(2):curBox(4),curBox(1):curBox(3));
        [m,im] = max(b(:));
        if (m < T)
            pts(k,:) = boxCenters(curBox);
        else
            [y,x] = ind2sub([curBox(4)-curBox(2)+1,curBox(3)-curBox(1)+1],im);
            pts(k,:) = [x+curBox(1), y+curBox(2)];
        end
    end
end