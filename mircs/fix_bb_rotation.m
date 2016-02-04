function bb = fix_bb_rotation(bb,I)
    sz = size(I);
    centerPoint = fliplr(sz(1:2)/2);
    for k = 1:size(bb,1)
        if bb(k,13)==0
            continue
        end
        curPts = box2Pts(bb(k,1:4));
%         boxSz = bb(3:4)-bb(1:2);
        % shift to center, rotate, and shift back. need only the center        
        
        imshow(I); hold on; hold on;plot(curPts(:,1),curPts(:,2),'g-+')
        curPts_c = bsxfun(@minus,curPts,centerPoint);
        R = rotationMatrix(-pi*bb(k,end)/180);
        curPts = bsxfun(@plus,curPts_c*R,centerPoint);
         hold on; hold on;plot(curPts(:,1),curPts(:,2),'r-+')
        bb(k,1:4) = pts2Box(curPts);
    end
end