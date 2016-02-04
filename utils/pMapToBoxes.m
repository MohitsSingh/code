function boxes = pMapToBoxes(pMap,rad,maxBoxes)

nBoxes = 0;
boxes = zeros(maxBoxes,5);
[subs,vals] = nonMaxSupr( double(pMap), 3,[],maxBoxes);
boxes = [inflatebbox(subs(:,[2 1 2 1]), rad, 'both',true) vals];
% figure,imagesc2(pMap);
% plotPolygons(subs(:,[2 1]),'m+')
% 
% while (nBoxes <= maxBoxes)
%     nBoxes = nBoxes+1;
%     [m,im] = max(pMap(:));
%     [y,x] = ind2sub(size2(pMap),im);
%     curBox = [inflatebbox([x y x y], rad, 'both',true) m];
%     boxes(nBoxes,:) = curBox;
%     pMap(poly2mask2(box2Pts(curBox),size2(pMap))) = 0;
% end

end