function groundTruth = alignGT(conf,groundTruth)
partIDS = [groundTruth.partID];
u = unique(partIDS);

% try to align all the shapes...
for iPart = 1:length(u)
    curSel = find(partIDS==u(iPart));
    for iObj = 1:length(curSel)
        curObj = groundTruth(curSel(iObj));
        x = curObj.polygon.x;
        y = curObj.polygon.y;
        x = round(x);
        y = round(y);
        x = x-min(x)+1;
        y = y-min(y)+1;
        R = roipoly(zeros(max(y),max(x)),x,y);
        rprops = regionprops(R,'orientation');
        orientation = rprops.Orientation;
        if (orientation < 0)
            groundTruth(curSel(iObj)).Orientation = 90+orientation;
        else
            groundTruth(curSel(iObj)).Orientation = orientation-90;
        end
%         rprops.Orientation
%         clf; subplot(1,2,1); imagesc(R); axis image; title('before');
%         subplot(1,2,2); imagesc(imrotate(R,-groundTruth(curSel(iObj)).Orientation)); axis image; title('after');
%                 pause;
    end
end

end