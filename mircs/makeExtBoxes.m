function extBoxes = makeExtBoxes(conf,dets,boxes)
extBoxes = {};
for k = 1:size(dets.cluster_locs,1)
%     if (~t_train(k))
%         continue;
%     end
    curBox =  dets.cluster_locs(k,:);
    lipBox = boxes(k,:);
    
    % flip the lip :-)
    
    %     disp(curBox(:,conf.consts.FLIP)==1)
    
    if (curBox(:,conf.consts.FLIP))
        lipBox = flip_box(lipBox,[128 128]);
    end
    
    [numRows numColumns area] = BoxSize(curBox);
    lipBox([1 3]) = numColumns*lipBox([1 3])/128 + curBox(1);
    lipBox([2 4]) = numRows*lipBox([2 4])/128+ curBox(2);
    
    faceCenter = boxCenters(curBox);
    lipCenter = boxCenters(lipBox);
    
    extBox = lipCenter + (lipCenter-faceCenter);
    extBox = [extBox extBox];
    extBoxes{k} = inflatebbox(extBox,[numColumns numRows]/1.3,'both',true);
end

extBoxes = cat(1,extBoxes{:});