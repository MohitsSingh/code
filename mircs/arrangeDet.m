function det_ = arrangeDet(det_,field)

for k = 1:length(det_)
    det_(k) = arrangeHelper(det_(k));
    
end

    function d = arrangeHelper(d)
        if (strcmp(field,'index'))
            [~,ii] = sort(d.cluster_locs(:,11),'ascend');
        elseif strcmp(field,'score')
            [~,ii] = sort(d.cluster_locs(:,12),'descend');
        else
            error('may only arrange detections by image index or score');
        end
        d.cluster_locs = d.cluster_locs(ii,:);
    end
end