function cur_img_dets = run_detector_nonms(I,patchModels,param)
% run a sliding window detector , possibly with rotations, on a set of
% images.
% scales = 1;
% debugging = true;
% all_dets = {};
scales = sort(param.scales,'ascend');
cur_img_dets = {};
sc_total = [];
for r = param.rotations;
    %         r
    II = imrotate(I,r,'bilinear','crop');
    for iScale = 1:length(scales)
        curScale = scales(iScale);
%         nKeep = 10;
        nKeep = inf;
        [detections, scores, hog, f] = detect(II, patchModels, param.cellSize, curScale,nKeep);
        
        if (param.avoid_out_of_image_dets)
            keep = inImageBounds([3 3 size2(II)-2],detections');
            detections = detections(:,keep);
            scores = scores(keep);
        end
        
        detections_centers = boxCenters(detections');
        %f = imrotate(f,-r,'bilinear','crop');
        %         if (isempty(sc_total))
        %             sc_total = -inf(size(f));
        %             %                 sc_total = -inf(size2(I_sub));
        %             rad = size(sc_total,1)/2;
        %             [xx,yy] = meshgrid(1:size(sc_total,2),1:size(sc_total,1));
        %             z = ((xx-rad).^2+(yy-rad).^2).^.5 <rad;
        %         end
        %         f = imResample(f,size(sc_total),'bilinear');
        %         sc_total = max(f,sc_total);
%         clf;imagesc2(II); plotPolygons(detections_centers,'r+')
        z = [sind(-r*ones(size(scores)));cosd(-r*ones(size(scores)))]'*10;
%         quiver(detections_centers(:,1),detections_centers(:,2),z(:,1),z(:,2));
        detections_centers = rotate_pts(detections_centers,-r*pi/180,fliplr(size2(II)/2));
%         blah =  inImageBounds(II,detections_centers);
%         f = find(~blah);
        
        %clf;imagesc2(I); plotPolygons(detections_centers,'r+')
        
%         pause;
        %             clf; imagesc(imResample(f,size2(II)));
        %             plotPolygons(detections_centers,'g+');
        % pause
        cur_img_dets{end+1} = [detections_centers';-r*ones(size(scores));scores]';
    end
%     if (param.debugging)
%         sc_total(~z) = -inf;
%         clf;vl_tightsubplot(1,2,1);imagesc2(II)
%         vl_tightsubplot(1,2,2); imagesc2(sc_total);
%         drawnow;
%     end
    %                             pause(.01);
    continue
end
cur_img_dets = double(cat(1,cur_img_dets{:}));
% all_dets{t} = cur_img_dets;
% if (param.debugging)
%     vl_tightsubplot(1,2,1);imagesc2(I_sub)
%     vl_tightsubplot(1,2,2); imagesc2(sc_total);
%     dpc;
%     edn
% end

%     all_dets = cat(1,all_dets{:});



