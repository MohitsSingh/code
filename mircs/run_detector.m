function all_dets = run_detector(imgs,patchModels,param)
% run a sliding window detector , possibly with rotations, on a set of
% images.
% scales = 1;
% debugging = true;
all_dets = {};
for t = 1:length(imgs)
    t
    [scales] = sort(param.scales,'ascend');
    I_sub = imgs{t};
    %     scales = 1;
    %     figure(1);
    cur_img_dets = {};
    sc_total = [];
    for r = param.rotations;
        %         r
        II = imrotate(I_sub,r,'bilinear','crop');
        for iScale = 1:length(scales)
            curScale = scales(iScale);
            nKeep = 100;
            [detections, scores, hog, f] = detect(II, patchModels, param.cellSize, curScale,nKeep);            
%             keep = nms([detections;scores]',.5);           
            keep = boxsuppress2(detections,scores,.5);
            detections = detections(:,keep);
            scores = scores(keep);
            detections_centers = boxCenters(detections');
                       
            f = imrotate(f,-r,'bilinear','crop');
            if (isempty(sc_total))
                sc_total = -inf(size(f));
                %                 sc_total = -inf(size2(I_sub));
                rad = size(sc_total,1)/2;
                [xx,yy] = meshgrid(1:size(sc_total,2),1:size(sc_total,1));
                z = ((xx-rad).^2+(yy-rad).^2).^.5 <rad;
            end
            f = imResample(f,size(sc_total),'bilinear');
            sc_total = max(f,sc_total);

            detections_centers = rotate_pts(detections_centers,-r*pi/180,fliplr(size2(II)/2));
%             clf; imagesc(imResample(f,size2(II)));
%             plotPolygons(detections_centers,'g+');
% pause

            cur_img_dets{end+1} = [detections_centers';-r*ones(size(scores));scores]';
        end
        if (param.debugging)
            sc_total(~z) = -inf;
            clf;vl_tightsubplot(1,2,1);imagesc2(II)
            vl_tightsubplot(1,2,2); imagesc2(sc_total);
            drawnow;
        end
        %                             pause(.01);
        continue
    end
    cur_img_dets = cat(1,cur_img_dets{:});
    all_dets{t} = cur_img_dets;
    if (param.debugging)
        vl_tightsubplot(1,2,1);imagesc2(I_sub)
        vl_tightsubplot(1,2,2); imagesc2(sc_total);
        dpc;
        edn
    end
    
%     all_dets = cat(1,all_dets{:});
    
end

