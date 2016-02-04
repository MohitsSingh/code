
% create image sizes struct....
% imgSizes = struct('sz',{},'I_rect',{});
% conf.get_full_image = true;
% for t = 1:length(s40_fra_faces_d)
%     if (mod(t,20)==0)
%         clc;
%         disp(t/length(s40_fra_faces_d))
%     end
%     [I,I_rect] = getImage(conf,s40_fra_faces_d(t));
%     imgSizes(t).sz = size2(I);
%     imgSizes(t).I_rect = I_rect;
% end

% save imgSizes imgSizes
% % s40_person_face_dets = struct('imageID',{},'boxes_rot',{},'boxes_algn',{});


function res = consolidateFaceDetections(conf,imgs,face_dets)
res = struct([]);
figure(1); clf;
for t = 1:length(imgs)
    if (mod(t,20)==0)
        clc;
        disp(t/length(imgs))
    end
    curImg = imgs(t);     
    conf.get_full_image = true;        
    [I,I_rect] = getImage(conf,curImg);
    I = im2uint8(I);
    
%     I = zeros(imgSizes(t).sz,'uint8');
%     I_rect = imgSizes(t).I_rect;
    
    %[I,I_rect] = getImage(conf,curImg);
    %I = im2uint8(I);
    all_rects = {};
    for u = 1:length(face_dets(t).detections)
        curBoxes = face_dets(t).detections(u).boxes;
        if isempty(curBoxes),curBoxes = -inf(1,6);end
        curBoxes(:,end+1) = face_dets(t).detections(u).rot;
        all_rects{end+1} = curBoxes;
    end
    all_rects = cat(1,all_rects{:});
    scores = all_rects(:,6);
    [m,im] = sort(all_rects(:,6),'descend');
    % rotate to coordinate system of original image
    rects_rotated = rotate_bbs(all_rects(:,1:4),I,all_rects(:,7));
    centers = cellfun2(@(x) mean(x,1), rects_rotated);centers = cat(1,centers{:});
    box_heights = all_rects(:,3)-all_rects(:,1);
    % forece the boxes to be axis aligned
    rects_axis_algnd = [inflatebbox([centers centers],box_heights,'both',true) all_rects(:,5:6)];
    % perform nms on the axis aligned boxes
    top = nms(rects_axis_algnd, 0.1);
    [ovps,ints] = boxesOverlap(rects_axis_algnd,I_rect); % this does nothing if rectangle is entire image
    [~,~,s] = BoxSize(rects_axis_algnd);
    top = intersect(top,find(ints./s > 0.8),'stable');
    
    %res(t).imageID = curImg.imageID;
    res(t).boxes_rot = all_rects(top,:);
    res(t).boxes_algn = rects_axis_algnd(top,:);
    %     continue;
    % %
    
    toVisualize = false;
    if (toVisualize)
        mm = 2;
        nn = 2;
        
        for u = 1
            k = im(u);
            curRect = all_rects(k,:);
            I_rot = imrotate(I,curRect(7),'bilinear','crop');
            curRect_rot = rects_rotated{k};
            clf; vl_tightsubplot(mm,nn,1);imagesc2(I_rot); plotBoxes(curRect(1:4));
            vl_tightsubplot(mm,nn,2); imagesc2(I); plotPolygons(curRect_rot,'g-','LineWidth',2);
            plotBoxes(rects_axis_algnd(k,:),'m--','LineWidth',2);
            vl_tightsubplot(mm,nn,3); imagesc2(I); plotBoxes(rects_axis_algnd(top,:),'g-','LineWidth',2);
            plotBoxes(I_rect,'r-.','LineWidth',2);
            vl_tightsubplot(mm,nn,4); imagesc2(I); plotBoxes(rects_axis_algnd(top(1),:),'g-','LineWidth',2);
            drawnow; pause
        end
    end
end