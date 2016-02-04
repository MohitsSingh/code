function imgs_and_faces = collectFaceDetections(params,imgDir,outDir,toShow)

imgs_and_faces = struct('path',{},'faces',{});
if nargin < 4
    toShow = false;
end

D = dir(fullfile(outDir,'*.mat'));
tic_id = ticStatus('loading face detections...', .5, 1);
for t = 1:length(D)
    imgs_and_faces(t).path = fullfile(imgDir,[D(t).name(1:end-4) '.jpg']);
    L = load(fullfile(outDir,D(t).name));
    imgs_and_faces(t).faces = L.detections.boxes;
    if (isfield(L.detections,'feats'))
        imgs_and_faces(t).feats = L.detections.feats;
    else
        imgs_and_faces(t).feats = [];
    end
    tocStatus(tic_id,t/length(D));
end
% for t = 1:length(params)
%     t
%     imgs_and_faces(t).path = params(t).path;
%     if (exist(j2m(outDir,params(t).name),'file'))
%         L = load(j2m(outDir,params(t).name));
%         imgs_and_faces(t).faces = L.detections.boxes;
%     end
% end

if (~toShow)
    return
end


img_inds = {};
boxes = {};
for t = 1:length(D)
    boxes{t} = imgs_and_faces(t).faces;
    img_inds{t} = ones(size(boxes{t},1),1)*t;
end




boxes = cat(1,boxes{:});
img_inds = cat(1,img_inds{:});

[u,iu] = sort(boxes(:,end),'descend');
%%

close all
figure(1)
jump_ = 10;
for ik = 1:jump_:length(u)
    ik
    k = iu(ik);
    %     k = ik
    bbox = boxes(k,:);
    if (bbox(end)>0)
        continue
    end
    imgPath = imgs_and_faces(img_inds(k)).path;
    
    clf; imagesc2(imread(imgPath));
    plotBoxes(bbox);
    title(num2str(bbox(end)));
    drawnow;
    %     pause
    pause(.1)
end
