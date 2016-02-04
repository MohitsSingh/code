%% 23/6/2014 make the face-related action dataset ready.
% 1. select desired classes.
% 2. for each class, specify the subset of image-names / ids within the
% entire dataset
% 3. for each image, specify :
% 3.1 class name, location of head, upper body, action object.


% initalization code

% go over all images and verify that the upper-body,face,action object are
% correct.

% load fra_db;

%%

% fra_db = fra_db;
debug_ = true;
%

faceDirs = {'~/storage/data/Stanford40/annotations/faces',
    '~/storage/data/Stanford40/annotations/faces_oct_31_14'};

newFaceDir = '~/storage/data/Stanford40/annotations/faces_sep_29_15';
ensuredir(newFaceDir);
debug_=false;
conf.get_full_image = true;
for t = 1:length(fra_db)
    t
    curImgData = fra_db(t);
    facePath = j2m(faceDirs{2},curImgData);
    L_face = load(facePath);
    faceBox = L_face.curImgData.faceBox;
    if iscell(faceBox)
        faceBox = faceBox{1};
    end
    faceBox = faceBox(1:4);
    plotBoxes(faceBox);
    %     dpc;continue;
    % get the ground-truth object annotations for this image.
    [objectSamples,objectNames] = getGroundTruth(conf,{fra_db(t).imageID},true);
    objects = struct('name',{},'poly',{});
    p = 0;
    for iObj = 1:length(objectSamples)
        curName = objectSamples{iObj}.name;
        if (any(strcmp(curName,{'face','hand'}))), continue,end;
        p = p+1;
        objects(p).name = curName;
        objects(p).poly = [objectSamples{iObj}.polygon.x(:) objectSamples{iObj}.polygon.y(:)];
    end
    fra_db(t).objects = objects;
    if (debug_)
        clf;
        [I,I_rect] = getImage(conf,curImgData.imageID);
        clf;imagesc2(I);
        hold on;
        plotBoxes(faceBox);
        %plotBoxes(inflatebbox(faceBox,3,'both',false),'m-','LineWidth',2);
        plotBoxes(curImgData.obj_bbox,'r-','LineWidth',2);
        plotPolygons({objects.poly},'y-','LineWidth',2);
        dpc
%         drawnow;
%         pause(.01)
    end
end
fra_db = rmfield(fra_db,{'faceLandmarks','faceScore','lipBox','sub_image',...
    'upperBodyDets','extra','gt_face','faceLandmarks_piotr','alternative_face',...
    'label'});
%%
% The following code is to annotate unmarked hands. Do this using piotr
% dollar's code, then integrate back into the DB. Run it only once (unless
% the dataset changed...)
updateBB_from_piotr = false;
ignoreImages = {'drinking_230','drinking_210','phoning_197','phoning_080',...
    'phoning_009','phoning_007','drinking_003','drinking_234','drinking_180',...
    'phoning_120'};

if (updateBB_from_piotr)
    tmpImgDir = '/home/amirro/data/Stanford40/JPEGImages_tmp';ensuredir(tmpImgDir);
    pDir = '/home/amirro/data/Stanford40/annotations/hands';
    % take care of hands now.
    load hand_locs;
    sourceImages = {hand_locs.sourceImage};
    fra_images = cellfun2(@(x) x(1:end-4), {fra_db.imageID});
    % for t = 1:length(fra_images)
    [lia,locb] = ismember(fra_images,sourceImages);
    for t = 1:length(lia)
        t
        fra_db(t).imageID
        srcImage =getImagePath(conf,fra_db(t).imageID);
        bbPath = j2m(pDir,srcImage,'.jpg.txt');
        if (lia(t))
            % add hands to fra
            if (isempty(hand_locs(locb(t)).rects))
                if (ismember(fra_images(t),ignoreImages))
                    continue;
                end
                'warning : empty rects'
                needToAnnotate = true;
                if (exist(bbPath,'file'))
                    obj = bbGt('bbLoad',bbPath);
                    if (~isempty(obj))
                        needToAnnotate = false;
                    end
                end
                if (needToAnnotate)
                    % annotate it!
                    [I,I_rect] = getImage(conf,fra_db(t));
                    clf; imagesc2(I);hold on;
                    plotBoxes(I_rect,'m--','LineWidth',2);
                    [~,api]=imRectRot('rotate',0);
                    objs = bbGt( 'create', 1 );
                    objs.lbl = 'obj';
                    bb = api.getPos();
                    bb = bb(1:4);
                    objs.bb = bb;
                    bbGt( 'bbSave', objs, bbPath );
                    bb(3:4) = bb(3:4)+bb(1:2);
                    fra_db(t).hands = bb;
                end
            else
                fra_db(t).hands = hand_locs(locb(t)).rects;
                continue;
            end
        end
        targetImage = j2m(tmpImgDir,srcImage,'.jpg');
        % %
        obj = bbGt('bbLoad',bbPath);
        for ii = 1:length(obj)
            curBB = obj(ii).bb;
            curBB(3:4) = curBB(3:4)+curBB(1:2);
            obj(ii).bb = curBB;
        end
        fra_db(t).hands = cat(1,obj.bb);
        if (exist(targetImage,'file')),continue,end;
        try
            copyfile(srcImage,targetImage,'f');
        catch me
        end
    end
end
% bbLabeler({'hand'},tmpImgDir,'/home/amirro/data/Stanford40/annotations/hands/');
bbLabeler({'hand'},conf.imgDir,'/home/amirro/data/Stanford40/annotations/hands/');



% facial landmarks - gt and mine
%%
% roiParams.infScale = 1.5;
% roiParams.absScale = 200;
% roiParams.centerOnMouth = false;
%
% D = defaultPipelineParams();
% roiParams = D.roiParams;
for t = 1:length(fra_db)
    curImgData = fra_db(t);
    t
    [kp_gt,goods] = loadKeypointsGroundTruth(curImgData,D.requiredKeypoints);
    fra_db(t).landmarks_gt = struct('xy',kp_gt(:,1:2),'goods',goods);
    [kp_preds,goods] = loadDetectedLandmarks(conf,curImgData);
    %    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImgData,roiParams);
    %    kp_preds
    fra_db(t).landmarks = struct('xy',kp_preds,'goods',goods);
% %     I = getImage(conf,curImgData);
% %     figure(1),clf;
% %     imagesc2(I);
% %     zoomToBox(curImgData.faceBox);
% %     plotPolygons(fra_db(t).landmarks_gt.xy,'g.','LineWidth',2);
% %     plotPolygons(fra_db(t).landmarks.xy,'r.','LineWidth',2);
% %     %    plotBoxes(kp_preds,'m-','LineWidth',3);
% %     dpc
end


%% face detections...
faceDir = '~/storage/faces_only_baw';
for t = 1:length(fra_db)
    t
    curImgData = fra_db(t);        
    R = load(j2m('~/storage/data/Stanford40/annotations/faces_oct_31_14',curImgData));
    
    cur_fra_struct = face_detection_to_fra_struct(conf,faceDir,curImgData.imageID);
    
    if isempty(cur_fra_struct.faceBox) 
        [I,I_rect] = getImage(conf,curImgData);
        [h,w] = BoxSize(I_rect);        
        cur_fra_struct.faceBox = round(makeSquare([I_rect(1:3) I_rect(2)+h/4]));
        cur_fra_struct.faceScore=-1;
        cur_fra_struct.faceComp=-1;
    else
        continue
    end
    %bb = R.curImgData.raw_faceDetections.boxes(1,:);    
    fra_db(t).faceBox_raw = cur_fra_struct.faceBox;
    fra_db(t).faceBox_raw_score = cur_fra_struct.faceScore;
    fra_db(t).faceBox_raw_comp = cur_fra_struct.faceComp;
%     continue
        I= getImage(conf,curImgData);
        clf; imagesc2(I);     
    plotBoxes(fra_db(t).faceBox);
    plotBoxes(fra_db(t).faceBox_raw,'r--');
    dpc%     
%     plotBoxes(bb);
% %     dpc
% %     figure(1),clf;
% %     imagesc2(I);
% %     zoomToBox(curImgData.faceBox);
% %     plotPolygons(fra_db(t).landmarks_gt.xy,'g.','LineWidth',2);
% %     plotPolygons(fra_db(t).landmarks.xy,'r.','LineWidth',2);
% %     %    plotBoxes(kp_preds,'m-','LineWidth',3);
% %     dpc
end

%% show the results - and correct them if necessary.

% if (debug_)
for offset = 1:50
    u = 0;
    clf;
    for t=offset:50:length(fra_db)
        curImgData = fra_db(t);
        [I,I_rect] = getImage(conf,curImgData.imageID);
        u = u +1;
        if (u==10), break;end
        vl_tightsubplot(3,3,u);
        imagesc2(I);
        hold on;
        plotBoxes(curImgData.faceBox);
        if (~isempty(curImgData.objects))
            plotPolygons({curImgData.objects.poly},'y-','LineWidth',2);
        end
        plotBoxes(curImgData.hands,'m--','LineWidth',3);
        plotBoxes(I_rect);
        my_text = text(10, 10,curImgData.imageID);
        my_text.FontSize = 15;
        my_text.Color = [1 0 0];
        my_text.Interpreter = 'none';
        [kp_gt,goods] = loadKeypointsGroundTruth(curImgData,D.requiredKeypoints);
        plotPolygons(kp_gt,'g.');
        %         L = load(R1);
        %pause;
    end
    dpc;
end






%%
fn ='~/annos_to_correct.txt';
fid = fopen(fn);
a = textscan(fid,'%s\n');
fclose(fid);
ids ={ fra_db.imageID};
ids_to_correct = cellfun2(@(x) [x '.jpg'],a{1});
ids_to_correct = unique(ids_to_correct);
[c,ia,ib] = intersect(ids_to_correct,ids);
% ids_to_correct{39}
%%

ib = 1;
ids_to_correct = 'phoning_179.jpg';

for it=1:length(ib)
    t = ib(it);
    [I,I_rect] = getImage(conf,fra_db(t).imageID);
    clf;
    imagesc2(I);
    hold on;
    plotBoxes(fra_db(t).faceBox);
    if (~isempty(fra_db(t).objects))
        plotPolygons({fra_db(t).objects.poly},'y-','LineWidth',2);
    end
    plotBoxes(fra_db(t).hands,'m--','LineWidth',3);
    plotBoxes(I_rect);
    my_text = text(10, 10,fra_db(t).imageID);
    my_text.FontSize = 15;
    my_text.Color = [1 0 0];
    my_text.Interpreter = 'none';
    bb = getSingleRect(true);
    fra_db(t).faceBox = bb;
end

%% ground truth facial landmarks

%% ground truth face polygons, from two possible locations
alternativeFacesDir = '/home/amirro/storage/data/Stanford40/annotations/faces_alt';
conf.get_full_image = true;
close all;
ids = {fra_db.imageID};
[ids,ir] = sort(ids);
fra_db = fra_db(ir);
for t = 1:length(fra_db)
    t    
    imgData = fra_db(t);
%     if imgData.isTrain,continue,end
%     if imgData.classID==1,continue,end
    altPath = j2m(alternativeFacesDir,imgData);
    found_anno = false;
    found_new_anno = false;
    if exist(altPath,'file')
%         continue
        found_new_anno = true;
        load(altPath);
    else
        [objectSamples] = getGroundTruth(conf,{imgData.imageID},true);
        objectNames = cellfun2(@(x) x.name,objectSamples);
%         disp(objectNames)
        u = cellfun(@any,strfind(objectNames,'face'));
        if any(u)
            u = find(u,1,'first');
%             continue
            xy = [objectSamples{u}.polygon.x,objectSamples{u}.polygon.y];
            found_anno = true;
        end
    end                
    assert(found_anno || found_new_anno);
        
    fra_db(t) = addToObjects(fra_db(t),'face',xy);
%     objects = imgData.objects;
%     bads = false(size(objects));
%     for  tt = 1:length(objects)
%         if isempty(objects(tt).poly) && isempty(objects(tt).name)
%             bads(tt) = true;
%         end
%     end
%     objects = objects(~bads);
%     iFace = find(strcmp({objects.name},'face'))
%     objects(iFace) = [];
%     p = length(objects);
%     objects(p+1).name = 'face';
%     objects(p+1).poly = xy;    
%     fra_db(t).objects = objects;
    continue
    
    fprintf('new:%d\nold:%d\n',found_new_anno,found_anno);
    I = getImage(conf,imgData);
    clf; imagesc2(I);
    zoomToBox(inflatebbox(imgData.faceBox,3,'both',false));
    if found_anno
        plotPolygons(xy,'g-','Linewidth',2);
        title('found anno');
        drawnow
%         pause(.1)
    elseif found_new_anno
        plotPolygons(xy,'g-','Linewidth',2);
        title('found new anno');
        drawnow
%       
    else
        h = impoly();
        xy = getPosition(h);
        save(altPath,'xy');
    end       
%     dpc
end
% save

% save fra_db fra_db;

for t = 1:length(fra_db)
    imgData = fra_db(t);
    clf; imagesc2(getImage(conf,imgData));
    plotPolygons(imgData.objects(2).poly,'g-','Linewidth',2);
    plotPolygons(imgData.objects(1).poly,'r-','Linewidth',2);
    dpc
end
