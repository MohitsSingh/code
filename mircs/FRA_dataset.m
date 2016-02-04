%% 23/6/2014 make the face-related action dataset ready.
% 1. select desired classes.
% 2. for each class, specify the subset of image-names / ids within the
% entire dataset
% 3. for each image, specify :
% 3.1 class name, location of head, upper body, action object.

default_init;
specific_classes_init;

% go over all images and verify that the upper-body,face,action object are
% correct.
data_sub = newImageData(validIndices);
%%
fra_db = data_sub;
debug_ = false;
%
for t = 1:length(data_sub)
    t
    curImgData = data_sub(t);
    I_rect = curImgData.I_rect;
    faceBox = curImgData.gt_face;
    if (~isfield(faceBox,'bbox'))
        faceBox = curImgData.faceBox + I_rect([1 2 1 2]);
    else
        faceBox = faceBox.bbox;
    end
    
    fra_db(t).faceBox = faceBox(1:4);
    fra_db(t).obj_bbox = fra_db(t).obj_bbox(1:4);
    fra_db(t).class = classNames{class_labels(t)};
    fra_db(t).classID = class_labels(t);
    
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
        plotBoxes(inflatebbox(faceBox,3,'both',false),'m-','LineWidth',2);
        plotBoxes(curImgData.obj_bbox,'r-','LineWidth',2);
        plotPolygons({objects.poly},'y-','LineWidth',2);
        drawnow;
        pause(.01)
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
        srcImage = getImagePath(conf,fra_db(t).imageID);
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

%% show the results...
u = 0;
clf;
% if (debug_)
for t=10:50:length(fra_db)
    [I,I_rect] = getImage(conf,fra_db(t).imageID);
    u = u +1;
    if (u==10), break;end
    vl_tightsubplot(3,3,u);
    imagesc2(I);
    hold on;
    plotBoxes(fra_db(t).faceBox);
    if (~isempty(fra_db(t).objects))
        plotPolygons({fra_db(t).objects.poly},'y-','LineWidth',2);
    end
    plotBoxes(fra_db(t).hands,'m--','LineWidth',3);
    drawnow;
    %         pause;
end
% end
%%
% % plot ratio of object bounding box intersecting with growing radii of
% head bounding box....

% do this per class.


for t = 1:length(fra_db)
    fra_db(t).obj_bbox = fra_db(t).obj_bbox(1:4);
end

for t = 1:length(fra_db)
    fra_db(t).imgIndex = t;
end

%%
for iClass = 1:length(unique([fra_db.classID]))
    sel_ = [fra_db.classID] == iClass;
    f_sel = find(sel_,1,'first');
    all_heads = cat(1,fra_db(sel_).faceBox);
    all_objects = cat(1,fra_db(sel_).obj_bbox);
    [~,~,obj_areas] = BoxSize(all_objects);    
    inflate_factor = linspace(1,10,100);
    mean_ratio = zeros(size(inflate_factor));
    min_ratio = zeros(size(inflate_factor));
    median_ratio = zeros(size(inflate_factor));
    all_in = zeros(size(inflate_factor));
    for ii = 1:length(inflate_factor)
        inflated_heads = inflatebbox(all_heads,inflate_factor(ii),'both',false); % :-) --> : - )
        ints = BoxIntersection(inflated_heads,all_objects);
        [~,~,int_area] = BoxSize(ints);
        mean_ratio(ii) = mean(int_area./obj_areas);
        min_ratio(ii) = min(int_area./obj_areas,[],1);
        median_ratio(ii) = median(int_area./obj_areas);
        all_in(ii) = mean((int_area./obj_areas) >.5);
    end
    
    clf,plot(inflate_factor,mean_ratio,'b-');
    hold on; plot(inflate_factor,median_ratio,'r-');
    hold on; plot(inflate_factor,all_in,'g-');
    hold on; plot(inflate_factor,min_ratio,'k-.');
    
    legend({'mean','median','above 90%'});
    title({fra_db(f_sel).class, 'Area of action objects in vicinity of faces'});
    xlabel('face inflation factor');
    ylabel('obj.ratio inside face area');grid on;
    pause;
end

%% some statistics about sizes of bounding boxes
roiParams.infScale = 1.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
roiParams.useCenterSquare = false;
roiParams.squareSide = 30*roiParams.absScale/105;

classInfo = struct('classID',{},'className',{},'mean_bb',{});

for iClass = 1:length(classes)
    classInfo(iClass).classID = iClass;
    classInfo(iClass).className = classNames{iClass};
    sel_class = find([fra_db.classID] == iClass & isTrain);
    class_boxes = {};
    for ik = 1:length(sel_class)
        ik
        k = sel_class(ik);
        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k));
        iObj = strmatch('obj',{rois.name});
        if (isempty(iObj)),continue,end
        class_boxes{end+1} = cat(1,rois(iObj).bbox);
    end
    
    class_boxes = cat(1,class_boxes{:});
    [w h] = BoxSize(class_boxes);
    mean_w = mean(w);
    mean_h = mean(h);
    classInfo(iClass).mean_bb = [mean_w mean_h];
    classInfo(iClass).class_boxes = class_boxes;
    %figure,plot(w,h,'r+');
end

% save fra_db fra_db

save classInfo classInfo


%%


