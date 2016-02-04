%annotateDatabase

% initpath;
% config;
%
% [action_rois,true_ids] = markActionROI(conf,roiPath);
%
addpath('/home/amirro/code/3rdparty/LabelMeToolbox/');
addpath(genpath('/home/amirro/code/3rdparty/annotationTool'));
AnnotationTool

%% annotate faces in the dataset.
annotateMissingFaces;
%% mark objects in the dataset...
annotateObjects;
%%
% annotate objects of interaction
% conf.class_subset = conf.class_enum.DRINKING;
% [action_rois,true_ids] = markActionROI(conf);
% conf.class_subset = conf.class_enum.SMOKING;
% action_rois = markActionROI(conf);
% conf.class_subset = conf.class_enum.BLOWING_BUBBLES;
% action_rois = markActionROI(conf);
% conf.class_subset = conf.class_enum.BRUSHING_TEETH;
% action_rois = markActionROI(conf);

annotateObjects_rot;

%% mark mouths in all images
mouth_annotations = annotateFaces(conf,fra_db,[],'mouth',1);
for t = 1:length(fra_db)
    if (size(mouth_annotations(t).xy',1)>1)
        t
        break
    end
    %     fra_db(t).mouth = mouth_annotations(t).xy';
end

%%
dbPath = '/home/amirro/storage/data/face_related_action';
load fra_db.mat;
%%
requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
conf.get_full_image = 1;
% fra_db = s40_fra;
for t = 1:1:length(fra_db)
    t
    %     if (fra_db(t).classID==5)
    annotateFacialLandmarks(conf,fra_db(t),requiredKeypoints,dbPath);
    %     end
end

%% annotate all faces doing anything in fra_db
annotateMissingFaces2;

%% mark all objects which belong to acting person.

% fix fra db:
% each object should appear only within the face-box. Hence, if more than
% one object appears, choose which should be removed.
for t = 1:length(fra_db)
    for u = 1:length(fra_db(t).objects)
        fra_db(t).objects(u).toKeep = true;
    end
end


%%

get_rois_fra(conf,fra_db(202))
RIGHT_MOUSE = 3;
LEFT_MOUTH = 1;
for t = 1:length(fra_db)
    t
    if length(fra_db(t).objects) > 1
        f = fra_db(t);
        objects = f.objects;
        toKeeps = [objects.toKeep];                        
        if (sum(toKeeps)-1<2)
            continue
        end
        [I,I_rect] = getImage(conf,f);
        finishedImage = false;
        while (~finishedImage)
            clf; imagesc2(I);
            z = zeros(size2(I));
            plotBoxes(fra_db(t).faceBox);
            plotBoxes(I_rect);
            objects = f.objects;
            for u = 1:length(objects)
                if (objects(u).toKeep)
                    plotPolygons(objects(u).poly,'g-','LineWidth',2);
                    z = z+u*poly2mask2(objects(u).poly,size2(I));
                end
            end
            
            title('left click to remove object, right click to continue');
            [x,y,b] = ginput(1);
            if (b==RIGHT_MOUSE)
                finishedImage = true;
                continue;
            end
            obj_index = z(round(y),round(x));
            if obj_index > 0
                f.objects(obj_index).toKeep = false;
            end
            fra_db(t).objects = f.objects;
        end
        
        %dpc;
    end
end
save fra_db_2015_10_08.mat fra_db
%% do the same for hands....
%%
% for t = 700:length(fra_db)
%     for u = 1:length(fra_db(t).objects)
%         fra_db(t).hands_to_keep = true(size(fra_db(t).hands,1),1);
%     end
% end
RIGHT_MOUSE = 3;
LEFT_MOUTH = 1;
for t = 1:length(fra_db)    
    if (fra_db(t).classID ~=4), continue,end
    if length(fra_db(t).hands) > 1
        t
        f = fra_db(t);
        hands = f.hands;
        toKeeps = fra_db(t).hands_to_keep;
        if (sum(toKeeps)<2)
            continue
        end
        [I,I_rect] = getImage(conf,f);
        finishedImage = false;
        while (~finishedImage)
            clf; imagesc2(I);
            z = zeros(size2(I));
%             plotBoxes(fra_db(t).faceBox);
            plotBoxes(I_rect);
            hands = f.hands;
            for u = 1:size(hands,1)
                if (toKeeps(u))
                    plotBoxes(hands(u,:));
                    z = z+u*poly2mask2(hands(u,:),size2(I));
                end
            end
            
            title('left click to remove object, right click to continue');
            [x,y,b] = ginput(1);
            if (b==RIGHT_MOUSE)
                finishedImage = true;
                continue;
            end
            obj_index = z(round(y),round(x));
            if obj_index > 0
                toKeeps(obj_index) = false;
            end
            fra_db(t).hands_to_keep = toKeeps;
        end
        
        %dpc;
    end
end
% save fra_db fra_db;
%% get hands from fra_db and complete to piotr annotations

myTmpDir = '~/tmp_imgs';
mkdir(myTmpDir);
for t = 1:length(fra_db)
    if fra_db(t).classID ~=4,continue,end
    clc
    disp(t)
    p = '/home/amirro/data/Stanford40/annotations/hands/';
    f = j2m(p,fra_db(t),'.jpg.txt');
    if exist(f,'file'),continue,end
    f1 = j2m(myTmpDir,fra_db(t),'.jpg.txt');
    if (exist(f1,'file'))
        movefile(f1,f);
        delete(fullfile(myTmpDir,fra_db(t).imageID));
        continue
    end
    %break
    disp('a rectangle is missing for this image:');
    disp(fra_db(t).imageID);
    bb = fra_db(t).hands;
    if (exist(fullfile(myTmpDir,fra_db(t).imageID),'file'))
        continue
    end
    if size(bb,1) > 0
        disp('but found hands annotated elsewhere, transferring...');
        disp('here''s out it looks:');
        [I,I_rect] = getImage(conf,fra_db(t));
        plotBoxes(I_rect);
    
        clf;imagesc2(I); plotBoxes(bb);
        plotBoxes(I_rect,'m--','LineWidth',2);
        title('left click to accept, right click to modify');
        [x,y,b] = ginput(1);
        if b==3
            disp('writing to temp dir, annotate this later');
            imwrite(I,fullfile(myTmpDir,fra_db(t).imageID));
        else
            disp('accepting, writing to gt dir');
            pause(.1);
            bbPath = f;
            objs = bbGt( 'create', 1 );
            objs.lbl = 'hand';
            bb(:,3:4) = bb(:,3:4)-bb(:,1:2);
            bb = bb(1:4);
            objs.bb = bb;
            bbGt( 'bbSave', objs, bbPath);
        end
    end
end

% bbLabeler({'hand'},myTmpDir,myTmpDir);

%%
bbLabeler({'hand'},conf.imgDir,'/home/amirro/data/Stanford40/annotations/hands/');
%%
annotateFacePolygons






