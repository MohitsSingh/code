%%% annotateObjects_rot
% % % % initpath;
% % % % config;
% % % % conf.get_full_image = true;
% % % % [learnParams,conf] = getDefaultLearningParams(conf,1024);
% % % % load fra_db.mat;
% % % % classNames = unique(all_class_names);
% % % % class_labels = [fra_db.classID];
% % % % classes = unique(class_labels);
% % % % isTrain = [fra_db.isTrain];
% % % % initialized = true;
% % % % classNames = conf.classes(classes);
% % % % imageNames={fra_db.imageID};


%% obtain whatever annotation is available for each image.

% toCopy = false(size(fra_db));
% tmpDir = '~/storage/tmp';
% 
% for t = 1:length(fra_db)
%     curImageData = fra_db(t);
%     [I,I_rect] = getImage(conf, curImageData);
%     
%     clf;imagesc2(I);
%     plotBoxes(curImageData.I_rect,'m-','LineWidth',2);
%     plotBoxes(curImageData.obj_bbox);
%     plotBoxes(curImageData.faceBox,'r-');    
%     % see if there is an accurate object annotation...
%     [objectSamples,objectNames] = getGroundTruth(conf,{curImageData.imageID} ,true);
%     if (isempty(objectSamples))
%         toCopy(t) = true;
%         imgPath = getImagePath(conf,curImageData.imageID);
%         newPath = fullfile(tmpDir,curImageData.imageID);
%         copyfile(imgPath,newPath);
%     end
% %         clf;imagesc2(I);
% %         [~,api]=imRectRot('rotate',1);
% % %         api.setPosSetCb(@helper_anno_fun);
% % %         [x,y,b] = ginput(1);
% %         
% %         plotBoxes(I_rect,'m--','LineWidth',2);
% %         objs = bbGt( 'create', 1 );
% %         objs.lbl = 'obj';
% %         bb = api.getPos();
% %         objs.bb = bb;
% %     end
% end

mkdir(tmpDir);
for t = 1:length(fra_db)
    
end


%%
close all;
nTrue = nnz(class_labels);
override = false;
for k = 1:length(newImageData)
    if (~class_labels(k)),continue,end;
    nTrue = nTrue-1;
    
    %     if k > 4325
    if (~override && ~isempty(newImageData(k).obj_bbox)),continue,end
    %     end
    
    disp([num2str(nTrue) ' to go...']);
    currentID = newImageData(k).imageID;
    if ~strcmp(currentID,'drinking_249.jpg'),continue,end
    fName = fullfile(annotationDir,[currentID '.txt']);
    needToAnnotate = true;
    if (exist(fName,'file'))
        [~,bb] = bbGt('bbLoad',fName);
        if (~isempty(bb))
            needToAnnotate = false;
        end
    end
    needToAnnotate = needToAnnotate || override;
    if (needToAnnotate)
        [I,I_rect] = getImage(conf,newImageData(k));
        clf; imagesc2(I);hold on;
        plotBoxes(I_rect,'m--','LineWidth',2);
        [~,api]=imRectRot('rotate',0);
        objs = bbGt( 'create', 1 );
        objs.lbl = 'obj';
        bb = api.getPos();
        objs.bb = bb(1:4);
        bbGt( 'bbSave', objs, fName );
    end
    bb(3:4) = bb(3:4)+bb(1:2);
    newImageData(k).obj_bbox = bb;
end

[groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);