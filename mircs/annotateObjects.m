    
classNames = conf.classes(classes);
imageNames={newImageData.imageID};
[r,ir] = sort(imageNames);
% newImageData = newImageData(ir);
class_labels = zeros(1,length(newImageData));
for iClass = 1:length(classes)
    isClass = strncmp(classNames{iClass},imageNames,length(classNames{iClass}));
    class_labels(isClass) = iClass;
end

annotationDir = '/home/amirro/storage/data/Stanford40/annotations/objects';
ensuredir(annotationDir);

% newImageData = rmfield(newImageData,'obj_bbox');
all_rois = {};
all_ids = {};
% load whatever already exists

for iClass = 1:length(classes)
    iClass
    conf.class_subset = classes(iClass);
    [action_rois,true_ids] = markActionROI(conf);
    [lia,lib] = ismember(true_ids,imageNames);  
    for ik = 1:length(lib)
        t = lib(ik);
        newImageData(t).obj_bbox = action_rois(ik,:);
    end
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