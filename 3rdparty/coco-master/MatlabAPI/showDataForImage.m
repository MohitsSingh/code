function showDataForImage(coco,dataDir,dataType,caps,curAction,adverbMap,actionMap,categoryMap)

imgId = curAction.image_id;

%% load and display image
img = coco.loadImgs(imgId);
I = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img.file_name));
figure(1);
clf;
imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display caption annotations
annIds = caps.getAnnIds('imgIds',imgId);
anns = caps.loadAnns(annIds); caps.showAnns(anns);

%% load and display instance annotations
annIds = coco.getAnnIds('imgIds',imgId,'iscrowd',[]);
anns = coco.loadAnns(annIds); coco.showAnns(anns);


hold on;
%% show the action for this image.
obj_ids = [anns.id];
fSubject = find(obj_ids == curAction.subject_id);
subjectBox = anns(fSubject).bbox;
subjectBox(3:4) = subjectBox(3:4)+subjectBox(1:2);
subjectCategory = categoryMap(anns(fSubject).category_id).name;


plotBoxes(subjectBox);
fprintf('%s:\n',subjectCategory);
fprintf('actions:\n-------\n');
for iAction = 1:length(curAction.visual_actions)
    fprintf('%s\n',actionMap(curAction.visual_actions(iAction)).name)
end

fprintf('adverbs:\n-------\n');
for iAction = 1:length(curAction.visual_adverbs)
    fprintf('%s\n',adverbMap(curAction.visual_adverbs(iAction)).name)
end

fObject = find(obj_ids == curAction.object_id);
if ~isempty(fObject)
    objectBox = anns(fObject).bbox;
    objectBox(3:4) = objectBox(3:4)+objectBox(1:2);
    plotBoxes(objectBox,'r-','LineWidth',2);

    objectCategory = categoryMap(anns(fObject).category_id).name;    
    fprintf('object: %s.\n',objectCategory);
end

