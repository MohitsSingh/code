%% Demo for the CocoApi (see CocoApi.m)

%% initialize COCO api for instance annotations
addpath(genpath('~/code/utils'));
%dataDir='../';
dataDir='~/storage/mscoco'
dataType='train2014';
annFile=sprintf('%s/annotations/instances_%s.json',dataDir,dataType);
if(~exist('coco','var')), coco=CocoApi(annFile); end

%% display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds());
nms={cats.name}; fprintf('COCO categories: ');
fprintf('%s, ',nms{:}); fprintf('\n');
nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
fprintf('%s, ',nms{:}); fprintf('\n');

%% get all images containing given categories, select one at random
catIds = coco.getCatIds('catNms',{'frisbee'});
% catIds = coco.getCatIds('catNms',{'dog','cat'});

%%
imgIds = coco.getImgIds('catIds',catIds );
imgId = imgIds(randi(length(imgIds)));

%% load and display image
img = coco.loadImgs(imgId);
I = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img.file_name));
figure(1); imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display instance annotations
annIds = coco.getAnnIds('imgIds',imgId,'iscrowd',[]);
anns = coco.loadAnns(annIds); coco.showAnns(anns);

%% initialize COCO api for caption annotations
annFile=sprintf('%s/annotations/captions_%s.json',dataDir,dataType);
if(~exist(annFile,'file')), return; end
if(~exist('caps','var')), caps=CocoApi(annFile); end

%% load and display caption annotations
annIds = caps.getAnnIds('imgIds',imgId);
anns = caps.loadAnns(annIds); caps.showAnns(anns);


%% now integrate the coco-a
addpath('~/code/utils');
verbNetFile = '~/storage/mscoco/visual_verbnet_beta2015.json';
verbNetData = gason(fileread(verbNetFile));

adverbMap = makeIDMap(verbNetData.visual_adverbs);
actionMap = makeIDMap(verbNetData.visual_actions);

actionFile = '~/storage/mscoco/cocoa_beta2015_fix.json';
actionData = gason(fileread(actionFile));

cocoa2 = actionData.annotations.a2;

% now go over some images and show their annotations and actions.
anno_id = 1;

curAction = cocoa2(anno_id);

%%

categoryMap = makeIDMap(coco.data.categories);

%% show image:
visited_images = []
for u = 100:length(cocoa2)
    clc
    
    if ismember(cocoa2(u).image_id,visited_images)
        continue
    else
        visited_images = [visited_images cocoa2(u).image_id]
    end
    showDataForImage(coco,dataDir,dataType, caps,cocoa2(u),adverbMap,actionMap,categoryMap);
    dpc
end