dataDir='~/storage/mscoco'; dataType='val2014';
annFile=sprintf('%s/annotations/instances_%s.json',dataDir,dataType);
if(~exist('coco','var')), coco=CocoApi(annFile); end

%% display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds());
nms={cats.name}; fprintf('COCO categories: ');
fprintf('%s, ',nms{:}); fprintf('\n');
nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
fprintf('%s, ',nms{:}); fprintf('\n');

%% get all images containing given categories, select one at random
catIds = coco.getCatIds('catNms',{'person','toothbrush'});
imgIds = coco.getImgIds('catIds',catIds );
imgId = imgIds(randi(length(imgIds)));

%% load and display image
I = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img.file_name));
figure(1); imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display instance annotations
%annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
annIds = coco.getAnnIds('imgIds',imgId);
anns = coco.loadAnns(annIds); coco.showAnns(anns);


