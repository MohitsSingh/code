%experiment 18
% train a face detector by piotr dollar.
initpath;
config;

load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat'); % ims
%montage3(ims(abs(yaw)<=10))
% ims = (ims(abs(yaw)<=10));
% ims = cellfun2(@(x) imResample(x,[40 40],'bilinear'),ims);
% 
% non_person_ids = getNonPersonIds(VOCopts);
% 
negImgDir = '~/storage/tmp/nonperson';
posImgDir = '~/storage/tmp/faces';

% a = multiRead(conf,posImgDir);
% mkdir(negImgDir);
% mkdir(posImgDir);
multiWrite(ims(1:5:end),posImgDir);
% 
% a = [];
% for k = 1:10:length(non_person_ids)
%     k
%     imagePath = getImagePathPascal(conf,non_person_ids{k});
%     system(sprintf('cp %s %s',imagePath,negImgDir));
% end

prms = acfTrain();

prms.posWinDir = posImgDir;
prms.negImgDir = negImgDir;
prms.modelDs = [40 40];
prms.modelDsPad = [40 40];
prms.name = 'acf_face_detector';
prms.stride = 2;
prms.nWeak=[32 128 512];
prms.nNeg = 5000;

detector = acfTrain(prms);


%% test a bit....
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
tt = train_ids(train_labels);
for k = 1:length(tt)
    I = getImage(conf,tt{k});
%     I = imResample(I,'bilinear');
    bbs = acfDetect(I,detector);
%     n1 = size(bbs,1);
%     bbs = bbNms(bbs,'thr',.5,'type','maxg','ovrDnm','union');
%     n2 = size(bbs,1);
%     n2-n1
    bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
%     pick = nms(bbs,.3);
    figure(1);clf;  imagesc (I); axis image; hold on;
    
    plotBoxes(bbs(pick(1),:),'g','LineWidth',2);
    %---------------->
    I = imResample(I,.5,'bilinear');
    bbs = acfDetect(I,detector);
%     n1 = size(bbs,1);
%     bbs = bbNms(bbs,'thr',.5,'type','maxg','ovrDnm','union');
%     n2 = size(bbs,1);
%     n2-n1
    bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
%     pick = nms(bbs,.3);
   figure(2);  clf; imagesc (I); axis image; hold on;
    
    plotBoxes(bbs(pick(1),:),'g','LineWidth',2);
    pause
end
