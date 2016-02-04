initpath;
run_once = 1;
if (run_once)
    defaultOpts;
    
    %     % custom opts
    %     globalOpts.numTrain = inf;
    %     globalOpts.det_rescale = 0;
    %     globalOpts.use_overlapping_negatives = true;
    %     globalOpts.scale_choice = [];
    %     globalOpts.numSpatialX = [1 2];
    %     globalOpts.numSpatialY = [1 2];
    %     globalOpts.debug  = 0;
    %     globalOpts.removeOverlappingDetections = true;
    %     globalOpts.maxTrain = inf;
    %     globalOpts.numTrain = inf;
    %     globalOpts.numWords = 1000;
    %     updateOpts;    
%     tt_0_1_1000;
        tt_0_1x2_1000;
%         tt_0_1_1000;
    %
    % tt_128_1x2_1000;
    %     tt_0_1_1000;
    %     tt_32_1x2;
    init;
    
    % model.vocab = model.vocab./repmat(sum(model.vocab),size(model.vocab,2
    
    % globalOpts.numSpatialX = [1 2 3];
    % globalOpts.numSpatialY = [1 2 3];
    % % globalOpts.det_rescale = 64;
    % for k = 1:25
    %     clf
    %     vl_plotsiftdescriptor(vocab(:,k));
    %     pause;
    % end
    %
end
if (1)
    %%
    % get the bboxes that belong to the train images.
    
    globalOpts.class_subset = 1;
    inds = 1:length(train_images);
    trainImageSel = inds;
    globalOpts.selection = 1;
    globalOpts.debug = 0;
    globalOpts.class_subset = 1;
    inds = 1:length(train_images);
    trainImageSel = inds;
    globalOpts.selection = 1;
    globalOpts.debug = 0;
    
    iter = 1;
    
    model = do_training(globalOpts,train_images,model,iter);
    
end

%%
% model = do_training_bow(globalOpts,train_images,model,retrain);

% end
%%
cls = 1;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_val']),'%s %d');
ids = ids(t==1);
globalOpts.debug = 0;

applyModel(globalOpts,model,ids,iter);
applyModel(globalOpts,model,test_images,iter);
%
%%
globalOpts.debug = 0;

%%
close all
% for cls =1:20
        cls = 1;
    close all;
    cls =1
    VOCopts.classes(cls)
    [ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_val']),'%s %d');
    % cls = 2;
    globalOpts.debug = 0;
    globalOpts.removeOverlappingDetections = true;
    ids = ids(t==1);
%     ids = ids(1:end);
%     ids_n = ids(t~=1);
%     ids_p = ids(t==1);
    collectResults(globalOpts,ids,cls,0,'',iter);
%     collectResults(globalOpts,test_images,cls,0,'',iter);
% end
%%

    
    %% evaluate...
    id = [globalOpts.exp_name '_comp3'];
%     for icls = 1:20
                icls =1;
        % %         icls = 10
        icls=1
        to_debug = 0;
        nDets =3000;
        cls_ = VOCopts.classes{icls};
        if (exist('cp','var'))
            [rec,prec,ap,cp] = VOCevaldet(VOCopts,id,cls_,1,nDets,cp,to_debug,globalOpts);
        else
            [rec,prec,ap,cp] = VOCevaldet(VOCopts,id,cls_,1,nDets,[],to_debug,globalOpts);
        end
%         pause;
%     end
    
%     plot(rec,prec)
    
%%
addpath('/home/amirro/code/3rdparty/RealtimeSiftSurfReleaseV1');

imageID = train_images{50};
I = imread(getImageFile(globalOpts,imageID));

figure,imshow(I);

% 
F = Image2HierarchicalGroupingWithOptions(I,.8,100,100,{'Size','Texture'},'Rgb');
sizes = zeros(size(F));
for k = 1:length(F)
    sizes(k) = numel(F{k}.mask);
end

[s,is] = sort(sizes,'descend');
F = F(is);
for k = 1:length(F)
    b = F{k}.rect;
    bigMask = false(size(I));    
    m = F{k}.mask;
    bigMask(b(1):b(3),b(2):b(4),:) = cat(3,m,m,m);
    I_ = zeros(size(I),'uint8');
    I_(bigMask) = I(bigMask);
    imshow(I_);
%     imshow(F{k}.mask);
    pause;
end

hold on;

imGray = rgb2gray(I);
[featSift infoSift] = DenseSift(imGray, 4, 1, 4, [0 0]);

clear featRgbSift;
[featSift infoSift] = DenseSift(imGray, 4, 1, 4, [0 0]);


[C,A] = vl_kmeans(featSift',1000);

% figure,hist(double(A),100);
% find the distance between each descriptor and the 
% centroid it was assigned.

Z = zeros(size(imGray));
Z(sub2ind(size(Z),round(infoSift.row),round(infoSift.col))) = A;

errors = sum((featSift(A,:)-C(:,A)').^2,2);

Z_error = zeros(size(imGray));
Z_error(sub2ind(size(Z),round(infoRgbSift.row),round(infoRgbSift.col))) = errors;


Z_error = imdilate(Z_error,ones(3));

% ZZ = Z_error(1:5:end,1:5:end);
% ZZ = imresize(ZZ,size(imGray));
figure,imagesc((Z_error.*(Z_error>1.2)).^2)
figure,imshow(Z_error.^2,[]);
figureputty,imshow(I);
% figure,imshow((),[]);
% figure,


figure,imagesc((Z(1:4:end,1:4:end)))



    