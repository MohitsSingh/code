% if (0)
% script for action recognition from hand images...

addpath('/home/amirro/code/3rdparty/vlfeat-0.9.14/toolbox');
addpath(genpath('/home/amirro/code/3rdparty/libsvm-3.12'));
addpath('/home/amirro/code/3rdparty/ssim');
vl_setup;


% vl_setup;
% addpath(genpath('D:\libsvm-3.12'));

opts.hands_locs_suff = 'hands_locs';
opts.hands_images_suff = 'hands_imgs';

% uncomment the following line if you wish to run the labeling tool.
%labeling_script;

% change this directory to where you put your standford40 dataset
%inputDir = 'D:\Stanford40\';
inputDir = '/home/amirro/data/Stanford40/sorted';
% actions_file = fullfile(inputDir,'bottle','actions.txt');
% [A,~] = textread(actions_file,'%s %s'); %#ok<REMFF1>

% choose a class subset
%%
% class_subset = {'bottle','can','cup','juicebox','wineglass','other'}
class_subset = {'bottle','cup'};

%imgDir = fullfile(inputDir,'JPEGImages');
imgDir = inputDir;
%%
close all;
clc
classData = struct('class',{},'img',{});
cnt = 0;
tic;
for iClass = 1:length(class_subset)
    
    classData(iClass).class = iClass;
    currentClass = class_subset{iClass};
    %currentDir = fullfile(imgDir, [currentClass '_' opts.hands_images_suff]);
    currentDir = fullfile(imgDir, [currentClass]);
    currentFiles = dir(fullfile(currentDir,'*.jpg'));
    for k = 1:length(currentFiles)
        if (toc > .3)
            disp(cnt)
            tic;
        end
        cnt = cnt+1;
        currentPath = fullfile(currentDir,currentFiles(k).name);
        classData(cnt).class = iClass;
        I = imread(currentPath);
        %         I = I(.25*end:.75*end,.25*end:.75*end,:);
        I = imresize(I,[64 NaN],'bilinear');
        classData(cnt).img = I;
    end
end


classes = [classData.class];
uniqueClasses = unique(classes);
classCounts = hist(classes,uniqueClasses)
%%

%%
% extract some features....

% self-sim params...
parms.size = 5;
parms.coRelWindowRadius=10;
% parms.numRadiiIntervals=2;
% parms.numThetaIntervals=4;

parms.numRadiiIntervals=3;
parms.numThetaIntervals=6;

parms.varNoise=25*3*36;
parms.autoVarRadius=1;
parms.saliencyThresh=0;
parms.nChannels=3;

radius=(parms.size-1)/2; % the radius of the patch
marg=radius+parms.coRelWindowRadius;


tic
for k = 1:length(classData)
    if (toc > .5)
        disp(100*k/length(classData));
        tic;
    end
    %
    %     I = im2single(classData(k).img);
    %     imshow(I);
    %     pause(.01)
    I = classData(k).img;
    marg=radius+parms.coRelWindowRadius;
    
    % Compute descriptor at every 5 pixels seperation in both X and Y directions
    %     [allXCoords,allYCoords]=meshgrid([marg+1:5:size(I,2)-marg],...
    %                                  [marg+1:5:size(I,1)-marg]);
    %     allXCoords=allXCoords(:)';
    %     allYCoords=allYCoords(:)';
    %
%         [resp,drawCoords,salientCoords,uniformCoords]=ssimDescriptor(double(I) ,parms ,32, 32);
    %
    I = rgb2gray(im2single(I));
    %     [F,D] = vl_phow(I,'Sizes',5, 'FloatDescriptors' , true,'Step',16);
    [F,D] = vl_sift(I,'Frames',[size(I,2)/2,size(I,1)/2,5,0]');
    [F,D1] = vl_sift(I,'Frames',[size(I,2)/2,size(I,1)/2,4,0]');
    [F,D2] = vl_sift(I,'Frames',[size(I,2)/2,size(I,1)/2,3,0]');
    
    [F,D3] = vl_sift(I,'Frames',[size(I,2)*.25,size(I,1)*.25,4,0]');
    [F,D4] = vl_sift(I,'Frames',[size(I,2)*.25,size(I,1)*.75,4,0]');
    [F,D5] = vl_sift(I,'Frames',[size(I,2)*.75,size(I,1)*.25,4,0]');
    [F,D6] = vl_sift(I,'Frames',[size(I,2)*.75,size(I,1)*.75,4,0]');
    
    
    D = [D;D1;D2;D3;D4;D5;D6];
%     [~,D] = vl_sift(I,'Frames',[size(I,2)/2,size(I,1)/2,4,0]');
    
    %     figure
    %     for k = 1:20
    %     clf
    %
    %     imshow(I);
    %     hold on;
    %     vl_plotsiftdescriptor(D,F)
    % %     vl_plotsiftdescriptor(D(:,k),F([1 2 4 3],k))
    %     pause;
    %     end
    
    %     [F,D] = vl_phow(im2single(I),'Sizes',5,'Step',2);
    
    
    %     D = [resp];%;double(D)/255];
    
    
    %     D = D(:);
    %     figure,imshow(I);
    %     hold on;
    %     vl_plotframe(F([1 2 4 3]));
    
    %
    %     A = squeeze(mean(mean(I)));
    %     r = I(:,:,1);
    %     g = I(:,:,2);
    %     b = I(:,:,3);
    %     B = [var(r(:));std(g(:));std(b(:))];
    classData(k).feat = (double(D)/128);
    %         classData(k).feat = D;
    if (sum(D) == 0)
        %         k
        %         break;
    end
end
%%
%opts.train_ratio = .7;

opts.nTrain = 50;

selTrain = [];
for k = 1:length(uniqueClasses)
    f = find(classes == uniqueClasses(k));
    selTrain = [selTrain,...
        vl_colsubset(f,opts.nTrain)];
end

selTest = setdiff(1:length(classes),selTrain);
%%
trainData = classData(selTrain)';
train_feats = double([trainData.feat])';
train_labels = double([trainData.class]');

testData = classData(selTest);
test_feats = double([testData.feat]');
test_labels = double([testData.class]');

%%

choice = 'SVM'

if (strcmp(choice,'SVM'))
    
    %model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');
    model = grid_search(train_labels,train_feats);
    
    [test_res, accuracy, decision_values] = svmpredict(test_labels, test_feats, model);
    
    
elseif (strcmp(choice,'kdtree'))
    %%
    % % train_feats = single(train_feats);
    % % train_labels = zeros(1,size(train_feats,2));
    % % for k = 1:length(trainData)
    % %     trainData(k).class_ = ones(1,size(trainData(k).feat,2))*trainData(k).class;
    % % end
    % % train_labels_ = [trainData.class_];
    %%
    forest = vl_kdtreebuild(train_feats,'NumTrees',4);
    
    % end1
    test_res = zeros(size(selTest));
    % tic
    for k = 1:length(selTest)
        k
        %     toc
        current_feat = single(testData(k).feat);
        [index, dist] = vl_kdtreequery(forest, train_feats, current_feat,...
            'maxnumcomparisons',15);
        test_res(k) = mode(train_labels_(index));
        if (isnan(test_res(k)))
            k
            break
        end
    end
    
    
end
%%
idx = sub2ind([length(uniqueClasses), length(uniqueClasses)], ...
    test_labels, test_res) ;
confus = zeros(length(uniqueClasses)) ;
confus = vl_binsum(confus, ones(size(idx)), idx);

% plotConfMat(confus);forest
figure,imagesc(confus)
confus = confus./repmat(sum(confus,2),1,size(confus,2));





