function applyModel(globalOpts,model,test_images,iter)
% apply the learned model to all the images in test_set and
% save the results to files.

resPath = getResultsPath(globalOpts,iter);
if (~exist(resPath,'dir'))
    mkdir(resPath);
end

nClasses = length(globalOpts.VOCopts.classes);
ids = test_images;

tic;
image_subset = 1:length(ids);
% scales = [2 4 6 8 10];
scales = [4];
% find out the size of a feature vector.
aibs = struct('map',{});
for iScale = 1:length(scales)
    aib_path = fullfile(globalOpts.expPath,...
        ['aibmap_' num2str(scales(iScale)) '.mat']);
    load(aib_path);
    [cut,map,short] = vl_aibcut(parents,globalOpts.aib_cut);
    aibs(iScale).map = map;
end



% train_data = prepare_training_data(globalOpts,iter);

for i=1:length(image_subset)
    % display progress
    if toc>1
        fprintf('test: %d/%d\n',i,length(image_subset));
        drawnow;
        tic;
    end
    
    currentImage = image_subset(i);
    currentID = ids{currentImage};
    im = imread(getImageFile(globalOpts,currentID));
    fPath = fullfile(resPath, [currentID '.txt']);
    
    if (exist(fPath,'file'))
        if (~globalOpts.debug)
            continue;
        end
    end
    
    clear psix;
    clear psix_test;
    
    boxesFileName = getBoxesFile(globalOpts, currentID);
    if (~exist(boxesFileName,'file'))
        error(['get_bbox_data ---> boxes for image' currentID ' don''t exist']);
    end
    
    boxes = [];
    load(boxesFileName); % got bboxes & F
    % remove bbobxes whose aspect ratio is terrible, or are too small
    bads = bad_bboxes(boxes,globalOpts);
    %bads = bad_bboxes(boxes);
    boxes = boxes(~bads,:);
    
    %     boxes = boxes(1,:);%TODO, this is a bug, retain all boxes.
    
    s = 1;
    
    nBoxes = size(boxes,1); %#ok<*NODEF>
    featPath = getFeatFile(globalOpts,[currentID '_' num2str(s)]);
%     if (~exist(featPath,'file'))
        
        
        features = [];
        for iScale = 1:length(scales)
            globalOpts.map = aibs(iScale).map;
            globalOpts.scale_choice = scales(iScale);
            features = [features;get_box_features2(globalOpts,boxes,{currentID},model,s)];
        end
%         save(featPath,'features');
%     else
%         load(featPath)
%     end
    
    
    psix_test = globalOpts.hkmfun(features);
    %         K = sparse(double(psix_test'*train_data));
    %         K = double(K);
    scores  = -1000*ones(nClasses,size(features,2));
    %         for iClass = 1:length(globalOpts.class_subset)
    %             cls = globalOpts.class_subset(iClass);
    %             currentModel = model(cls).model;
    %             [predicted_label, accuracy, decision_values] = ...
    %                 svmpredict(ones(size(K,1),1),  [(1:size(K,1))' K], currentModel);
    %
    %             decision_values(bad_bboxes(boxes*scales,globalOpts)) = -1000;
    %
    %             scores(iClass,:) = decision_values;
    %         end
    scores = model.w' * psix_test + (model.b' * ones(1,size(psix_test,2)));
    scores(isnan(scores)) = -1000;
    mscores = scores;
    
    [mscores,imm] = sort(scores,2,'descend');
    % print the scores in sorted order.
    
    if (~globalOpts.debug)
        fid = fopen(fPath,'a');
        for row = 1:nClasses
            for c = 1:min(100,nBoxes) % TODO!!!!
                fprintf(fid,'%s %f %f %f %f %f %d\n',currentID,mscores(row,c),boxes(imm(row,c),[2 1 4 3]),row);
            end
        end
        fclose(fid);
    end
end


function train_data = prepare_training_data(globalOpts,iter)

instancesPath = fullfile(globalOpts.expPath,'instances0.mat');
load(instancesPath);

nPosSamples = size(trainingInstances.posFeatureVecs,2);
nNegSamples = size(trainingInstances.negFeatureVecs,2);
if (iter > 1)
    
    negsPath = fullfile(globalOpts.expPath,sprintf('negs_%03.0f.mat',iter-1));
    load(negsPath);
    
    cls = 1; %TODO - this is a bug, only for testing aeroplanes.
    % check it later...
    trainingInstances.negFeatureVecs = [trainingInstances.negFeatureVecs,hard_negs{cls}];
    nNegSamples = size(trainingInstances.negFeatureVecs,2);
end

trainingInstances.posFeatureVecs = repmat(trainingInstances.posFeatureVecs,...
    1,round(nNegSamples/nPosSamples));


nPosSamples = size(trainingInstances.posFeatureVecs,2);


W = [trainingInstances.posFeatureVecs,...
    trainingInstances.negFeatureVecs];
A = sum(W);
A = (isnan(A));
W = W(:,~A);
%train_data = hkm(full(W));
train_data = hkm(W);
% train_data = [];
% for p = 1:1000:size(W,2)
%     p
%     s_start = p;
%     s_end = min(p+999,size(W,2));
%     train_data = [train_data,sparse(hkm(full(W(:,s_start:s_end))))];
% end
