
% if (0)
% drinking grammar - learn a set of parts and a grammar between them.

% first, annotate the data

doAnnotations = false;
if (doAnnotations)
    annotateDatabase;
end

initpath;
config;
conf.get_full_image = true;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
groundTruth = alignGT(conf,groundTruth);

learnWithShape(conf,groundTruth,'cup');

partNames = partNames(1:4); % don't include "face"
faceGT = getFaceGT(conf);

% partModelsDPM = learnModelsDPM(conf,train_ids,train_labels,groundTruth,partNames);
% partNames = partNames(1:4); % currently do only these two.

% currentID = 'drinking_001.jpg';
% [regions,regionOvp,G] = getRegions(conf,currentID,false);
%
% I = getImage(conf,currentID);
% % % displayRegions(I,regions,cellfun(@(x) -nnz(x) ,regions));
% % %
% bfe = BOWFeatureExtractor(conf,conf.featConf(1:2));
% profile on;
% x = bfe.extractFeatures(currentID);
% profile viewer

% R = RelativeLayoutFeatureExtractor(conf);
% [ii jj] = find(G);
% R.extractFeatures(currentID,regions,[ii jj]);
% partNames{end+1} = 'face';
initUnaryModels;

% faceModel = learnFaceModel_new(conf);
initUnaryModels2;
if (0)
    groundTruth = [groundTruth,faceGT];
    groundTruth = fixGroundTruthOrder(groundTruth);
    initBinaryModels;
    % partNames = {'face'};     partModels = faceModel;
    faceModelDPM = learnFaceModelDPM(conf);
    % learn
    partModelsDPM = learnModelsDPM(conf,train_ids,train_labels,groundTruth,partNames);
    visualizemodel(partModelsDPM{2})
end
%%

% partModels = learnModels(conf,train_ids,train_labels,groundTruth,partNames(3));

[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
drinking_test = test_ids(test_labels);
% drinking_test = test_ids;
%%
% % drinking_test = drinking_test(randperm(length(drinking_test)));
% partModels = [partModels,faceModel];
% partNames = {partNames{:},'face'};
% %
% % train_image_ids = imageData.train.imageIDs;
% % [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% % drinking_test = train_ids(train_labels);
% % % find only the images which do not appear as faces in the train set...
% %
% % isTraining = false(size(drinking_test));
% % for k = 1:length(isTraining)
% %     isTraining(k)=any(cellfun(@any,strfind(imageData.train.imageIDs,drinking_test{k})));
% % end
%%

% inspectDPMResults




% n
base=uint8(1-logical(imread('chars.bmp')));
addpath(genpath(fullfile('~/code/3rdparty/SelectiveSearchPcode')));
addpath('/home/amirro/code/fragments/boxes');
selectiveSearch = false;
doJump = false;
imageData = initImageData;
    imageSet = imageData.test;

% for k = 1
    
for k = 1:length(drinking_test)
    %         if (~isTraining(k))
%     %             continue;
    %         end
    
    clf;
    clc;
    disp(num2str(k));
%     currentID = imageSet.imageIDs{505};
    currentID = drinking_test{k};
    disp(currentID);
    if (~any(strfind(currentID,'brushing')))
        %         continue
    end
    %     currentID
    %     currentID = 'drinking_227.jpg';
    I = getImage(conf,currentID);
    %     [regions,rOvp] = getRegions(conf,currentID,true);
    
    if (doJump)
        [regions,sz,jump] = getRegionGrid(I,[]);
        sz
        jump        
    else        
        [regions,regionOvp,G] = getRegions(conf,currentID,false);        
    end
    
    if (selectiveSearch)
        L = load(fullfile('~/storage/boxes_s40',strrep(currentID,'.jpg','.mat')));
        regions = double(L.boxes(:,[1 2 3 4]));
        regions = mat2cell2(regions,[size(regions,1) 1]);
    end

    L_feats = load(fullfile('~/storage/bow_s40_feats/',strrep(currentID,'.jpg','.mat')));
    feats = (vl_homkermap(L_feats.feats, 1, 'kchi2', 'gamma', 1));
    
    rs = zeros(length(partModels),size(feats,2));
    for iPart = 1:length(partModels)
        [res_pred, res] = partModels(iPart).models.test(feats);
        rs(iPart,:) = row(res);
    end
    rs(isnan(rs)) = -inf;
    %     ovp = regionsOverlap(regions)
    
    B = load(fullfile('~/storage/objectness_s40',strrep(currentID,'.jpg','.mat')));
    propsFile = fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat'));
    load(propsFile);
    segmentBoxes = cat(1,props.BoundingBox);
    segmentBoxes = imrect2rect(segmentBoxes);
    
    ovp=  boxesOverlap(segmentBoxes,B.boxes);
    ovp_seg_obj = boxesOverlap(segmentBoxes,B.boxes);
    weights = sum(ovp_seg_obj,2);
    objectnessScores = row(ovp_seg_obj*B.boxes(:,5));
%     scores = scores./weights;

    rs = bsxfun(@plus,rs,.25*objectnessScores);
    subsets = suppresRegions(regionOvp,.5,rs,I,regions);
    ss = {};
    for q = 1:length(subsets)
        ss{q} = subsets{q}(1:5);
    end
    ss = [ss{:}];
    regions(ss) = fillRegionGaps(regions(ss));
    if (size(regions{1},2)==4)
        clf,
        for iPart = 1:length(partNames)
            %subplot(zz,zz,iPart);
            windows = cat(1,regions{:});
            zz = computeHeatMap(I,[windows rs(iPart,:)'],'max');
            zz(isinf(zz)) = min(zz(~isinf(zz(:))));
            zz1 = normalise(zz).^2;
            
            subplot(2,length(partNames),iPart);imagesc(repmat(zz1,[1 1 3]).*I);axis image;
            title(partNames{iPart});
            subplot(2,length(partNames),iPart+length(partNames)); imagesc(zz); colormap hot; axis image;
            colorbar;
        end
        pause;
        
        %sc(cat(3,I,zz),'prob');
    else
        zz = ceil(sqrt(size(rs,1)));
        toQuit = false;
        for q = 1:5
            if (toQuit)
                break;
            end
            clf;
            bbbb = {};
            
            
            displayRegions(I,regions(subsets{iPart}),rs)
            
            for iPart = 1:length(partModels)
                if (toQuit)
                    break;
                end
                subplot(zz,zz,iPart);
                alpha_ = .8;
                
                
                
                curRegion = regions{subsets{iPart}(q)};
                %curZ =rendertext(curZ,num2str(k),[250 250 210], [1, 1],base);
                bbb=blendRegion(I,curRegion,.5);
                bbb = max(bbb,0);
                bbb = min(bbb,1);
                tt = [partNames{iPart} ',' num2str(q) ':',...
                    num2str(rs(iPart,subsets{iPart}(q)))];
%                 bbb = rendertext(im2uint8(bbb),tt,[250 250 210], [1, 1],base);
                imagesc(bbb); axis image;
                bbbb{iPart} = bbb;
                %sc();
                %                 displayRegions(I,{curRegion},0,-1);
                %                 imagesc(alpha_*I.*repmat(curRegion,[1 1 3]) + ...
                %                     (1-alpha_)*I.*repmat(~curRegion,[1 1 3]));
%                 axis image;
                title([num2str(k) ' ' partNames{iPart} ', ' num2str(q) ' : ',...
                    num2str(rs(iPart,subsets{iPart}(q)))]);
            end
            disp('(c)ontinue,(s)ave,(q)uit');
            ch = getkey;
            
            if (ch == 'q')
                disp('quitting...');
                toQuit = true;
            elseif (ch == 's')
                fileName = fullfile('~/notes/images/segment_chains/samples/',...
                    sprintf('%s_%d_%d.jpg',strrep(currentID,'.jpg',''),iPart,q));
                imwrite(multiImage(bbbb),fileName);
            end
            
            
            %a = input('(c)ontinue,(s)ave,(q)uit > ','s');
            %             pause;
        end
        if (toQuit)
            break;
        end
    end
    clf;
    %     [r,ir] = sort(res,'descend');
    %     displayRegions(I,regions(ir(1:5)),r(1:5));
end

%%
% currentID = train_ids{1};
% [regions,rOvp,G] = getRegions(conf,currentID);
% [ii,jj] = find(G);
% pairs = [ii(2000) jj(2000)];
% imshow(regions{jj(2000)})


