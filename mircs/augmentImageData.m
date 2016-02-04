% define an augmented dataset, where we focus only on the area around the
% detected face.

function newImageData = augmentImageData(conf,newImageData)
% ,addSubImages,...
%     addUpperBodies)

%
% if (addSubImages)
%     subImagesPath = '~/storage/misc/subImages.mat';
%     if exist(subImagesPath,'file')
%         load subImagesPath;
%     else
%
%     end
% else
%     subImages = cell(size(newImageData));
% end
% if (addUpperBodies)
%     upperBodiesPath = '~/storage/misc/upperBodies.mat';
%     if exist(upperBodiesPath,'file')
%         load upperBodiesPath;
%     else
%
%     end
% else
%     subImages = cell(size(newImageData));
% end

resPath = '~/storage/misc/imageData_new_sub.mat';
if (exist(resPath,'file'))
    load(resPath)
else
    
    conf.max_image_size = inf;
    conf.get_full_image = true;
    imageIDS = {newImageData.imageID};
    for k = 1:length(imageIDS)
        100*k/length(imageIDS)
        currentID = imageIDS{k};
        [I,I_rect] = getImage(conf,currentID);
        M = getSubImage(conf,newImageData,currentID);
        newImageData(k).sub_image = im2uint8(M);
        newImageData.I_rect = I_rect;
    end
    for k = 1:length(imageIDS)
        newImageData(k).isTrain = k <= 4000;
    end
    
    for k = 1:length(newImageData) % lip scores for this face
        k
        L_lips = load(j2m('~/storage/s40_lip_detection',newImageData(k)));
%                 if (~isempty(L_lips.bbs))
%                 'dhgdfg'
%                 break;
%             end
        newImageData(k).lipBoxes = L_lips.bbs;
    end
    
    
    for k = 1:length(newImageData)
        k
        piotr_landmarks_path = j2m(conf.landmarks_piotrDir,newImageData(k));
        if (exist(piotr_landmarks_path,'file'))
            load(piotr_landmarks_path);
%             break;        
        xy = reshape(res.xy(1:58),[],2);
        xy = bsxfun(@plus,xy,res.bbox(1:2));
        newImageData(k).faceLandmarks_piotr.poly = xy;
        newImageData(k).faceLandmarks_piotr.occ = res.xy(59:end);
        %         I = getImage(conf,newImageData(k).imageID);
        %         clf; imagesc2(I);
        %         hold on;
        %         plotPolygons(xy,'g+');pause;
        end
    end
    
    %%
    newImageData(k).upperBodyDets = getUpperBodyDets(conf,imageID);
    
    for k = 1:length(newImageData)
        100*k/length(newImageData)       
        [I,I_rect] = getImage(conf,newImageData(k));
        newImageData(k).I_rect = I_rect;
        newImageData(k).size = size2(I);
    end
    
    %%
    
    m = readDrinkingAnnotationFile('train_data_to_read.csv');
    newImageData = augmentGT(newImageData,m);
    
    %% annotate missing faces...
    annotateMissingFaces;
    
    save(resPath,'newImageData');
end