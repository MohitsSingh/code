function [patchModels,w] = learnPatchModels(param,negImagePaths)
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
imgSize = param.imgSize;
windowSize = param.windowSize;
nPts = param.nPts;
cellSize = param.cellSize;
all_labels = cat(3,param.labels{:});

debug_ = 1;
nNegSamples = round(3000/debug_);
nNegImages = round(100/debug_);
nSamplesPerImage = round(nNegSamples/nNegImages);
neg_samples = samplePatches(negImagePaths,nNegImages,nSamplesPerImage,windowSize,cellSize);

patchModels = cell(nPts,1);

for iPt = 1:nPts
    u = squeeze(all_labels(iPt,:,:))';    
    isVisible = ~any(isnan(u),2);
    u = u(isVisible,1:2);
    u = u*imgSize;
    sub_rect = [u u];
    sub_rect = inflatebbox(sub_rect,windowSize,'both',true);
    subImgs = multiCrop([],param.patterns(isVisible),round(sub_rect));
    X_pos = getImageStackHOG(subImgs,windowSize,true,false,cellSize);
    
    nPos = size(X_pos,2);
    nNeg = size(neg_samples,2);
    labels = ones(nPos+nNeg,1);
    labels(nPos+1:end) = -1;
    X = sparse(double([X_pos neg_samples]));    
    patchModels{iPt} = train(labels,X,'-s 2 -c .1 -B -1','col');
    
%     dummy_ = zeros(size(X,2),1);
%     [predicted_label, accuracy, decision_values] = predict(dummy_, X, model,'-b 1','col')
    
%     aa = exp(decision_values)./(1+exp(decision_values))
%     figure,plot(aa)%
%     predict(X)
%     for z = 1:length(parm.patterns)
%         clf; imagesc2(parm.patterns{z});
%         plotPolygons(imgSize*u(:,z)','r.');
%         drawnow; pause
%     end
end


r = cellfun2(@(x)x.Label,patchModels);r = cat(2,r{:});
assert(all(r(1,:)==1));
w = cellfun2(@(x)x.w,patchModels); w = cat(1,w{:});

function res = samplePatches(imgPaths,nImages,nPerImage,windowSize,cellSize)    
    imgPaths = vl_colsubset(row(imgPaths),nImages,'Uniform');
    res = {};
    for t = 1:length(imgPaths)
        I = im2single(imread(imgPaths{t}));
        I = imResample(I,[100 100],'bilinear');
        [X,boxes] = all_hog_features_single_scale(windowSize, cellSize,I);
        res{end+1} = vl_colsubset(X,nPerImage,'Random');
    end
    res = cat(2,res{:});

