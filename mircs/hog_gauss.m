function hog_gauss_data = hog_gauss()
if (exist('hog_gauss.mat','file'))
    load('hog_gauss.mat');
    return;
end

hog_gauss_data = [];
initpath;
config;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');

n = 0;
sampleImages = train_ids(1:2:end);
conf.get_full_image = true;
sampleMean = [];
for k = 1:length(sampleImages)
    currentID = sampleImages{k};
    I = getImage(conf,currentID);
    [X,uus,vvs,scales,t ] = allFeatures( conf,I,1 );
    n = n+size(X,2);
    X = sum(X,2);
    if (k==1)
        sampleMean = X;
    else
        sampleMean = sampleMean + X;
    end
    k
end
sampleMean = sampleMean/n;

% calculate the covariance matrix....
n = 0;
covMat = zeros(size(sampleMean,1));
for k = 1:length(sampleImages)
    currentID = sampleImages{k};
    I = getImage(conf,currentID);
    [X,uus,vvs,scales,t ] = allFeatures( conf,I,1 );
    n = n+size(X,2);    
    t = bsxfun(@minus,X,sampleMean);
    covMat = covMat + t*t';
    k
end
covMat = covMat/n;

hog_gauss_data.sampleMean = sampleMean;
hog_gauss_data.covMat = covMat;
[V,D] = eig(covMat);
d = diag(D);
hog_gauss_data.V = V;
hog_gauss_data.d = d;
save hog_gauss hog_gauss_data


[V,D] = eig(covMat);

d = diag(D);

tt = randperm(length(train_ids));
I = getImage(conf,train_ids{tt(1)});
[X,uus,vvs,scales,t ] = allFeatures( conf,I,1 );

ii = randperm(size(X,2));
bboxes = uv2boxes(conf,uus,vvs,scales,t);

%%
for q = 1:length(ii)
    k = ii(q);
    clf;
    subplot(2,2,1);
    imagesc(I); axis image; hold on;
    bb = bboxes(k,:);
    plotBoxes2(bb([2 1 4 3]),'g','LineWidth',2);
    subplot(2,2,2);
    bb = round(bb);
    
    imagesc(I(bb(2):bb(4),bb(1):bb(3),:));axis image;
    x = X(:,k);
    v1 = showHOG(conf,x.^2);
    subplot(2,2,3); 
    imagesc(v1); axis image; title('orig');
    x_1 = x-sampleMean;
    lambda_ = 1;
    x_1 = (d./(d.^2+lambda_.^2)).*x_1;
    v1 = showHOG(conf,x_1);
    subplot(2,2,4); 
    imagesc(v1); axis image; title('whitened');
    pause;
end

end