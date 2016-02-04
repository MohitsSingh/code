
initpath;
addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));
%%
% load m1_train
% load qq1

%%

%% detect faces + landmark using deva ramanan's code.
addpath('/home/amirro/code/3rdparty/face-release1.0-basic');

conf.max_image_size = 256;
imshow(getImage(conf,train_ids{qq1(k).cluster_locs(1,11)}));

f_true = find(train_true_labels);

landmarks = detect_landmarks(m1_train);

im = (getImage(conf,train_ids{qq1(k).cluster_locs(f_true(5),11)}));


figure,imshow(multiImage(m1_train_true))


debug_ = true;
% sel_ = f_gt;
ZZZ= {};
m_to_try = m1_train_true;
% sel_ = 1:length(m);
sel_ = 1:length(m_to_try);
sel_ = iss;
% sel_ = 1:1000;
sel_ = [5:7 10 13:14 17:21 25:27 59 61];
% figure,imshow(multiImage(aaa,false))



mouthImages = multiCrop(m_to_try(sel_),mouthRect);
mouthImages = multiCrop(m1_train_true,mouthRect);

figure,imshow(multiImage(mouthImages,true));

[mm,~,x,y] = multiImage(mouthImages ,false);
mm = im2double(rgb2gray(mm));
E = edge(mm,'canny',0,.1);
figure,imagesc(E);
figure,imagesc(mm);colormap gray;


mouthImages_false = multiCrop(m1_train_false,mouthRect);
figure,imshow(multiImage(mouthImages_false,false));

[mm,~,x,y] = multiImage(mouthImages ,false);
x = x';
y = y';

figure,imagesc(mm);
hold on;

for kk = 1:length(mouthImages_false)
    text(x(kk),5+y(kk),num2str(kk),'color','y','FontSize',10);
end


x1 = imageSetFeatures2(conf,mouthImages_false(1:50));
x1 = cat(2,x1{:});


lipsRetrained(1)

mouthImagesTest = multiCrop(m_test,mouthRect);
% mouthImagesTest = multiCrop(m_test,mouthRect);
x3 = imageSetFeatures2(conf,mouthImagesTest);
x3 = cat(2,x3{:});
D = l2(x3',x1');


%%
im1 = m1_train_true{1};
figure,imagesc(im1)
[Z0,w,b] = segmentFace_stochastic(im1,3);
figure,imagesc(Z0)

%%

% [d,id] = mean(D(:,1:5),[],2);
%%
% d = mean(D(:,1:5).^.5,2);
[d_lips,id] = min(D.^.5,[],2);
% d_face = qq1_test(k).cluster_locs(:,12);
% choice_vec = ones(size(d));
% top_choice = 1000;
% choice_vec(top_choice+1:end) = 0;
% d_face = zeros(size(d_face));
% d_face(1:1000) = 1;
% d_total =  (d+.1*d_face).*choice_vec;
d_total = d_lips;
[r,ir] = sort(d_total,'descend');

% gt_labels = test_labels(qq1_test(k).cluster_locs(:,11));
% figure,imshow(multiImage(mouthImagesTest(gt_labels),false))
figure,imshow(multiImage(mouthImagesTest(ir(1:100)),false))

%%
figure,imshow(multiImage(m1_train(ir(1:100)),false));
%%

%%


Z = createConsistencyMaps(qLips_test(4),[80 80],[],1000,[5 3]);
figure,imshow(m_test{1})
figure,imshow(jettify(Z{1}));


imwrite(multiImage(m1_train_false,false,false),'false_drinkers.jpg')
% imshow(im);

circles = fitCircles(mouthImages_false{1});

plotCircles(circles);

im = m_to_try{sel_(7)};

figure,imshow(multiImage(m_to_try(sel_)))

[C,A] = vl_kmeans(allCircles,100);
rr = allCircles(3,:)< 15 & allCircles(1,:) > 0  & allCircles(2,:) > 0;
figure,plot(allCircles(1,rr),allCircles(2,rr),'r+');


figure,imagesc(C);
figure,hist(double(A))
aa = find(A==65);
figure,imshow(im);


figure,hist(double(A))
 
legals = circles(:,3) < 15;
circles = circles(legals,:);
mses = mses(legals);

[r,ir] = sort(mses,'ascend');
f = find(legals);

figure,imshow(im);
hold on;
for ii = 1:length(mses);           
        circle(circles(ir(ii),1:2),circles(ir(ii),3),32,[0 1 0])
end
 

trueMouthImages = multiImage(mouthImages,false);
figure,imshow(trueMouthImages)
E = {}
for kk= 1:length(mouthImages)
E{kk} = edge(im2double(rgb2gray(mouthImages{kk})),'canny');
end
figure,imshow(multiImage(E,false))
E1 = {}
for kk= 1:length(mouthImages_false)
E1{kk} = edge(im2double(rgb2gray(mouthImages_false{kk})),'canny');
end

imwrite(multiImage(E1,false),'non-drinking-edges.tif')

figure,imshow(multiImage(E1(1:100),false));
E = edge(im2double(rgb2gray(mouthImages_false{1})),'canny');


EE = zeros(size(E1{1}));
for kk = 1:length(EE)
    EE = EE+1-bwdist(E1{kk});
end
figure,imagesc(exp(EE/10))

figure,imshow(multiImage(mouthImages_false(1:16),false));

imwrite(multiImage(mouthImages_false ,false),'non-drinking-mouths.tif')


%%
% sel_ = theSel
d = zeros(size(sel_));
%mouthRect = [13 50 42 70];
mouthRect = [13 50 42 65];
mouthSector = [1 50 42 80];
viss = {};
sel_ = iss;
% for ikk = 10
debug_ = true;
for ikk = 1:length(sel_)
    %        for  ikk =1
    % show d, mouth detection score, face detection score...
    kk = sel_(ikk)
    im = m_to_try{kk};
    if (debug_)
        close all;
    end
    %[segImage labels map gaps E] = vl_quickseg(im, .5,2, 10);
    
    %im = imresize(im,2);
    [segImage labels map gaps E] = vl_quickseg(im, .5,2,10);
    try
        [faceSeg_1,w,b] = segmentFace_stochastic2((im),3);
    catch ME
        continue; % this probably means fitting a gaussian to the
        % color distribution was ill posed. This shouldn't happen in face
        % images.
    end
    %     continue;
    
    
    faceBW = faceSeg_1 > .5;
    faceLabels =  bwlabel(faceBW);
    rprops_face = regionprops(faceLabels,'Area','ConvexImage','FilledImage','PixelIdxList','BoundingBox');
    faceBlobAreas = [rprops_face.Area];
    if (debug_)
        figure,imagesc(faceLabels)
    end
    [maxFaceSize,maxFaceBlob] = max(faceBlobAreas)
    faceBlob = rprops_face(maxFaceBlob);
    faceBW = zeros(size(faceBW));
    faceBW(faceBlob.PixelIdxList) = 1;
    
    faceBW = imclose(imopen(faceBW,ones(3)),ones(3));
    rprops_face = regionprops(faceBW,'Area','ConvexImage','FilledImage','PixelIdxList','BoundingBox');
    faceBlobAreas = [rprops_face.Area];
    [maxFaceSize,maxFaceBlob] = max(faceBlobAreas)
    if (isempty(maxFaceBlob))
        continue;
    end
    faceBlob = rprops_face(maxFaceBlob);    
    
    faceOnlyImage = zeros(size(faceLabels));
    fbb = rprops_face(maxFaceBlob).BoundingBox;
    fbb = round(fbb);
        
    face_conv_def = faceBlob.ConvexImage & ~faceBlob.FilledImage;
    faceOnlyImage(fbb(2):fbb(2)+fbb(4)-1,fbb(1):fbb(1)+fbb(3)-1) =face_conv_def;
    
    %     figure,imshow(faceOnlyImage);
    faceOnlyImage = imerode(faceOnlyImage,ones(3));
    if (debug_)
        figure,imshow(faceOnlyImage);
    end
    
    
    % find which segment of low face probability is part of a large
    % convexity defect in the face...
    
    %segs_in_face = unique(labels.*faceOnlyImage);
    labels = labels.*faceOnlyImage;
    labels = RemapLabels(labels)-1; % ignore background
    %     segs_in_face(segs_in_face==0) = [] ;
    
    rprops=  regionprops(labels,faceSeg_1, 'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
        'MinorAxisLength','MeanIntensity','Solidity');
    
    if (isempty(rprops)) % no convex deficiencies found in face.
        continue;
    end
    
    
    sol = [rprops.Solidity]';
    [segImage,c] = paintSeg(im,labels);
    meanColors = im2double(uint8([[c{1}.MeanIntensity];[c{2}.MeanIntensity];[c{3}.MeanIntensity]]'));
    colorProbabilities = sigmoid(b+meanColors*w);
    probImage = paintRegionProps(labels,rprops,colorProbabilities);
    
    segmentBoxes= cat(1,rprops.BoundingBox);
    segmentBoxes(:,3) = segmentBoxes(:,3)+segmentBoxes(:,1);
    segmentBoxes(:,4) = segmentBoxes(:,4)+segmentBoxes(:,2);
    
    [~,~,boxes_i] = BoxSize(BoxIntersection(segmentBoxes,mouthRect));
    [~,~,boxes_i_sector] = BoxSize(BoxIntersection(segmentBoxes,mouthSector));
    [~,~,boxes_s] = BoxSize(segmentBoxes);
    insideBoxes = boxes_i./boxes_s >= .1 & boxes_i_sector./boxes_s >= .65;
    %     adjMatrix = boundarylen(double(labels),length(rprops));
    
    % now look for segments intesecting the mouth area with low
    % probability to be part of the face
    
    %     zz = zeros(size(insideBoxes));
    %     zz(segs_in_face) = 1;
    %     insideBoxes= insideBoxes & zz;
    
    insideProbs = colorProbabilities(insideBoxes);
    colorProbabilities1 = colorProbabilities;
    colorProbabilities1(find(~insideBoxes)) = 0;
    areas = col([rprops.Area]);
    probImage1 = paintRegionProps(labels,rprops,colorProbabilities1);
    areaImage1= paintRegionProps(labels,rprops,insideBoxes.*areas);
    
    
    if (~any(insideBoxes))
        continue;
    end
    
    insideProbs = areas(insideBoxes).*(1-insideProbs).*double(sol(insideBoxes)>=.85);
    %sol_ = sol(:).*insideBoxes;
    
    colorProbabilities1(insideBoxes) = insideProbs;
    probImage1 = paintRegionProps(labels,rprops,colorProbabilities1);
    
    
    if (debug_)
        figure;
        subplot(1,3,1),imagesc(probImage1); axis off; axis equal; title(num2str(max(probImage(:))));
        subplot(1,3,2),imagesc(im);axis off; axis equal
        
        subplot(1,3,3),imagesc(labels);axis off; axis equal
        hold on;
        plotBoxes2(mouthRect([2 1 4 3]),'color','g');
        plotBoxes2(segmentBoxes(insideBoxes,[2 1 4 3]));
        plotBoxes2(mouthSector([2 1 4 3]),'color','m');
        
        
        pause;
    end
    %insideProbs = insideProbs(areas(insideBoxes)>=200);
    if (any(insideProbs))
        [maxProb,iMaxProb] = max(insideProbs);
        
        d(ikk) = max(insideProbs);
    end
    
    % %     if (debug_)
    % %         figure,imagesc(labels);
    % %         hold on;plot(topPoints(cc,1),topPoints(cc,2),'r+');
    % %
    % %         %
    % %         subplot(1,3,1);
    % %         imshow(im);title('img');
    % %
    % %         subplot(1,3,2);
    % %         imshow(segImage);title('seg');
    % %
    % %
    % %         viss{ikk} = [im segImage Z_cc];
    % %         %         subplot(3,2,6);
    % %         %         Z_svm = paintRegionProps(labels,rprops,decision_values);
    % %         %         imagesc(Z_svm);colorbar;
    % %         pause;
    % %     end
end

%
%%
%d_face = qLips_test(4).cluster_locs(:,12);
d_face = qq1_test(k).cluster_locs(:,12);
alpha_d =1
%alpha_face =200
d_total = alpha_d*d(:).*double(d_face>-.1).*d_face; % **GOOD
%!!!
d_total = alpha_d*d(:).*double(d_face>0.1);% **GOOD

% d_total = alpha_d*d(:).*double(d_face>-.1).*d_face+20*d_lips; % **GOOD
% d_total = alpha_d*d(:).*(1+d_face).^4;
% .*double(d_face(sel_) >= 0);
% d_total(isnan(d_total)) = -1000;

[ss,iss] = sort(d_total,'descend');
% save d d iss
subSet = iss(1:50)'
displaySet = m_to_try(sel_(subSet));
figure;imshow(multiImage(displaySet,false,false));


% imwrite(multiImage(displaySet,false,false),'face_perturb.tif')

%%

d(isinf(d)) = 1000;
alpha_d = 1;

d_total = d(:).*d_face;


[ss,iss] = sort(d_total,'descend');
% displaySet = m_to_try;
displaySet = m_to_try(sel_(iss(1:50)));
figure;imshow(multiImage(displaySet,false,false));

%%

d_mouth = qLips_test(4).cluster_locs(:,12);

d_face = qLips_test(4).cluster_locs(qLips_test(4).cluster_locs(:,11),12);
% alpha_d = 1;
% alpha_face =.3
% alpha_mouth =1;
alpha_d = 1
alpha_face =.1
alpha_mouth =0

d_total = alpha_d*d(:)+alpha_mouth*d_mouth+alpha_face*d_face;
[ss,iss] = sort(d_total,'descend');
%displaySet = lipImages(iss(ss>0));
displaySet = lipImages(iss(1:50))
figure(2000);imshow(multiImage(displaySet,false,false));



%%

%%
imwrite(multiImage(displaySet,false,false),'face_.1_drink.jpg');

%
%
% %%
%
% vvv = viss([1:3 5:9 12:22]);
%
%
% imshow(multiImage(vvv,true))
%
% explanations = repmat('+',1,length(vvv));
% explanations(1) = 'f';
% explanations([4 7 11 16 17 19]) = 's';
% explanations([6 12 13 15 18]) = 'a';
% explanations([14]) = '?';
% explanations([9]) = 's';
% imshow(multiImage(vvv,explanations));
%
% imwrite(multiImage(vvv,explanations),'straws_explained.tif');
%
% imshow(cat(1,vvv{:}))
