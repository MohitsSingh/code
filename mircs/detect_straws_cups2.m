
debug_ = true;
% sel_ = f_gt;
ZZZ= {};
m_to_try = lipImages_train;
sel_ = 1:length(m_to_try);
d = zeros(size(sel_));

% sel_ = 1:length(m_to_try);

% sel_ = theSel

viss = {};

for ikk = 1:length(sel_)
    % show d, mouth detection score, face detection score...
    kk = sel_(ikk);
    im = m_to_try{kk};
    if (debug_)
        close all;
    end
    %[segImage labels map gaps E] = vl_quickseg(im, .5,2, 10);
    
    %im = imresize(im,2);
    [segImage labels map gaps E] = vl_quickseg(im, .5,2,10);
    %     segImage = allSegs{kk};
    %     labels = allLabels{kk};
    
    %    edgeim = edge(im2double(rgb2gray(im)),'canny');
    [edgelist, labelededgeim] = edgelink(edgeim, 10);
    tol = 4;
    seglist = lineseg(edgelist, tol);
    
    %startRect = [15 30 54 45];
    %startRect = [47 49 120 83];
    startRect = [29 34 55 46];
    endRect = [1,size(im,1)-14,size(im,2),size(im,2)];
    rprops=  regionprops(labels,'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
        'MinorAxisLength');
     figure,imagesc(paintRegionProps(labels,rprops,[rprops.Eccentricity]));
     figure,imagesc(paintRegionProps(labels,rprops,[rprops.Orientation]));
     figure,imagesc(paintRegionProps(labels,rprops,[rprops.Area]));
    % expect some form of vertical line at this point
    
    %     figure,imagesc(paintRegionProps(labels,rprops,[rprops.Area]));
    
    eccentricity = col([rprops.Eccentricity]);
    orientation = col([rprops.Orientation]);
    majorAxis = col([rprops.MajorAxisLength]);
    minAxis = col([rprops.MinorAxisLength]);
    areas = col([rprops.Area]);
    
    curFeats = [eccentricity orientation majorAxis minAxis areas];
       
    con1 =abs(orientation) >= 30;
    con2 = false;  
    con3 = eccentricity >= .85 & areas > 15;
    cc = con1 & (con2 | con3);
    eee = max(eccentricity(find(cc)));
    if (any(eee))
        d(ikk) = eee;
    end
    if (debug_)
        figure,imagesc(labels);
%         hold on;plot(topPoints(cc,1),topPoints(cc,2),'r+');
        
        %
        subplot(1,3,1);
        imshow(im);title('img');
        
        subplot(1,3,2);
        imshow(segImage);title('seg');
        %         subplot(3,2,3);
        %         imshow(edgeim); title('edges');
        %         hold on;
        %         drawedgelist(edgelist, size(im), 1, 'rand');
        %
        %         subplot(3,2,4);
        %         h = imshow(edgeim); title('edges');
        %         hold on;
        %         drawedgelist(seglist, size(im), 2, 'red');
        %
        subplot(1,3,3);
        Z_cc = paintRegionProps(labels,rprops,cc);
        imshow(Z_cc); title('good segments');
        
        im =  im2double(im);
        Z_cc = repmat(Z_cc,[1 1 3]);
        
        viss{ikk} = [im segImage Z_cc];
        %         subplot(3,2,6);
        %         Z_svm = paintRegionProps(labels,rprops,decision_values);
        %         imagesc(Z_svm);colorbar;
        pause;
    end
end

%

landmarks_train = detect_landmarks(conf,train_ids);

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
