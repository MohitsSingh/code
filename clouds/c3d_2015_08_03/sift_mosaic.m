function [p1,p2] = sift_mosaic(im1, im2,m1,m2,toShow,T,checkHomog,stackFeats,step)
% SIFT_MOSAIC Demonstrates matching two images using SIFT and RANSAC
%
%   SIFT_MOSAIC demonstrates matching two images based on SIFT
%   features and RANSAC and computing their mosaic.
%
%   SIFT_MOSAIC by itself runs the algorithm on two standard test
%   images. Use SIFT_MOSAIC(IM1,IM2) to compute the mosaic of two
%   custom images IM1 and IM2.

% AUTORIGHTS
if nargin < 6
    T = 1.5;
end
if nargin < 7
    checkHomog = false;
end
if nargin < 8
    stackFeats = false;
end
if nargin < 8
    step = 1;
end
if nargin == 0
    im1 = imread(fullfile(vl_root, 'data', 'river1.jpg')) ;
    im2 = imread(fullfile(vl_root, 'data', 'river2.jpg')) ;
end

% make single
im1 = im2single(im1) ;
im2 = im2single(im2) ;

% make grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1) ; else im1g = im1 ; end
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end

if nargin < 3
    m1 = true(size2(im1));
    m2 = true(size2(im2));
end
m1 = imdilate(m1,ones(3));
m2 = imdilate(m2,ones(3));

if nargin < 5
    toShow = false;
end
% --------------------------------------------------------------------
%                                                         SIFT matches
% --------------------------------------------------------------------

% step = 1;

% bounds1 = region2Box(m1);
% bounds1(1:2) = bounds1(1:2)-15;
% bounds1(3:4) = bounds1(3:4)+15;
% bounds2 = region2Box(m2);
% bounds2(1:2) = bounds2(1:2)-15;
% bounds2(3:4) = bounds2(3:4)+15;

% multiscale features
% [f1,d1] = vl_dsift(im1g,'Step',step,'Bounds',bounds1);
% [f2,d2] = vl_dsift(im2g,'Step',step,'Bounds',bounds2);
% 
sizes = 4;
[f1, d1] = vl_phow(im1g,'Step',step,'Fast',true);
[f2, d2] = vl_phow(im2g,'Step',step,'Fast',true);

assert(all(col(f1(1:2,:)-round(f1(1:2,:)))==0));
assert(all(col(f2(1:2,:)-round(f2(1:2,:)))==0));
f1_inds = sub2ind2(size2(im1g),round(f1([2 1],:)'));
f1_sel = m1(f1_inds);
f1 = f1(:,f1_sel);
d1 = d1(:,f1_sel);

f2_inds = sub2ind2(size2(im2g),round(f2([2 1],:)'));
f2_sel = m2(f2_inds);
f2 = f2(:,f2_sel);
d2 = d2(:,f2_sel);
if (stackFeats)
    [f1,d1] = stack_features(f1,d1);
    [f2,d2] = stack_features(f2,d2);
end




% x2(im1g); plotPolygons(f1','r.')
% x2(im2g); plotPolygons(f2','r.')
% get matches for each line independently!!!!

f1_y = floor(f1(2,:));
f2_y = floor(f2(2,:));

u_y = unique(f1_y);
if (1)
    all_matches = {};
    all_scores = {};
    for t = 1:length(u_y)
        %     error('finish this');
        sel_1 = find(f1_y==u_y(t));
        sel_2 = find(abs(f2_y-u_y(t))<2);
        if (isempty(sel_1) || isempty(sel_2))
            continue
        end
        %     zzzzz=0;
        %curMatches = vl_ubcmatch(d1(:,sel_1),d2(:,sel_2),T);
        
        curMatches = best_bodies_match(d1(:,sel_1)',d2(:,sel_2)');
%         curMatches = knn_match(d1(:,sel_1)',d2(:,sel_2)');
        curMatches = [sel_1(curMatches(1,:));sel_2(curMatches(2,:))];
        all_matches{t} = curMatches;
        %     x2(im1g); plotPolygons(f1(:,sel_1)','r.');x2(im2g); plotPolygons(f2(:,sel_2)','r.');
        %     clf;match_plot_h(im1g,im2g,f1(:,sel_1(curMatches(1,:)))',f2(:,sel_2(curMatches(2,:)))')
    end
    
    matches = cat(2,all_matches{:});
else
    [matches, scores] = vl_ubcmatch(d1,d2) ;
end

numMatches = size(matches,2) ;

X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;
if ~checkHomog
    p1 = f1(:,matches(1,:))';
    p2 = f2(:,matches(2,:))';
    return
end

% --------------------------------------------------------------------
%                                         RANSAC with homography model
% --------------------------------------------------------------------

clear H score ok ;
for t = 1:100
    % estimate homograpyh
    subset = vl_colsubset(1:numMatches, 4) ;
    A = [] ;
    for i = subset
        A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
    end
    [U,S,V] = svd(A) ;
    H{t} = reshape(V(:,9),3,3) ;
    
    % score homography
    X2_ = H{t} * X1 ;
    du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
    dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
    zzz = 2;
    ok{t} = (du.*du + dv.*dv) < zzz*6*6 ;
    score(t) = sum(ok{t}) ;
end

[score, best] = max(score) ;
H = H{best} ;
ok = ok{best} ;

p1 = f1(:,matches(1,ok))';
p2 = f2(:,matches(2,ok))';
%x2(im1g); plotPolygons(p1,'r.');x2(im2g); plotPolygons(p2,'r.')
% --------------------------------------------------------------------
%                                                         Show matches
% --------------------------------------------------------------------
if toShow
    dh1 = max(size(im2,1)-size(im1,1),0) ;
    dh2 = max(size(im1,1)-size(im2,1),0) ;
    
    figure(1) ; clf ;
    subplot(2,1,1) ;
    imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
    o = size(im1,2) ;
    line([f1(1,matches(1,:));f2(1,matches(2,:))+o], ...
        [f1(2,matches(1,:));f2(2,matches(2,:))]) ;
    title(sprintf('%d tentative matches', numMatches)) ;
    axis image off ;
    
    subplot(2,1,2) ;
    imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
    o = size(im1,2) ;
    line([f1(1,matches(1,ok));f2(1,matches(2,ok))+o], ...
        [f1(2,matches(1,ok));f2(2,matches(2,ok))]) ;
    title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
        sum(ok), ...
        100*sum(ok)/numMatches, ...
        numMatches)) ;
    axis image off ;
    
    drawnow ;
    
    % --------------------------------------------------------------------
    %                                                               Mosaic
    % --------------------------------------------------------------------
    
    box2 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
    box2_ = inv(H) * box2 ;
    box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
    box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
    ur = min([1 box2_(1,:)]):max([size(im1,2) box2_(1,:)]) ;
    vr = min([1 box2_(2,:)]):max([size(im1,1) box2_(2,:)]) ;
    
    [u,v] = meshgrid(ur,vr) ;
    im1_ = vl_imwbackward(im2double(im1),u,v) ;
    
    z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
    u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
    v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
    im2_ = vl_imwbackward(im2double(im2),u_,v_) ;
    
    mass = ~isnan(im1_) + ~isnan(im2_) ;
    im1_(isnan(im1_)) = 0 ;
    im2_(isnan(im2_)) = 0 ;
    mosaic = (im1_ + im2_) ./ mass ;
    
    figure(2) ; clf ;
    imagesc(mosaic) ; axis image off ;
    title('Mosaic') ;
    
    if nargout == 0, clear mosaic ; end
    
end
end