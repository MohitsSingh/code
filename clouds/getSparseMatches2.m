function [xy_src,xy_dst,d] = getSparseMatches2(I1,I2,resampleFactor,constrainBuddies)
if nargin < 3
    resampleFactor = 1;
end
if nargin < 4
    constrainBuddies = false;
end
I1 = imResample(I1,resampleFactor);
I2 = imResample(I2,resampleFactor);
m1 = I1>10/255;
m2 = I2>10/255;
bounds1 = region2Box(m1);
bounds1(1:2) = bounds1(1:2)-15;
bounds1(3:4) = bounds1(3:4)+15;
bounds2 = region2Box(m2);
bounds2(1:2) = bounds2(1:2)-15;
bounds2(3:4) = bounds2(3:4)+15;
S = 4;
[f1,d1] = vl_dsift(I1,'Step',1,'Bounds',bounds1,'size',S,'FloatDescriptors');
[f2,d2] = vl_dsift(I2,'Step',1,'Bounds',bounds2,'size',S,'FloatDescriptors');

d1 = rootsift(d1);
d2 = rootsift(d2);
% 
% sizes = [2 4];
% [f1,d1] = vl_phow(I1,'Step',1,'sizes',sizes);
% [f1,d1] = stack_features(f1,d1);
% [f2,d2] = vl_phow(I2,'Step',1,'sizes',sizes);
% [f2,d2] = stack_features(f2,d2);
% 

f1_inds = sub2ind2(size2(I1),round(f1([2 1],:)'));
f1_sel = m1(f1_inds);
f1 = f1(:,f1_sel);
d1 = d1(:,f1_sel);
f2_inds = sub2ind2(size2(I2),round(f2([2 1],:)'));
f2_sel = m2(f2_inds);
f2 = f2(:,f2_sel);
d2 = d2(:,f2_sel);
% clf;imagesc2(I1); vl_plotframe(f1(:,1:10:end));

% [f1,d1] = vl_covdet(I1,'Frames',f1,'EstimateAffineShape',true,'EstimateOrientation',false,'descriptor','liop');
% [f2,d2] = vl_covdet(I2,'Frames',f2,'EstimateAffineShape',true,'EstimateOrientation',false,'descriptor','liop');




% matches = best_bodies_match(d1',d2');
% %             matches = knn_match(d1',d2');
% xy_src = f1(1:2,matches(1,:))'/resampleFactor;
% xy_dst = f2(1:2,matches(2,:))'/resampleFactor;

xy_src_ = {};
xy_dst_ = {};
for t = 1:1
    t
    d =0;
    if constrainBuddies
        [matches,d] = best_bodies_match(d1',d2',f1,f2);
    else
        [matches,d] = best_bodies_match(d1',d2');
    end
    
    z1 = false(1,size(d1,2));
    z2 = false(1,size(d2,2));
    z1(matches(1,:)) = true;
    z2(matches(2,:)) = true;
    %             matches = knn_match(d1',d2');
    xy_src_{end+1}= f1(1:2,matches(1,:))'/resampleFactor;
    xy_dst_{end+1}= f2(1:2,matches(2,:))'/resampleFactor;
    
    f1 = f1(:,~z1);
    f2 = f2(:,~z2);
    d1 = d1(:,~z1);
    d2 = d2(:,~z2);
end

xy_src = cat(1,xy_src_{:});
xy_dst = cat(1,xy_dst_{:});