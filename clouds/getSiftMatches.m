function [xy_src,xy_dst] = getSiftMatches(I1,I2)
[f1,d1] = vl_covdet(I1,'Method','MultiscaleHessian');%,'descriptor','patch');
[f2,d2] = vl_covdet(I2,'Method','MultiscaleHessian');%,'descriptor','patch');

d1 = rootsift(d1);
d2 = rootsift(d2);
% [f1,d1] = vl_sift(I1);
% [f2,d2] = vl_sift(I2);
matches = best_bodies_match(single(d1)',single(d2)');
%     matches = vl_ubcmatch(d1,d2);
%     matches = knn_match(d1',d2');
xy_src = f1(1:2,matches(1,:))';
xy_dst = f2(1:2,matches(2,:))';