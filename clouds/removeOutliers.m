function ff = removeOutliers(ff)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% remove unlikely disparities
% % xy1 = ff.xy_src;
% xy2 = ff.xy_dst;
% norms = vec_norms(xy1-xy2);
%figure,plot(norms,ff.xyz(:,3),'r.')

%ff.xyz(:,1)
%figure,hist(ff.xyz(:,3),100)
z = ff.xyz(:,3);
pd = fitdist(z,'normal');

L = log(pd.pdf(z));
T = -8.5;
ff.xy_src(L<T,:) = [];
ff.xy_src_rect(L<T,:) = [];
ff.xy_dst(L<T,:) = [];
ff.xy_dst_rect(L<T,:) = [];
ff.xyz(L<T,:) = [];
end

