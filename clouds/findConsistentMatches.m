function ff = findConsistentMatches( ff , T)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    T = 20;
end
dst1 = ff(1).xy_dst;
src2 = ff(2).xy_src;

[m,im] = min(l2(dst1,src2),[],2);
%goods = m < .5;
%xyz_1 = ff(1).xyz;%(goods,:);
ff(2) = getPointSubset(ff(2),im);
xyz_1 = ff(1).xyz;
xyz_2 = ff(2).xyz;
diff_3d = xyz_1-xyz_2;
n = vec_norms(diff_3d);
goods = m < .5;
sel_ = goods & n < T;
ff(1) = getPointSubset(ff(1),sel_);
ff(2) = getPointSubset(ff(2),sel_);

function f = getPointSubset(f,sel_)
    f.xy_src = f.xy_src(sel_,:);
    f.xy_dst = f.xy_dst(sel_,:);
    f.xy_src_rect = f.xy_src_rect(sel_,:);
    f.xy_dst_rect = f.xy_dst_rect(sel_,:);
    f.xyz = f.xyz(sel_,:);

