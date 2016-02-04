function dist_signed = bwdist_sign(bw)
%bwdist_sign Return signed distance to boundary of binary image,
% where negative distance are distances to the complementary image (i.e
% inside of blobs)
dist_signed = bwdist(bw);
dist_neg = bwdist(~bw);
mask = dist_signed == 0;
dist_signed(mask)=-(dist_neg(mask)-1);

end

