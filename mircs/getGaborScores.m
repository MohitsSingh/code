function new_r = getGaborScores(frs,pss)
new_r = zeros(size(frs));
sz = size(pss{1});
[xx,yy] = meshgrid(1:sz(2),1:sz(1));
rads = ((yy-sz(1)/2).^2+(xx-sz(2)/2).^2).^.5;
rad_i = [0 15 20 25];
rr = zeros(3,18,length(new_r));
for k = 1:length(new_r)
    k
    fr2 = frs{k};
    ps = pss{k};
    if (isempty(fr2))
        continue;
    end
    for iRad = 2:length(rad_i)
        curRange = rads < rad_i(iRad) & rads>=rad_i(iRad-1);
%         m = bsxfun(@times,fr2,curRange.*(ps.^.5));
        m = bsxfun(@times,fr2,curRange);
        rr(iRad-1,:,k) = max(max(m,[],1),[],2);
    end
end


%%

new_r = squeeze(max(rr,[],2));
% scores = 1*(new_r(1,:)+0*new_r(2,:))+0*new_r(3,:);%-(hh>.3);