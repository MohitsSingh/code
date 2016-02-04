function regions = get_sp_agglom_regions(I,seg_method)
I = im2uint8(I);
if (size(I,3)==1)
    I = cat(3,I,I,I);
end
spagglom_options;
if (nargin == 2)
    opts.seg_method = seg_method;
end
[region_parts,orig_sp,hists] = spagglom(I,opts);


size2 = @(x) [size(x,1) size(x,2)];

%
close all
n = length(region_parts);
m = zeros(size2(I));
xx = {};
yy = {};


regions = {};

% for k = length(region_parts):-1:1
for k =length(region_parts):-1:1
    if (mod(k,100)==0)
        k
    end
    %     x = double(sp{u}.pixels(:,1));
    %     y = double(sp{u}.pixels(:,2));
    %     mask = logical(full(sparse(x,y,ones(size(x,1),1), size(Ic,1),size(Ic,2))));
    mask = false(size2(I));
    curRegionParts = region_parts{k};
    for iu = 1:length(curRegionParts)
        u = curRegionParts(iu);
        if orig_sp{u}.size == 0
            continue
        end
        x = double(orig_sp{u}.pixels(:,1));
        y = double(orig_sp{u}.pixels(:,2));
        mask(sub2ind2(size(mask),[x y])) = 1;
        %         mask = mask | logical(full(sparse(x,y,ones(size(x,1),1), size(I,1),size(I,2))));
    end
    
    regions{k} = mask;
    m = m+mask;
    
    %         if (mod(k,20)==0)
    %             clf
    %             subplot(2,1,1),imagesc2(m);
    %             subplot(2,1,2);imagesc2(I);drawnow
    %         end
    %         if (mod(k,10)==0)
    %             displayRegions(I,mask,[],-1);pause(.01);drawnow
    %         end
    %             Ic = sp_image(I, [], orig_sp, [],region_parts{k});
    %             clf; imagesc(Ic); axis image; drawnow;
end

% subplot(2,1,1),imagesc2(m);
% subplot(2,1,2);imagesc2(I);
% drawnow; pause
% imshow(Ic)
end