function prob2perf(maps,masks)
maxSize = 150;

noObject = cellfun(@(x) nnz(x)==0,masks);

for t= 1:length(maps)
    %maps{t} = imResample(maps{t},maxSize/size(maps{t},1),'bilinear');    
    %masks{t} = imResample(masks{t},size(maps{t}),'nearest');
    maps{t} = col(maps{t});
    masks{t} = col(masks{t});
end
maps(noObject) = [];
masks(noObject) = [];
maps = cat(1,maps{:});
masks = double(cat(1,masks{:}));
masks(masks==0) = -1;

 vl_pr(masks, maps);