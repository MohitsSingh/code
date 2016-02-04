function hogpic = jettify(hogpic,do_normalize,mask)
%Turn a matrix which can be viewed via imagesc into an image using
%the same jet scheme, but now the result can be written to a file
if (iscell(hogpic))
    for k = 1:length(hogpic)
        hogpic{k} = jettify(hogpic{k});
    end
    return;
end
NC = 200;
if (nargin < 2)
    do_normalize = 1;
end
colorsheet = jet(NC);
if (nargin < 3)
    mask = true(size(hogpic));
end

dists = hogpic(:);

  if (do_normalize)
    dists(mask(:)) = dists(mask(:)) - min(dists(mask(:)));
    dists(mask(:)) = dists(mask(:))/ (max(dists(mask(:)))+eps);
end
dists = round(dists*(NC-1)+1);
colors = colorsheet(dists,:);
hogpic = reshape(colors,[size(hogpic,1) size(hogpic,2) 3]);