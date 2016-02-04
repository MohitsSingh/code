function hists = getHists(conf,model,imgs,masks,feats)
if (nargin < 4)
    masks = [];
end
if (nargin < 5)
    feats = [];
end
n = length(imgs);
hists = cell(1,n);

% parfor   k = 1:n
for k = 1:n
%     k
    img = getImage(conf,imgs{k});
    curFeats = [];
    if (~isempty(feats))
        curFeats = feats(k);
    end
    
    if (~isempty(masks))
        if (iscell(masks))
                hists{k} = getImageDescriptor(model, img,masks{k},curFeats );
        else
            hists{k} = getImageDescriptor(model, img,masks(k,:),curFeats );
        end
    else
        hists{k} = getImageDescriptor(model, img,ones(size(img,1),size(img,2)),curFeats );
    end
    %     hists{k} = sparse(double(hists{k}));
end
hists = cat(2,hists{:});
end