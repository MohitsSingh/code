
function inds = makeInds(elems)
    inds = cell(size(elems));
    for k = 1:numel(elems)
        inds{k} = k*ones(size(elems{k},1),1);
    end
    inds = cat(1,inds{:});
end