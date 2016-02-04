function M =regionAdjacencyGraph(regions)
regions2 = cellfun(@(x) imdilate(x,ones(3)), regions, 'UniformOutput',false);

int_before = zeros(length(regions));
int_after = zeros(length(regions));
for ii = 1:length(regions)
    %     ii
    for jj = ii+1:length(regions)
        int_before(ii,jj) = nnz(regions{ii} & regions{jj});
        int_after(ii,jj) = nnz(regions2{ii} & regions2{jj});
    end
end

M = int_after > 0 & int_before == 0;
% end