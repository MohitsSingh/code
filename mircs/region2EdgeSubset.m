function d = region2EdgeSubset(G,data,region_sel)
d = zeros([size(G) size(data,1)]);
%     d = cell(size(G));
[ii,jj] = find(G);
for k = 1:length(ii)
    d(ii(k),jj(k),:) = data(:,k);
    d(jj(k),ii(k),:) = data(:,k);
end
%     d = cell(size(G));
%     [ii,jj] = find(G);
%     for k = 1:length(ii)
%         d{ii(k),jj(k)} = data(:,k);
%         d{jj(k),ii(k)} = -data(:,k);
%     end
d = d(region_sel,region_sel,:);
end