function x = replaceEmpty(x)

sizes = cellfun(@(x) size(x,1),x);
[maxSize,iMaxSize] = max(sizes(:));
zz = size(x{iMaxSize});
for t = 1:length(x)
    if sizes(t)<maxSize
        x{t} = nan(zz);
    end
end

% end
% firstNotEmpty = find(cellfun(@(x) ~isempty(x),x),1,'first');
% if (any(firstNotEmpty))
%     zz = size(x{firstNotEmpty});
%     for t = 1:length(x)
%         if (isempty(x{t}))
%             x{t} = zeros(zz);
%         end
%     end
% else
% warning('replaceEmpty: all elements are empty, returning 0');
% x = {0};
end