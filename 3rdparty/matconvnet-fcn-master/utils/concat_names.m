function subsetName = concat_names(class_names,f,prefix);
%CONCAT_NAMES Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    subsetName = {};
else
    subsetName = {prefix};
end
for t = 1:length(f)
    subsetName{end+1} = class_names{f(t)};
    if t < length(f)
        subsetName{end+1} = '_and_';
    end
end
subsetName = cat(2,subsetName{:});

end

