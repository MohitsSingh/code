function res = makeStringEnum(strings,toSort,toUpper)
if nargin < 2
    toSort = true;
end

if nargin < 3
    toUpper = false;
end

if toSort
    strings = sort(strings);
end
res = [];
for k = 1:length(strings)
    s = strings{k};
    if (toUpper)
        s = upper(s);
    end
    s = strrep(s,' ','_');
    eval(sprintf('res.%s=%d;',s,k));
end
end