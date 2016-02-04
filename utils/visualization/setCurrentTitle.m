function setCurrentTitle(str,toAppend)
if (nargin < 2)
    toAppend = true;
end
curTitle = '';
if toAppend
    curTitle = get(get(gca,'title'),'string');
end

title([curTitle str]);
end
