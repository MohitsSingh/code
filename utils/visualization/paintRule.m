function m = paintRule(m,labels,trueColor,falseColor,borderWidth)
if (nargin < 3) %|| isempty(trueColor))
    trueColor = [0 255 0];
end
if (nargin < 4) %|| isempty(falseColor))
    falseColor = [255 0 0];
end
if (nargin < 5)
    borderWidth = 1;
end
for k = 1:length(m)
    if (labels(k)>0)
        if (~isempty(trueColor))
            m{k} = addBorder(m{k},borderWidth,trueColor);
        end
    else
        if (~isempty(falseColor))
            m{k} = addBorder(m{k},borderWidth,falseColor);
        end
    end
end
end