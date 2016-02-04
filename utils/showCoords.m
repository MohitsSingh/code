function showCoords(poly,toShow,varargin)
x = poly(:,1);
y = poly(:,2);
if (nargin < 2 || isempty(toShow))
    toShow = cell(length(x),1);
    for t = 1:length(x)
        toShow{t} = num2str(t);
    end
end
for kk = 1:length(x)
    text(x(kk),y(kk),toShow{kk},'color','c','FontSize',11,'fontweight','bold','Background','white',varargin{:})
end
end