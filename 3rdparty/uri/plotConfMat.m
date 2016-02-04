function plotConfMat(confMat, varargin)

n = size(confMat,1);
f = 1/n;

labels = checkVarargin(varargin, 'labels');
xsmall = checkVarargin(varargin, 'xsmall', false);
ysmall = checkVarargin(varargin, 'ysmall', false);
xlab = checkVarargin(varargin, 'xlab', true);
ylab = checkVarargin(varargin, 'ylab', true);
xrot = checkVarargin(varargin, 'xrot', true);
yrot = checkVarargin(varargin, 'yrot', true);
fontsize = checkVarargin(varargin, 'fontsize', 12);

if (isempty(labels))
    labels = arrayfun(@num2str, 1:n, 'uniformoutput', false);
    slabels = labels;
else
    slabels = cellfun(@(s) [s(1) s(end)], labels, 'uniformoutput', false);
end

imagesc(confMat);
if (xlab)
    text(1.00, -0.01, 'predicted', 'units', 'normalized', 'rotation', 90, 'fontsize', fontsize, 'HorizontalAlignment', 'right')
end

if (ylab)
    text(-0.01, 1, 'actual', 'units', 'normalized', 'fontsize', fontsize, 'HorizontalAlignment', 'right')
end

if (xsmall)
    set(gca,'xticklabel', slabels);
else
    if (xrot)
        set(gca,'xticklabel', repmat('',1,n));
        text(f/2:f:1-f/2, zeros(1,n)-0.01, labels, 'units', 'normalized', 'rotation', 45, 'HorizontalAlignment', 'right', 'fontsize', fontsize)
    else
        set(gca,'xticklabel', labels);
    end
end

if (ysmall)
    set(gca,'yticklabel', slabels);
else
    if (yrot)
        set(gca,'yticklabel', repmat('',1,n));
        text(zeros(1,n)-0.02, f/2:f:1-f/2, labels(end:-1:1), 'units', 'normalized', 'rotation', 45, 'HorizontalAlignment', 'right', 'fontsize', fontsize)
    else
        set(gca,'yticklabel', labels);
    end
end;

% text
t = arrayfun(@fstr, norm1(confMat,2), 'uniformoutput', false);
[x y] = meshgrid([0:f:1-f]+0.0225, [1-f:-f:0]+0.07);
text(x(:), y(:), t(:), 'units', 'normalized', 'fontsize', fontsize);

% font
set(gca, 'fontsize', fontsize);

% color
load white2green;
colormap(colmap);


function str = fstr(f)
if f == 1
    str = '1.0';
else
    str = sprintf('%.2f', f);
    str = str(2:end);
end
