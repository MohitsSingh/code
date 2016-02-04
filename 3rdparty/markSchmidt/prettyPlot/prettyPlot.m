function [] = prettyPlot(xData,yData,options)
% prettyPlot(xData,yData,options)
%
% available options:
% legend - cell array containing legend entries (default = [])
% title - string containing plot title (default = [])
% xlabel - string containing x-axis label (default = [])
% ylabel - string containing y-axis label (default = [])
% lineWidth - width of lines (default = 3)
% colors - cell array or (n by 3) matrix containing line colors (default = 'b')
% lineStyles - cell array containing line styles (default = '-')
% markerSize - size of markers (default = [])
% markers - cell array containing markers (default = [])
% markerSpacing - (n by 2) matrix containing spacing between markers and offset for first marker
% xlimits - 2-vector containing lower and upper limits for x-axis (can be inf to show full range)
% ylimits - 2-vector containing lower and upper limits for y-axis (can be inf to show full range)
% logScale - can be 0 (regular scale), 1 (semilogx), or 2 (semilogy)
% legendLoc - location of legend (default = 'Best')
% useLines - whether to use lines (default = 1)
% fillFace - whether to fill markers (default = 1)
% errors - (n by p by 2) array containing upper and lower error lines
% errorStyle - line style for error bars

if nargin < 3
	options = [];
end

[legendStr,plotTitle,plotXlabel,plotYlabel,lineWidth,colors,lineStyles,...
	markerSize,markers,markerSpacing,xlimits,ylimits,logScale,legendLoc,...
	useLines,fillFace,errors,errorStyle,errorColors] = ...
	myProcessOptions(options,'legend',[],'title',[],'xlabel',[],'ylabel',[],...
	'lineWidth',3,'colors',[],'lineStyles',[],...
	'markerSize',12,'markers',[],'markerSpacing',[],...
	'xlimits',[],'ylimits',[],...
	'logScale',0,'legendLoc','Best','useLines',1,'fillFace',1,...
	'errors',[],'errorStyle',{'--'},'errorColors',[]);

if logScale == 1
	plotFunc = @semilogx;
elseif logScale == 2
	plotFunc = @semilogy;
else
	plotFunc = @plot;
end

if useLines == 0
	defaultStyle = 'b.';
else
	defaultStyle = 'b';
end

if iscell(yData)
	nLines = length(yData);
else
	nLines = size(yData,1);
end

for i = 1:nLines
	
	% Get yData for line
	if iscell(yData)
		y{i} = yData{i};
	else
		y{i} = yData(i,:);
	end
	
	% Get xData for line
	if isempty(xData)
		x{i} = 1:length(y);
	elseif iscell(xData)
		x{i} = xData{i};
	elseif size(xData,1) == 1
		x{i} = xData(1:length(y{i}));
	else
		x{i} = xData(i,:);
	end
	
	% Plot
	h(i) = plotFunc(x{i},y{i},defaultStyle);
	hold on;
end

if isempty(markerSpacing)
	for i = 1:length(h)
		h(i) = applyStyle(h(i),i,lineWidth,colors,lineStyles,markers,markerSpacing);
	end
else
	for i = 1:length(h)
		h(i) = applyStyle(h(i),i,lineWidth,colors,lineStyles,markers,markerSpacing);
		if ~isempty(markers) && ~isempty(markers{1+mod(i-1,length(markers))})
			hM = plotFunc(x{i}(markerSpacing(i,2):markerSpacing(i,1):end),y{i}(markerSpacing(i,2):markerSpacing(i,1):end),'b.');
			applyStyle(hM,i,lineWidth,colors,[],markers,[]);
			hM = plotFunc(x{i}(markerSpacing(i,2)),y{i}(markerSpacing(i,2)),defaultStyle);
			h(i) = applyStyle(hM,i,lineWidth,colors,lineStyles,markers,[]);
		end
	end
end

if ~isempty(errors)
	if isempty(errorColors)
		errorColors = colors+.75;
		errorColors(:) = min(errorColors(:),1);
	end
	for i = 1:length(h)
		hL = plotFunc(x{i},errors{i,1},defaultStyle);
		hU = plotFunc(x{i},errors{i,2},defaultStyle);
		applyStyle(hL,i,lineWidth,errorColors,errorStyle,[],markerSpacing);
		applyStyle(hU,i,lineWidth,errorColors,errorStyle,[],markerSpacing);
	end
end

set(gca,'FontName','AvantGarde','FontWeight','normal','FontSize',12);

if ~isempty(legendStr)
	h = legend(h,legendStr);
	set(h,'FontSize',10,'FontWeight','normal');
	set(h,'Location','Best');
	set(h,'Location',legendLoc);
end

if ~isempty(plotTitle)
	h = title(plotTitle);
	set(h,'FontName','AvantGarde','FontSize',12,'FontWeight','bold');
end

if ~isempty(plotXlabel) || ~isempty(plotYlabel)
	h1 = xlabel(plotXlabel);
	h2 = ylabel(plotYlabel);
	set([h1 h2],'FontName','AvantGarde','FontSize',12,'FontWeight','normal');
end

set(gca, ...
	'Box'         , 'on'     , ...
	'TickDir'     , 'out'     , ...
	'TickLength'  , [.02 .02] , ...
	'XMinorTick'  , 'off'      , ...
	'YMinorTick'  , 'off'      , ...
	'LineWidth'   , 1         );

if ~isempty(xlimits)
	xl = xlim;
	xlimits(xlimits == -inf) = xl(1);
	xlimits(xlimits == inf) = xl(2);
	xlim(xlimits);
end
if ~isempty(ylimits)
	yl = ylim;
	ylimits(ylimits == -inf) = yl(1);
	ylimits(ylimits == inf) = yl(2);
	ylim(ylimits);
end

set(gcf, 'PaperPositionMode', 'auto');


	function [h] = applyStyle(h,i,lineWidth,colors,lineStyles,markers,markerSpacing)
		hold on;
		set(h,'LineWidth',lineWidth);
		if ~isempty(colors)
			if iscell(colors)
				set(h,'Color',colors{1+mod(i-1,length(colors))});
			else
				set(h,'Color',colors(1+mod(i-1,size(colors,1)),:));
			end
		end
		if ~isempty(lineStyles) && useLines
			if isempty(lineStyles)
				set(h,'LineStyle','-');
			else
				if ~isempty(lineStyles{1+mod(i-1,length(lineStyles))})
					set(h,'LineStyle',lineStyles{1+mod(i-1,length(lineStyles))});
				end
			end
		end
		if ~isempty(markers)
			if ~isempty(markers{1+mod(i-1,length(markers))})
				if isempty(markerSpacing)
					set(h,'Marker',markers{1+mod(i-1,length(markers))});
					set(h,'MarkerSize',markerSize);
					if fillFace
						set(h,'MarkerFaceColor',[1 1 .9]);
					end
				end
			end
		end
	end
end