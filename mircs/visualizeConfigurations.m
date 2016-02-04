function res = visualizeConfigurations(I,configs,scores,maxToDisplay,delay,coarseBox,imageID)
if nargin < 3
    scores = zeros(size(configs));
end
if nargin < 4
    maxToDisplay = length(scores);
end
if nargin < 5
    delay = 0;
end
if nargin < 6
    coarseBox = []
end
if nargin < 7
    imageID = []
end
maxToDisplay=  min(maxToDisplay,length(scores));
[scores,ir] = sort(scores,'descend');
configs = configs(ir);
res = {}
for t = 1:maxToDisplay
    t
    hh = clf; imagesc2(I);
    boxColors = ['r','g','b'];
    r = {};
    for u = 1:length(configs{t})
        %plotBoxes(configs{t}(u).bbox,[boxColors(u) '-'],'LineWidth',2);
        plotPolygons(configs{t}(u).xy,[boxColors(u) '-'],'LineWidth',2);
        r{u} = poly2mask2(configs{t}(u).xy,size2(I));
        if u > 1
            prevBox = configs{t}(u-1).bbox;
            curBox = configs{t}(u).bbox;
            box_int = getInteractionRegion(prevBox,curBox);
            plotPolygons(boxCenters(box_int),'m*','MarkerSize',8,'LineWidth',4)
            %plotBoxes(box_int,'m-');        
        end
        %         plotBoxes(configs{t}(u).bbox);
    end
    res{t} = sum(cat(3,r{:}),3)>0;
    if ~isempty(coarseBox)
        %plotBoxes(coarseBox,'y--','LineWidth',2);
    end        
    % show the linking features....
%     get_in
    title(num2str(scores(t)))
% % %     if ~isempty(imageID)
% % %         outDir = '/home/amirro/notes/images/2015_06_15'
% % %         outFile = fullfile(outDir,sprintf('%s_%03.0f.png',imageID,t));
% % %         saveas(hh,outFile)
% % %     end
% % %     %
    dpc(delay)
end