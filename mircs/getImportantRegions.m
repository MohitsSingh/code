function importantRegions = getImportantRegions(conf,w,ids,UU,toDebug)
if (nargin < 5)
    toDebug = false;
end
importantRegions = struct('imageID',{},'rect',{},'feats_inside',{},'feats_outside',{},'score',{});
% ticID = ticStatus( 'finding imporant regions...', .5, .5, true);
% ids = ids(1:40:end);

UU_ids = {UU.imageID};

[c,ia,ib] = intersect(ids,UU_ids,'stable');

for it = 1:length(ids)
    t = ia(it);
    curID = ids{t};
    L = UU(ib(it));
    %     if (none(strfind(curID,'drink'))),continue,end
    %     featsPath = j2m('~/storage/s40_kriz_fc6_block_5_2',curID);
    %     L = load(featsPath);
    rects = L.rects;
    scores = w(1:end-1)'*L.feats;
    [r,iRect] = min(scores);
    importantRegions(t).imageID = curID;
    importantRegions(t).rect = rects(iRect,:);
    importantRegions(t).feats_inside = L.feats_windows(:,iRect);
    importantRegions(t).feats_outside = L.feats(:,iRect);
%     tocStatus(ticID,t/length(ids));
    if (toDebug)
            clf;imagesc2(getImage(conf,curID));plotBoxes(rects(iRect,:)); dpc
    end
end
% tocStatus(ticID,1);