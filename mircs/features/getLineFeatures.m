function feats = getLineFeatures(lines_,D)
global pTypes;
% draw lines on a padded image to remove border issues.
% Z = zeros(dsize(I,1:2));
% [Z,xy] = paintLines(Z,round(lines_(:,[1 2 3 4])));
feats = struct('params',{},'xy',{});
% nonEmpty = cellfun(@(x)~isempty(x),xy);
% xy = xy(nonEmpty);
% lines_ = lines_(nonEmpty,:);
for k = 1:size(lines_,1)
    feats(k).params = lines_(k,:);
    %     feats(k).xy = xy{k};
    
    feats(k).startPoint = feats(k).params(1:2);
    feats(k).endPoint = feats(k).params(3:4);
    feats(k).bbox = pts2Box([ feats(k).startPoint; feats(k).endPoint]);
    feats(k).center = mean([feats(k).startPoint;feats(k).endPoint],1);
    feats(k).curveCenter = feats(k).center;
    feats(k).type = 'line';
    feats(k).typeID = pTypes.TYPE_LINE;
    feats(k).special = 0;
    if (nargin ==2 )
        feats(k).energy = ucmScore(D,feats(k).xy);
    end
    feats(k).absLength = norm(feats(k).endPoint-feats(k).startPoint);
end