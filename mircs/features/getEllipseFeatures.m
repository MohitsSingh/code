
function feats = getEllipseFeatures(ellipses_,D)

global pTypes;
feats = struct('params',{},'xy',{},'center',{},'bbox',{},'isDegenerate',{},...
    'startPoint',{},'endPoint',{},'middleVector',{},'curveCenter',{});

de = findDegenerateEllipses(ellipses_,3);

for k = 1:size(ellipses_,1)
    a = ellipses_(k,:);
    % render the ellipse's points.
    [~,x,y] = plotEllipse2(a(1),a(2),a(3),a(4),a(5:7),'g',100,2,[],false);
    feats(k).params = a;
    xy = [x(:) y(:)];
    feats(k).xy = xy;
    feats(k).bbox = pts2Box(xy);
    feats(k).center = a([2 1]);
    feats(k).isDegenerate = de(k);
    
    % the start-end point should be the two furthest apart points on this
    % ellipse
    d = triu(l2(xy,xy));
    
    [ii,jj] = find(d==max(d(:)));
    if (length(ii) >1)
        m = abs(jj-ii);
        [m,im] = max(m);
        ii = ii(im); jj = jj(im);
    end
    
    if (nargin == 2)
        feats(k).energy = ucmScore(D,xy);
    end
    
    %     clf; imagesc(D); hold on; plot(xy(:,1),xy(:,2),'g','LineWidth',2);
    %     title(num2str(feats(k).energy));
    %     pause;
    
    %     feats(k).ucmScore = ucmScore(ucm,xy);
    feats(k).startPoint = xy(ii,:);
    feats(k).endPoint= xy(jj,:);
    center_index = round((ii+jj)/2);
    feats(k).curveCenter = xy(center_index,:);
    %     feats(k).startPoint = xy(1,:);
    %     feats(k).endPoint= xy(end,:);
    %feats(k).curveCenter = xy(floor(end/2),:);
    feats(k).middleVector = feats(k).curveCenter-feats(k).center;
    feats(k).type = 'ellipse';
    feats(k).typeID = pTypes.TYPE_ELLIPSE;
    feats(k).special = 0;
end

