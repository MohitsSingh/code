function landmarks = extractLandmarks(I,thetas,models)
toScale = 1;useTopHalf = false;
if (nargin < 3)
    thetas = -20:20:20;
end
% thetas = -20;
landmarks = struct('s',{},'c',{},'xy',{},'level',{},'polys',{},'rotation',{},'isvalid',{});
if (nargin < 3)
    models = {'face_p146_small','multipie_independent'};
end
n = 0;
for iModel = 1:length(models)
    for iTheta =1:length(thetas)
        %         iTheta
        n = n+1;
        curTheta = thetas(iTheta);
        U = imrotate(I,curTheta,'bilinear','crop');
        curLandmarks = detect_landmarks(U,toScale,useTopHalf,models{iModel});
        curLandmarks = curLandmarks{1};
        if (~isempty(curLandmarks))
            curLandmarks = curLandmarks(1);
            landmarks(n).s = curLandmarks.s;
            landmarks(n).c = curLandmarks.c;
            landmarks(n).xy = curLandmarks.xy;
            landmarks(n).level = curLandmarks.level;
            landmarks(n).isvalid = true;
            polys = rotate_bbs(curLandmarks.xy,U,curTheta);
            landmarks(n).polys = polys(:);
            landmarks(n).rotation = curTheta;
        else
            landmarks(n).isvalid = false;
        end
    end
end