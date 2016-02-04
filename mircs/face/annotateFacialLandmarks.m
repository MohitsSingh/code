function annotateFacialLandmarks(conf,imgData,reqKeyPoints,dbPath)
annoPath = j2m(dbPath,imgData);
%curLandmarks = struct('pts',{},'skipped',{})
if (exist(annoPath,'file'))
    load(annoPath);
    % end
else % initialize keypoints struct.
    for t = 1:length(reqKeyPoints)
        fn = reqKeyPoints{t};
        curLandmarks.(fn).pts = [];
        curLandmarks.(fn).skipped = false;
    end
end
all_pts = {};
anyEmpty = false;
for k = 1:length(reqKeyPoints)
    fn = reqKeyPoints{k};
    if (~isfield(curLandmarks,fn))
        curLandmarks.(fn).pts = [];
        curLandmarks.(fn).skipped = false;
    end
    curPts = curLandmarks.(fn).pts;
    if (~isempty(curPts))
        all_pts{end+1} = curLandmarks.(fn).pts;
    else
        anyEmpty  =true;
    end
end
if (~anyEmpty)
    return;
end
I = getImage(conf,imgData);

faceBox = inflatebbox(imgData.faceBox,[1 1]*2,'both',false);
LEFT_CLICK = 1;
RIGHT_CLICK =3;
SPACEBAR = 32;
toQuit = false;
% while(~toQuit)
clf;
imagesc2(I);
xlim(faceBox([1 3]));
ylim(faceBox([2 4]));

curPts = cat(1,all_pts{:});
plotPolygons(curPts,'g.');
% pause;


curPts = cat(1,all_pts{:});
plotPolygons(curPts,'g.');
for t = 1:length(reqKeyPoints)
    % go over keypoints, all but skipped
    curKP = reqKeyPoints{t};
    if isempty(curLandmarks.(curKP).pts)
        if ~curLandmarks.(curKP).skipped
            T = sprintf('left click to annonate %s\nright click to skip point\nSPACE to quit',curKP);
            title(T);
            
            [x,y,b] = ginput(1);
            switch b
                case LEFT_CLICK
                    curLandmarks.(curKP).pts = [x y];
                    plot(x,y,'g.');
                    curLandmarks.(curKP).skipped = false;
                case RIGHT_CLICK
                    curLandmarks.(curKP).skipped = true;
                case SPACEBAR
                    toQuit = true;
                    break
            end
        end
    end
    if (toQuit),break,end
    %
    %         break;
    %     else
    %         ii = 'y';
    %     end
    %
end
% ii = input('would you like to save the current annotations?(y/n) ','s');
% if (strcmp(ii,'y'))
save(annoPath,'curLandmarks');
% end

% end

end