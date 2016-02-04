function faceLandmarks = loadFaces(conf,ids,suff,baseDir,resName)
faceLandmarks = struct;
% suff = '_x2.mat';

debug_ = false;
conf.get_full_image = true;

resFile = fullfile('~/storage', [resName '.mat']);
if (exist(resFile,'file'))
    load(resFile);
    return;
end
for k = 1:length(ids)
    k
    curID = ids{k};
    
    
    [I,xmin,xmax,ymin,ymax]  = getImage(conf,curID);
    personBox = [xmin ymin xmax ymax];
    bs_ = [];
    if (debug_)
        
        clf;
        imshow(I);hold on;
    end
    for iSuff = 1:length(suff)
        curFile = fullfile(baseDir,strrep(curID,'.jpg',suff{iSuff}));
        load(curFile); % -->bs
        
        if (debug_)
            
            
            for n = 1:length(bs)
                bc = boxCenters(bs(n).xy)/iSuff;
                plot(bc(:,1),bc(:,2),'g.');
            end
            
            
        end
        
        for n = 1:length(bs)
            bs(n).xy = bs(n).xy/iSuff;
        end
        
        bs_ = [bs_,bs];
    end
    if (debug_)
        
        pause;
    end
    faceLandmarks(k).bs = bs_;
    faceLandmarks(k).personBox = personBox;
end
save(resFile,'faceLandmarks');