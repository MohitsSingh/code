function [M,allLocs,allFeats] = findNeighbors3(conf,samples,ids,suffix...
    ,toSave,do_flip,getQE,id_locs,keepFeats,locWeight)

% Given a set of samples, finds the nearest neighbor of each sample in each
% image, including locations and feature values.

%     seedFeats = samples{seedIdx};

if (nargin < 9)
    keepFeats = 1;
end

if (nargin < 7)
    getQE = 0;
end

if (nargin < 5)
    toSave = 0;
end

if (nargin < 10)
    locWeight = 0;
end

if (nargin < 4)
    suffix = '';
    if (isfield(conf,'suffix'))
        suffix = conf.suffix;
    end
end

feat_size = 31*conf.features.winsize^2;

if (toSave)
    
    neighborsPath = fullfile(conf.cachedir,['neighbors' suffix '.mat']);
    if (exist(neighborsPath,'file'))
        load(neighborsPath);
        return;
    end
end

n = length(ids);
m = size(samples,2);
M = zeros(m,n);

allLocs = cell(1,n);
allFeats = cell(1,n);

for k =1:n
    imageID = ids{k};
    disp(['scanning images: %' num2str(100*k/n)]);
    if (ischar(imageID))
        I = toImage(conf,getImagePath(conf,imageID));
    else
        I = imageID;
    end
    
    [X,uu,vv,scales,t ] = allFeatures( conf,I);
    locs_ = uv2boxes(conf,uu,vv,scales,t);
    
    if (~isempty(X) && ~isempty(id_locs))
        c = bsxfun(@minus,locWeight*boxCenters(locs_)',id_locs(:,k));
        X = [X;c];
    end
    
%     if (addLocation)
%         X = [X;addLocation*boxCenters(locs_)'];
%     else
       % samples = samples(1:feat_size,:);
%     end     
        
    D = l2(samples',X');
    
    %     Z = fillBoxes([size(I,1),size(I,2)],locs_,E,1);
    %     figure,imshow(Z,[]);
    if (getQE)
        E = getQuantizationError(X,conf.dict);
        E = max(E)-E;
        D = bsxfun(@plus,D,E');
    end
    
    if (do_flip) % also flip images when searching for nn
        I = flip_image(I);
        [X_flip,uu_flip,vv_flip,scales_flip,t_flip ] = allFeatures( conf,I );
        locs_flip = uv2boxes(conf,uu_flip,vv_flip,scales_flip,t_flip);
        
        if (~isempty(X_flip) && ~isempty(id_locs))
            c = bsxfun(@minus,locWeight*boxCenters(locs_flip)',id_locs(:,k));
            X_flip = [X_flip;c];
        end
%         if (addLocation)
%             X_flip = [X_flip;addLocation*boxCenters(locs_flip)'];
%         end
%         else
%             X_flip = X_flip(1:feat_size,:);
%         end
        D_flip = l2(samples',X_flip');
        if (getQE)
            E_flip =  getQuantizationError(X_flip,conf.dict);
            E_flip = max(E_flip)-E;
            %         Z_flip = fillBoxes([size(I,1),size(I,2)],locs_flip,E_flip,1);
            %         figure,imshow(flip_image(Z_flip),[]);
            D_flip = bsxfun(@plus,D_flip,E_flip');
        end
        
        
        [q,iq] = min([D,D_flip],[],2);
        c = iq>size(D,2);
        iq(c) = iq(c)-size(D,2);
        locs_(iq(c),:) = flip_box(locs_flip(iq(c),:),size(I));
        locs_(iq(c),7) = 1;
        X(:,iq(c)) = X_flip(:,iq(c));
    else
        [q,iq] = min(D,[],2);
        
    end
    if (~isempty(locs_))
        locs_(iq,11) = k;
        allLocs{k} = locs_(iq,:);
        if (keepFeats)
            allFeats{k} = X(:,iq);
        end
    end
    M(:,k) = q;
end
