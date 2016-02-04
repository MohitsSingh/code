function [parts,allRegions,scores] = followSegments2(conf,regions,G,regionConfs,I,...
    startBox,faceBox,regionOverlaps,binaryModels,binaryScores,faceConfs)
parts = [];
allRegions = [];
scores = [];
%%%%%DEBUGGING FLAG %%%%
debug_ = 0;
%%%%%DEBUGGING FLAG %%%%
% apply a class-specific decision process which forms chains of segments
% trying to maximize the score for a certain action.

partNames =  {'cup'    'hand'    'straw'    'bottle'};


% want the first segment to be inside the lip area.
[ovp_lip,ints_lip,areas_lip] = boxRegionOverlap(startBox,regions,size(regions{1}));

% [ss,iss] = sort(faceConfs,'descend');
% segScore_start = ss(1); % start with first segment...
% startRegions = iss(1);

startRegions = find(ovp_lip > 0);

G = max(G,G');

% do a dfs on nodes...
maxDepth = 2;

modelNames = {'cup','hand','straw','bottle'};

% todo - make sure that the model orders are sorted long to short,
% otherwise there will be a bug!!
% % % modelOrders = {[3,1],...% straw->cup
% % %     [1,2],...% cup->hand
% % %     [4,2]}; % bottle->hand

modelOrders = {[1,2]};% cup->hand
    

modelLength = cellfun(@length,modelOrders);
[L,iL] = sort(modelLength,'descend');
modelOrders = modelOrders(iL);

bestScore = -inf;
selectionStrategy = 1;

allRegions = [];
allOrders = [];

for iStart = 1:length(startRegions)
    Q = startRegions(iStart);
    Z = zeros(size(regions{Q}));
    visited = false(size(regions));
    regionGroups = {};
    z = zeros(size(visited));
    d = zeros(size(z));
    tic; % important, this is to keep an order on the detected segments.
    if (selectionStrategy) % do bfs from a start region
        bfsHelper(Q,0,maxDepth);
    else % just find all triplets of adjacent segments.
        regionGroups = {};
        for r = 1:size(G,1)
            f = find(G(r,:));
            for ii = 1:length(f)
                for jj = ii+1:length(f)
                    regionGroups{end+1} = [r f(ii) f(jj)];
                end
            end
        end
    end
    
    regionGroups = cat(1,regionGroups{:});
    regionGroups = regionGroups(:,2:end);
    if (isempty(regionGroups))
        continue;
    end
    
    for iOrder = 1:length(modelOrders);
        modelOrder = modelOrders{iOrder};
        
        t = ones(size(modelOrder)); % score weights
        curRegionGroups = regionGroups(:,1:length(modelOrder));
        curRegionGroups = unique(curRegionGroups,'rows');
        %scores = ones(1,size(curRegionGroups,1))*5*segScore_start(iStart);
        scores = zeros(1,size(curRegionGroups,1));
        %t = [1 1 1];
        %         faceInt = ints_face./areas_face;
        for iModel = 1:length(modelOrder)
            %             if (modelOrder(iModel)==0) % no part expected
            %                 continue;
            %             end
            curInds = curRegionGroups(:,iModel);
            curModelScores = regionConfs(modelOrder(iModel)).score;
            curModelScores = row(curModelScores(curInds));
            if (t(iModel) ~= 0) % this is to avoid 0*inf which is NaN
                scores = scores + t(iModel)*curModelScores;
            end
        end
        
        % add binary scores
        if (0) % TODO
            
            curBinaryScores = zeros(size(scores));
            binaryScoreWeight = 1;
            for iModel = 1:length(modelOrder)-1
                m1 = modelOrder(iModel);
                m2 = modelOrder(iModel+1);
                %             if (m1==0 || m2==0)
                %                 continue;
                %             end
                e1 = curRegionGroups(:,iModel);
                e2 = curRegionGroups(:,iModel+1);
                q_match = -1;
                for q = 1:length(binaryModels)
                    if strcmp(binaryModels(q).name,...
                            [partNames{m1} '_' partNames{m2}]) || ...
                            strcmp(binaryModels(q).name,...
                            [partNames{m2} '_' partNames{m1}])
                        q_match = q;
                        break;
                    end
                end
                
                %binaryScores{q_match}(e1,e2) % TODO - you were about to
                % write some code to get the binary scores for each
                % potential couple of neighbors.
                subs_ = [e1 e2 repmat(q_match,size(e1))];
                r = binaryScores(sub2ind2(size(binaryScores),subs_));
                %r = binaryScores(sub2ind(size(binaryScores),e1,e2));
                %                 r = cat(2,r{:});
                %                 [yhat, f] = binaryModels(m1,m2).classifier.test(r);
                curBinaryScores = curBinaryScores+binaryScoreWeight*r';
            end
            scores = scores + curBinaryScores;
            %         if(~isempty(allRegions))
        end
        allRegions{end+1} = [curRegionGroups,scores(:)];
        %         else
        %             allRegions = {[curRegionGroups,scores(:)]};
        %         end
        allOrders = [allOrders;iOrder*ones(size(scores(:)))];
    end
end

%     figure,plot(scores)
regions_ = cell(length(allRegions),1);
for k = 1:length(allRegions)
    r = {};
    for kk = 1:size(allRegions{k},1)
        r{kk} = allRegions{k}(kk,:);
    end
    regions_{k} = r(:);
end

allRegions = cat(1,regions_{:});
scores = zeros(length(allRegions),1);
for k = 1:length(allRegions)
    scores(k) = allRegions{k}(end);
end
% scores = {};
% for k = 1:length(allRegions)
%     scores{k} =  allRegions{k}(:,end);
% end

% scores = allRegions(:,end);
% allOrders = cat(1,allOrders{:});
scores(isnan(scores)) = -inf;
[r,ir] = sort(scores,'descend');

showDebug = false || conf.demo_mode;
if (~debug_ && ~showDebug)
    scores = r(1:5);
    allRegions = allRegions(ir(1:5));
    parts = cell(length(allRegions),1);
    %parts = cell(length(regions),5);
    for k = 1:size(parts,1)
        modelOrder = modelOrders{allOrders(ir(k))};
        parts{k} = modelNames(modelOrder);
    end
    
    %[parts,regions,scores,totalScore]
    %     for k = 1:5 % store top 5 results.
    %         modelOrder = modelOrders(allOrders(ir(k)),:);
    %
    %     end
else
    curCount = 0;
    k = 1;
    while (curCount < 1) % show only top 1
        %for k = 1:3%length(regionGroups)
        k
        if (r(k) > bestScore)
            bestScore = r(k);
            
        elseif (r(k) == bestScore)
            k = k+1;
            continue;
        end
        curCount = curCount + 1;
        k = k + 1;
        Z = zeros(dsize(I,1:2));
        modelOrder = modelOrders{allOrders(ir(k))};
        
        for kk = 1:length(modelOrder)
            Z(regions{allRegions{ir(k)}(kk)}) = Z(regions{allRegions{ir(k)}(kk)}) + kk;
            
        end
        
        clf;subplot(2,2,1); imagesc(I); axis image;
        subplot(2,2,2);
        bestIMG = repmat(Z>0,[1 1 3]).*I;
        imagesc(bestIMG);title(num2str(r(k)));axis image;
        subplot(2,2,3); imagesc(Z); axis image; hold on;
        
        [ii,jj] = find(~Z);
        
        sel_ = randi([1 length(ii)],3,2);
        
        for kk = 1:length(modelOrder)
            [yy,xx] = find(regions{allRegions{ir(k)}(kk)});
            xx = mean(xx);
            yy = mean(yy);
            x = jj(sel_(kk,1));
            y = ii(sel_(kk,2));
            text(x,y,modelNames{modelOrder(kk)},'Color','g');
            quiver(x,y,xx-x,yy-y,...
                0,'Color','g','LineWidth',2);
        end
        
        subplot(2,2,4);
        imagesc(Z); axis image; hold on;
        for kk = 1:size(modelOrder,2)
            [yy,xx] = find(regions{allRegions{ir(k)}(kk)});
            [yy,iyy] = min(yy); xx = xx(iyy);
            text(xx,yy,num2str(kk),'Color','g');
        end
        pause
    end
end

    function bfsHelper(curNode,curDepth,maxDepth)
        if (curDepth > maxDepth)
            return;
        end
        z(curNode) = toc;
        d(curNode) = curDepth;
        visited(curNode) = true;
        if (debug_)
            
            Z = Z+(curDepth+1)*regions{curNode};
            %             if (curDepth == 1)
            clf;imagesc(Z); axis image; hold on;
            %             end
            
            f = find_order(z);
            for ff = f
                [ii,jj] = find(regions{ff});
                ii = mean(ii); jj = mean(jj);
                text(jj,ii,num2str(d(ff)),'Color','y');
            end
            
            if (curDepth == maxDepth)
                f = find_order(z);
                title(num2str(f));
                %                 pause;
                pause(.01)
                
            end
        end
        
        % I don't want the neighbors to overlap any of the previously
        % visited vertices...
        
        
        curNeighbors = find(G(curNode,:) & ~visited);
        curOVP = sum(regionOverlaps(curNeighbors,find_order(z)),2);
        curNeighbors(find(curOVP)) = [];% TODO - check if removing this
        %         causes buggy situations.
        
        if (curDepth == maxDepth)
            f = find_order(z);
            regionGroups{end+1} = f;
        else
            
            for k = 1:length(curNeighbors)
                bfsHelper(curNeighbors(k),curDepth+1,maxDepth)
            end
        end
        if (debug_)
            Z = Z-(curDepth+1)*regions{curNode};
        end
        z(curNode) = false;
        visited(curNode) = false;
    end
end

function f = find_order(m)
inds = find(m);
[~,ii] = sort(m(inds),'ascend');
f = inds(ii);
end

function binaryRelations = getBinaryRelations(regions,regionGraph)
areas = cellfun(@(x) sum(x(:)), regions);
centroids = zeros(length(regions),2);
for k = 1:length(regions)
    [yy,xx] = find(regions{k});
    centroids(k,:) = [mean(xx) mean(yy)];
end
[ii,jj] = find(regionGraph);
areaRatio = zeros(size(regionGraph));
binaryRelations.centroid_diff = zeros([size(regionGraph) 2]);
for k = 1:length(ii)
    e1 = ii(k);
    e2 = jj(k);
    areaRatio(e1,e2) = areas(e1)/areas(e2);
    binaryRelations.centroid_diff(e1,e2,:) = normalize_vec(centroids(e2,:)-centroids(e1,:));
end
binaryRelations.areaRatio = areaRatio;
end

function s = knnScore(samples,candidates,distType)
s = 0;
if (strcmp(distType,'dist'))
    D = l2(candidates,samples);
    
    s = row(sum(exp(-D/var(samples)),2));
    
elseif strcmp(distType,'dot')
    s = sum(exp(samples*candidates'));
else
    error('knnScore : distType must be ''dist'' or ''dot''');
end
end