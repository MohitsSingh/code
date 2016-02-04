function binaryFactors = learnBinaryFactors(conf,train_ids,train_labels,groundTruth,partNames)

debug_ = false;
conf.get_full_image = true;
gtInds = zeros(size(groundTruth));
gtInds(1) = 1;
% split ground-truth according to source images.
count_ = 1;
for k = 2:length(groundTruth)
    if (~strcmp(groundTruth(k).sourceImage,groundTruth(k-1).sourceImage)) % still the same.
        count_ = count_+1;
    end
    gtInds(k) = count_;
end
rsf = RelativeShapeFeatureExtractor(conf);
relativeShape = struct('part1',{},'part2',{},'shape',{});
for i1 = 1:length(partNames)
    for i2 = 1:length(partNames)
        relativeShape(i1,i2).part1 = partNames{i1};
        relativeShape(i1,i2).part2 = partNames{i2};
        relativeShape(i1,i2).shape = {};
    end
end

u = unique(gtInds);
for k = 1:length(u)
    currentInds = find(gtInds==u(k));
    curObjects = groundTruth(currentInds);
    co_occurence_local = zeros(length(partNames));
    %     curCentroids = centroids(currentInds,:);
    curImage = getImage(conf,groundTruth(currentInds(1)).sourceImage);
    if (debug_)
        curImage = getImage(conf,groundTruth(currentInds(1)).sourceImage);
    end
    for i1 = 1:length(curObjects)
        if (debug_)
            clf; imagesc(curImage);
            hold on;
        end
        x1 = curObjects(i1).polygon.x;
        y1 = curObjects(i1).polygon.y;
        r1 = roipoly(curImage,x1,y1);
        if (debug_)
            plot([x1;x1(1)],[y1;y1(1)],'g-','LineWidth',2);
            plot(curCentroids(i1,1),curCentroids(i1,2),'r+');
        end
        id1 = curObjects(i1).partID;
        %         counts(id1) = counts(id1)+1;
        [x1,y1] = poly2cw(x1,y1);
        a1 =polyarea(x1,y1);
        for i2 =  setdiff(1:length(curObjects),i1)%length(curObjects)
            %         for i2 =  i1+1:length(curObjects)
            x2 = curObjects(i2).polygon.x;
            y2 = curObjects(i2).polygon.y;
            if (debug_)
                plot([x2;x2(1)],[y2;y2(1)],'r-','LineWidth',2);
                plot(curCentroids(i2,1),curCentroids(i2,2),'g+');
            end
            [x2,y2] = poly2cw(x2,y2);
            r2 = roipoly(curImage,x2,y2);
            x = rsf.extractFeatures({r1,r2},[1 2]);
            
            
            a2 = polyarea(x2,y2);
            id2 = curObjects(i2).partID;
            if (~isnan(x))
                relativeShape(id1,id2).shape{end+1} = x;
            end
            
            %             fprintf('i1=%d\ti2=%d\nid1=%d\tid2=%d\na1=%f\ta2=%f\n',i1,i2,id1,id2,a1,a2);
            %             pause;
            
            % union
            
            [x,y] = polybool('intersection',x1,y1,x2,y2);
            if (debug_)
                plot(x,y,'m-','LineWidth',2);
            end
            currentInt = polyarea(x,y);
            if (currentInt > 0) % there is a shared boundary.
                %                 r1 = roipoly(
            end
            
            co_occurence_local(id1,id2) = 1;
            %             allCounts(id1,id2) = allCounts(id1,id2)+1;
            %             currentOVP = currentInt/currentUnion;
            %             if (isnan(currentOVP))
            %                 currentOVP = 0;
            %             end
            %             ovps(id1,id2) = ovps(id1,id2) + currentOVP;
            %         op    %area_ratio(id1,id2) = area_ratio(id1,id2)+a1/a2;
            %             c_diff = curCentroids(i2,:)-curCentroids(i1,:);
            %             centroidRelations{id1,id2}{end+1} = c_diff;
            %             area_ratio{id1,id2}(end+1) = a1/a2;
            %             if (debug_ && id1 == 3 && id2 == 1)
            %                 quiver(curCentroids(i1,1),curCentroids(i1,2),c_diff(1),c_diff(2),0,'g','LineWidth',2,...
            %                     'MaxHeadSize',1);
            %                 pause;
            %             end
        end
        
    end
    %     area_ratio-area_ratio'
    %     pause
    %     co_occurence = co_occurence+co_occurence_local;
end

for i1 = 1:length(partNames)
    for i2 = 1:length(partNames)
        relativeShape(i1,i2).shape = cat(2,relativeShape(i1,i2).shape{:});
    end
end

%
% ovps = ovps./(allCounts+eps);
% % area_ratio = (allCounts>0).*(area_ratio./(allCounts+eps));
% % P_12 = bsxfun(@rdivide,co_occurence,sum(co_occurence,2)); % chance of column given row.
% binaryFactors.centroidRelations = centroidRelations;
% binaryFactors.ovps = ovps;
% binaryFactors.co_occurence = co_occurence;
% binaryFactors.area_ratio = area_ratio;
% binaryFactors.allCounts = allCounts;
%
%
% cr = binaryFactors.centroidRelations;
% for r = 1:size(cr,1)
%     for c = 1:size(cr,2)
%         xy = cr{r,c};
%         binaryFactors.area_ratio{r,c} = binaryFactors.area_ratio{r,c}(:);
%         if (isempty(xy))
%             continue;
%         end
%         xy = cat(1,xy{:});
%         xy = normalize_vec(xy')';
%         binaryFactors.centroidRelations{r,c} = xy;
%     end
% end
%
%     function [x,y] = getPolygonCenter(x,y)
%         x = round(x);
%         y = round(y);
%         xmin = min(x);
%         ymin = min(y);
%         x = x-xmin+1;
%         y = y-ymin+1;
%         z = poly2mask(x,y,max(y),max(x));
%         [yy,xx] = find(z);
%         yy = mean(yy);
%         xx = mean(xx);
%         x = xx + xmin -1;
%         y = yy + ymin -1;
%     end
% end
%
binaryFactors.relativeShape = relativeShape;

