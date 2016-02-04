function [ovp,ints,uns] = regionsOverlap2(regions1,regions2)
if (nargin == 1)
    regions2 = regions1;
end
if (~iscell(regions1)), regions1 = {regions1}; end
if (~iscell(regions2)), regions2 = {regions2}; end

% % for k = 1:length(regions2) % expand polygon to mask, if required
% %     curRegion = regions2{k};          
% % end

n1 = length(regions1);
n2 = length(regions2);

if (n1 > n2)
    [ovp,ints,uns] = regionsOverlap2(regions2,regions1);
    ovp = ovp';ints = ints'; uns = uns';
    return;
end

ovp = zeros(length(regions1),length(regions2));
ints = zeros(length(regions1),length(regions2));
uns = zeros(length(regions1),length(regions2));

regions1 = cellfun2(@(x) x(:), regions1); regions1 = cat(2,regions1{:});
regions2 = cellfun2(@(x) x(:), regions2); regions2 = cat(2,regions2{:});

% sums1 = cellfun(@nnz,regions1);
% sums2 = cellfun(@nnz,regions2);
sums1 = sum(regions1,1);
sums2 = sum(regions2,1);


for i1 = 1:size(regions1,2)
    %     ii
    rA = regions1(:,i1);
    %     rA = find(rA);
%     f = find(rA);
    for i2 = 1:size(regions2,2)
        %         if (boxOverlaps(i1,i2) == 0)
        %             continue;
        %         end
        rB = regions2(:,i2);
        int = nnz(rB(rA));
        ints(i1,i2) = int;
        uns(i1,i2) = sums1(i1)+sums2(i2)-int;
    end
end
ovp = ints./(uns+eps);
ovp(ints==0) = 0;


    function boxes  = getRegionBoxes(regions)
        
        boxes = zeros(length(regions),4);
        for iBox = 1:length(regions)
            [y,x] = find(regions{iBox});
            boxes(iBox,:) = [min(x) min(y) max(x) max(y)];
        end
    end
end