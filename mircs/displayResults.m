drinking_test = drinking_test(randperm(length(drinking_test)));
% partModels = faceModel;
% partNames = {'face'};
for k = 1:100
    clf;
    clc;
    disp(num2str(k));
    currentID = drinking_test{k};
    [regions,rOvp] = getRegions(conf,currentID,false);
    regionSel = suppresRegions(rOvp,.5);
    regions = regions(regionSel);
%     profile on;
    feats = partModels(1).extractor.extractFeatures(currentID,regions);
%     profile viewer;
    rs = zeros(length(partNames),size(feats,2));
    zs = {};
    I = getImage(conf,currentID);
    for iPart = 1:length(partNames)
        w = partModels(iPart).models.w;
        [res_pred, res] = partModels(iPart).models.test(feats);
        rs(iPart,:) = res;
         
        z = zeros(dsize(I,1:2));
        for q = 1:length(res)
            z(regions{q}) = max(z(regions{q}),res(q));
        end
        zs{iPart} = z;
    end        
    
    %     res = partModels(iPart).model_combined.model.ws' * feats - partModels(iPart).model_combined.b;
    
    rs(isnan(rs)) = -inf;
   
    
    [rr,ir] = sort(rs,2,'descend');
    zz = ceil(sqrt(size(rs,1)));
    for q = 1:5
        clf;
        for iPart = 1:length(partNames)
            subplot(2*zz,zz,iPart);
            imagesc(.8*I.*repmat(regions{ir(iPart,q)},[1 1 3]) + ...
                .2*I.*repmat(~regions{ir(iPart,q)},[1 1 3]));
            axis image;
            title([partNames{iPart} ', ' num2str(q) ' : ',...
                num2str(rr(iPart,q))]);            
            subplot(2*zz,zz,iPart+length(partNames));
            imagesc(zs{iPart}); axis image;
        end
        pause;
    end
    clf;
    %     [r,ir] = sort(res,'descend');
    %     displayRegions(I,regions(ir(1:5)),r(1:5));
end