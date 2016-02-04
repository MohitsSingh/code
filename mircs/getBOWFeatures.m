function hists = getBOWFeatures(conf,bowModels,images,masks,bowImages)
if (nargin < 4)
    masks = [];
end
if (nargin < 5)
    feats = [];
    
    hists = {};
    for k = 1:length(images)
        k/length(images)
        [F,D] = vl_phow(images{k},'color','RGB','step',1);
        bowModels.vocab;
        bins = minDists(single(D),single(bowModels.vocab),5000);
        bowImages = makeBowImage(images{k},F,bins);
        clf; subplot(1,2,1); imagesc(images{k}); axis image;
        subplot(1,2,2);imagesc(bowImages(:,:,1)); axis image;
        pause;
        masks= true(dsize(images{k},[2 1]));
        hists{k} = getImageDescriptor([bowModels],masks,{bowImages});
        
    end
    hists = cat(2,hists{:});
    hists = vl_homkermap(hists, 1, 'kchi2', 'gamma', 1) ;
end

%hists = getHists(conf,bowModels,images,masks,bowImages);
% hists = sparse(double(hists));
% feats = sparse(3*size(hists,1),size(hists,2));
% for k = 1:size(hists,2)
%     feats(:,k) = sparse(double(vl_homkermap(hists(:,k), 1, 'kchi2', 'gamma', .5)));
% end
% feats = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;

%if (any(isnan(hists(:))))
% f = find(isnan(sum(hists)));
% % if (any(f))
% %     warning(['there were NaN element in features : ' num2str(f)]);
% % end
end
%     hists(isnan(hists)) = 0;
%end