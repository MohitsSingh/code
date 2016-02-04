figure(3); clf;

for ii = 1:size(subsets,1)
ii
    labels_to_block = find(subsets(ii,:));
    perfName = concat_names(class_names,labels_to_block,'perf_blk_');    
    subplot(1,3,1);
    [perfs,diags] = test_net_perf(expDir,50,imdb,train,val,test,{'none','face','hand','obj'},labels_to_block ,...
        perfName);       
    figure(3);
    subplot(2,4,ii);
    imagesc(perfs.cm_n);%;axis equal;
%     title(perfName,'interpreter','none');
dpc
end
