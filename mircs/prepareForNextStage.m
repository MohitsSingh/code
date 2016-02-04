function prepareForNextStage(results,imgs,dataDir,tmpDir,curStageParams,top_k_false)

for k = 1:length(results)
    k
    if (isempty(results(k).labels))
        continue
    end
    %     break
    orig_img = imgs(k);
    R = j2m(tmpDir,orig_img);
    if (exist(R,'file'))
        continue
    end
    resPath = j2m(dataDir,orig_img);
%         if (orig_img.classID~=9),continue,end
    %L = load(resPath,'feats','moreData');
    L = load(resPath,'feats','moreData');
    decision_values = results(k).decision_values;
    [r,ir] = sort(decision_values,2,'descend');
    % find the top k false elements
    ir = ir(:,1:min(top_k_false,length(ir)));
    ovp  = results(k).ovps;
    false_ovp = find(ovp < curStageParams.learning.negOvp);
    sel_neg = intersect(ir(:),false_ovp);
    sel_pos = find(ovp > curStageParams.learning.posOvp & ~[L.feats.is_gt_region]);
    toKeep = union(sel_pos,sel_neg);
    moreData  = L.moreData;
    save(R,'moreData','toKeep');
end