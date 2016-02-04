function [prec,rec] = calc_roc(VOCopts,trainIDs,unary)
%CALC_ROC Summary of this function goes here
%   Detailed explanation goes here
    
    scores = {};
    trues = {};
    

    for k= 1:length(trainIDs)
        if (mod(k,50)==0)
            disp(k);
        end
        fg_mask_ = imread(sprintf(VOCopts.seg.clsimgpath,trainIDs{k}));
        dc_mask = fg_mask_ ==255;
        fg_mask = fg_mask_ > 0 & ~dc_mask;
%         bg_mask = ~fg_mask_ & ~dc_mask;
        
        unary{k} = imresize(unary{k},.25,'nearest');
        dc_mask = imresize(dc_mask,.25,'nearest');
        fg_mask = imresize(fg_mask,.25,'nearest');
        
        
        
        scores{k} = unary{k}(~dc_mask);                        
        trues{k} = fg_mask(~dc_mask);
        
%         [r,ir] = sort(scores{k});
%         plot(r,trues{k}(ir));
                
%         scores{k} = scores{k}(1:16:end);
%         trues{k} = trues{k}(1:16:end);
    end  
    
    scores = single(cat(1,scores{:}));
    trues = (cat(1,trues{:}));
    [~,is] = sort(scores,'descend');
    trues = trues(is);
    trues = single(trues);
    tp = cumsum(trues);
    fp = cumsum(~trues);
    npos = sum(trues);
    rec = tp/npos;
    prec = tp./(tp+fp);
    ap = VOCap(rec,prec);
end

