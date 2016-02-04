function svms = learn_fine_grained(conf,trues,falses,locWeight)
% secondaryClustering(conf,top_dets(2),test_set,gt_labels);
x1 = {};x2 = {};
span_ = 1;
A = cat(2,trues{span_});
conf.detection.params.detect_min_scale = 1;
conf.detection.params.init_params.sbin = 8;
conf.features.winsize = 8;
for ii = 1:length(A)
    disp(['calculating descriptors for first set: %' num2str(100*ii/length(A))]);
    if (ischar(A{ii}))
        I1 = toImage(conf,getImagePath(conf,A{ii}));
    else
        I1 = A{ii};
    end
    I1 = imresize(I1,[128 128]);
    [X1,uus,vvs,scales,t] = allFeatures(conf,I1);
    [ bbs ] = uv2boxes( conf,uus,vvs,scales,t );
    
    
    overlaps = boxesOverlap(bbs);
    [mm,nn] = find(overlaps>.5);
    
    removed = false(size(overlaps,1),1);
    for ki = 1:length(nn)
        % only remove a box which overlaps with a box which
        % hasn't been removed yet.
        if (~removed(mm((ki))))
            removed(nn(ki)) = true;
        end
    end
    
    X1 = X1(:,~removed);
    bbs = bbs(~removed,:);
    
    xy = locWeight*boxCenters(bbs)';
    X1 = [X1;xy];    
    x1{ii} = single(X1);
end

B = cat(2,falses{span_});
conf.detection.params.detect_min_scale = 1;
for ii = 1:min(30,length(B))
    disp(['calculating descriptors for second set: %' num2str(100*ii/length(B))]);
    if (ischar(B{ii}))
        I1 = toImage(conf,getImagePath(conf,B{ii}));
    else
        I1 = B{ii};
    end
    I1 = imresize(I1,[128 128]);
    [X1,uus,vvs,scales,t] = allFeatures(conf,I1);
    [ bbs ] = uv2boxes( conf,uus,vvs,scales,t );
    xy = locWeight*boxCenters(bbs)';
    X1 = [X1;xy];
    
    x2{ii} = single(X1);
end

x1 = cat(2,x1{:});
x2 = cat(2,x2{:});
vecPerImage = size(x1,2)/length(A);

svms = struct;
for q = 1:vecPerImage
    q
    % try to train an svm for all of the features from the same
    % location...
    x1_ = x1(:,q:vecPerImage:end);
    x2_ = x2(:,q:vecPerImage:end);
    
    %     x1_ = vl_homkermap(x1_, 1, 'kchi2', 'gamma', .5) ;
    %     x2_ = vl_homkermap(x2_, 1, 'kchi2', 'gamma', .5) ;
    
    % learn lda...
%     Input = [x1_ x2_]';
%     Target = zeros(1,size(Input,1));
%     Target(size(x1_,2)+1:end) = 1;
%     W = LDA(Input,Target');%,Priors)
    
    
% %     [ws,b,sv] = train_classifier(x1_,x2_,.01,1);
% %     svms(q).ws = ws;
% %     svms(q).b = b;
% %     svms(q).nsv = size(sv,2)/(size(x2_,2) +size(x1_,2));
    svms(q).pos = x1_;
    svms(q).neg = x2_;
end

    function [ws,b,sv] = train_classifier(pos_samples,neg_samples,C,w1);
        npos = size(pos_samples,2);
        y = ones(npos,1);
        y = [y;-ones(size(neg_samples,2),1)];
        
        ss = double([pos_samples,neg_samples])';
        
        % svm C, pos weight according to exemplarsvm, but with c = .1
        svm_model = svmtrain(y, ss,sprintf(['-s 0 -t 0 -c' ...
            ' %f -w1 %.9f -q'], C, w1));
        
        % sum support vectors with coefficients
        w = svm_model..Label(1)*svm_model.SVs'*svm_model.sv_coef;
        
        ws = w(:);
        b = svm_model.rho;
        sv = svm_model.SVs';
    end
end