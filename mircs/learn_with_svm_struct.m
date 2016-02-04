function w=learn_with_svm_struct(train_feats,train_labels)
% function test_svm_struct_learn
% TEST_SVM_STRUCT_LEARN
%   A demo function for SVM_STRUCT_LEARN(). It shows how to use
%   SVM-struct to learn a standard linear SVM.

randn('state',0) ;
rand('state',0) ;

% ------------------------------------------------------------------
%                                                      Generate data
% ------------------------------------------------------------------

th = pi/3 ;
c = cos(th) ;
s = sin(th) ;

patterns = {} ;
labels = {} ;
for i=1:size(train_feats,2)
    patterns{i} = train_feats(:,i);
    labels{i}   = train_labels(i);
    %patterns{i}(2) = patterns{i}(2) + labels{i} ;
    %patterns{i} = [c -s ; s c] * patterns{i}  ;
end

% ------------------------------------------------------------------
%                                                    Run SVM struct
% ------------------------------------------------------------------

parm.patterns = patterns ;
parm.labels = labels ;
parm.lossFn = @lossCB ;
parm.constraintFn  = @constraintCB ;
parm.featureFn = @featureCB ;
parm.endIterationFn = @iterCB ;
parm.dimension = 64*31*3+2 ;
parm.verbose = 1 ;
model = svm_struct_learn(' -c .1  -o 2 -v 1 -e 0.0001 ', parm) ;
w = model.w ;

% ------------------------------------------------------------------
%                                                              Plots
% ------------------------------------------------------------------

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

    function delta = lossCB(param, y, ybar)
        delta = abs(ybar - y) ;
    end
% 
%     function psi = featureCB(param, x, y)
%         psi = [y ;
%             y * x ;
%             0*y * x^2 ;
%             0*y * x^3 ;
%             - 0.5 * y^2] ;
%         psi = sparse(psi) ;
%     end

% % %     function yhat = constraintCB(param, model, x, y)
% % %         w = model.w ;
% % %         z = w(1) + w(2) * x + w(3) * x.^2 + w(4) * x.^3 ;
% % %         yhat = [] ;
% % %         if w(5) > 0
% % %             yhat = [z - 1, z + 1] / w(5) ;
% % %             yhat = max(min(yhat, 1),-1) ;
% % %         end
% % %         yhat = [yhat, -1, 1] ;
% % %         
% % %         augLoss = @(y_) abs(y_ - y) + z * y_ - 0.5 * y_.^2 * w(5) ;
% % %         [drop, worse] = max(augLoss(yhat)) ;
% % %         yhat = yhat(worse) ;
% % %     end

    function yhat = constraintCB(param, model, x, y)
        w = model.w ;
        w1 = w(1);
        w2_4 = w(2:2+64*31*3-1)'*[x;x.^2;x.^3];
        %w2 = w(2:1117).*x;
        %w3 = w(1118:2233).*(x.^2)
        %w4 = w(2234:3349).*(x.^3);
        w5 = -.5*w(end);
        
        f = @(z) (z-y)^2+(1+w1+w2_4)*(z-y)-.5*w5*(z^2-y^2);
        u = -1:.1:1;
%         u = [-1 1];
        ff = zeros(size(u));
        for t = 1:length(ff)
            ff(t) = f(u(t));
        end
        [m,worse] = max(ff);
        yhat = u(worse);
        if (~ismember(yhat,[-1 1]))
            p = 0;
        end
%         
%         %z = w(1) + w(2) * x + w(3) * x.^2 + w(4) * x.^3 ;
%         
%         if w(5) > 0
%             yhat = [z - 1, z + 1] / w(5) ;
%             yhat = max(min(yhat, 1),-1) ;
%         end
%         yhat = [yhat, -1, 1] ;
%         
%         augLoss = @(y_) abs(y_ - y) + z * y_ - 0.5 * y_.^2 * w(5) ;
%         [drop, worse] = max(augLoss(yhat)) ;
%         yhat = yhat(worse) ;
    end


    function cont = iterCB(param, model)
        iter = iter + 1 ;
        w = model.w ;
        z = w(1) + w(2) * xr + w(3) * xr.^2 + w(4) * xr.^3 ;
        F = yr'*z - 0.5 * yr'.^2 * ones(size(z)) * w(5) ;
        F_ = bsxfun(@plus, F, -max(F)) ;
        
        figure(1) ; clf ;
        subplot(1,3,1) ;
        plot(cell2mat(patterns), cell2mat(labels), 'b*') ;
        z = w(1) + w(2) * xr + w(3) * xr.^2 + w(4) * xr.^3 ;
        y = z / w(5) ;
        hold on ;
        plot(xr,y,'g','linewidth',3) ;
        ylim([-1 1]) ;
        xlim([min(xr),max(xr)]) ;
        title(sprintf('cutting plane iteration %d', iter)) ;
        
        subplot(1,3,2) ;
        imagesc(xr,yr,F) ;
        set(gca,'ydir','normal') ;
        title('scoring function') ;
        
        subplot(1,3,3) ;
        imagesc(xr,yr,F_) ;
        title('column rescaled') ;
        set(gca,'ydir','normal') ;
        drawnow ;
        
        if print_figures
            vl_figaspect(3) ;
            vl_printsize(1) ;
            print(1, sprintf('f/cut-%d.pdf', iter), '-dpdf') ;
        end
        cont = false ;
    end
end

