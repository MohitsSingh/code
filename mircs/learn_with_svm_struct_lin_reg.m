function w=learn_with_svm_struct_lin_reg(train_feats,train_labels)
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
%parm.featureFn = @featureCB ;
parm.kernelFn= @kernelCB ;
%parm.dimension = 3 ;
parm.verbose = 1 ;
model = svm_struct_learn(' -c 1.0 -o 1 -v 1 -t 4 ', parm) ;
 %   parm.patterns = patterns ;
%   parm.labels = labels ;
%   parm.lossFn = @lossCB
%   parm.constraintFn  =@constraintCB ;
%   parm.kernelFn = @kernelCB ;
%   parm.verbose = 1 ;
%   model = svm_struct_learn(' -c 1.0 -o 1 -v 1 -t 4 ', parm) ;
%   

w = cat(2, model.svPatterns{:}) * (model.alpha .* cat(1, model.svLabels{:})) / 2 ;

% ------------------------------------------------------------------
%                                                              Plots
% ------------------------------------------------------------------

% % figure(1) ; clf ; hold on ;
% % x = [patterns{:}] ;
% % y = [labels{:}] ;
% % plot(x(1, y>0), x(2,y>0), 'g.') ;
% % plot(x(1, y<0), x(2,y<0), 'r.') ;
% % set(line([0 w(1)], [0 w(2)]), 'color', 'y', 'linewidth', 4) ;
% % xlim([-3 3]) ;
% % ylim([-3 3]) ;
% % set(line(10*[w(2) -w(2)], 10*[-w(1) w(1)]), ...
% %     'color', 'y', 'linewidth', 2, 'linestyle', '-') ;
% % axis equal ;
% % set(gca, 'color', 'b') ;
% % w
end

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
delta = (y-ybar)^2;
if param.verbose
    fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
end
end

function k = kernerlCB(param, x, y, xp, yp)
k = vl_homkermap([x;y],1)'*vl_homkermap([xp;yp],1);

%psi = sparse(double([x*y;y]));
% if param.verbose
%fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
%    x, y, full(psi(1)), full(psi(2))) ;
end

function ybar = constraintCB(param, model, x, y)
if size(model.svPatterns, 2) == 0
    %w = zeros(size(x)) ;
    w = rand(6,1);
else
    w = cat(2, model.svPatterns{:}) * ...
        (model.alpha .* cat(1, model.svLabels{:})) / 2 ;
end

yz = -1:.1:1;
x  =repmat(x,1,length(yz));
x = [x;row(yz)];
x = vl_homkermap(x,1);
u = w'*x;
[m,im] = max(u);
ybar = yz(im);


end

% % % function yhat = constraintCB(param, model, x, y)
% % % % slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% % % % margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
% % % %if dot([x;y], model.w) > 1, yhat = y ; else yhat = - y ; end
% % % w = model.w;
% % % f = @(y_hat) (y-y_hat)^2 + w'*([x*y_hat;y_hat]-[x*y;y]);
% % % f1 = f(pi/2);
% % % f2 = f(-pi/2);
% % %
% % % z = -5:.01:5;
% % % p = zeros(size(z));
% % % for u = 1:length(z)
% % %     p(u) = f(z(u));
% % % end
% % %
% % % [m,im] = max(p);
% % % yhat = z(im);
% % %
% % % % if (f1>f2)
% % % %     yhat = pi/2;
% % % % else
% % % %     yhat = -pi/2;
% % % % end
% % % % if param.verbose
% % % %     fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
% % % %         model.w, x, y, yhat) ;
% % % end
% end