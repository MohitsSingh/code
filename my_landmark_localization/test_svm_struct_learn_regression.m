function test_svm_struct_learn_regression
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

x = -2:.1:2;
y = (.1*x).^2-.1*x;

for i=1:length(x)
    patterns{i} = x(i);
    labels{i}   = y(i);    
end

% ------------------------------------------------------------------
%                                                    Run SVM struct
% ------------------------------------------------------------------

parm.patterns = patterns ;
parm.labels = labels ;
parm.lossFn = @lossCB ;
parm.constraintFn  = @constraintCB ;
parm.featureFn = @featureCB ;
parm.dimension = 5 ;
parm.verbose = 0 ;
model = svm_struct_learn(' -c .01 -o 1 -v 1 ', parm) ;
w = model.w ;

% ------------------------------------------------------------------
%                                                              Plots
% ------------------------------------------------------------------

figure(1) ; clf ; hold on ;
x = [patterns{:}] ;
y = [labels{:}] ;
% plot(x,y,'r.');

[xx,yy] = meshgrid(-1:.01:1,-1:.01:1);

%psi = featureCB(parm, xx, yy);
psis = {};
parm.verbose = 0;
for u = 1:length(xx(:))
    psis{u} = featureCB(parm,xx(u),yy(u));
end
% 
% psis = {};
% parm.verbose = 0;
% for u = 1:length(x(:))
%     psis{u} = featureCB(parm,x(u),y(u));
% end

psis = cat(2,psis{:});

z = w'*psis;
z = (reshape(z,size(xx)));
[m,im] = max(z);
figure,plot(xx(1,1:end),yy(im,1))
ZZ = bsxfun(@rdivide,z,max(abs(z)));
imagesc2(ZZ);

%imagesc2

% plot(x(1, y>0), x(2,y>0), 'g.') ;
% plot(x(1, y<0), x(2,y<0), 'r.') ;
% set(line([0 w(1)], [0 w(2)]), 'color', 'y', 'linewidth', 4) ;
% % xlim([-3 3]) ;
% ylim([-3 3]) ;
% set(line(10*[w(2) -w(2)], 10*[-w(1) w(1)]), ...
%     'color', 'y', 'linewidth', 2, 'linestyle', '-') ;
% axis equal ;
set(gca, 'color', 'b') ;
w
end

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
delta = abs(ybar - y) ;
if param.verbose
    fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
end
end

function psi = featureCB(param, x, y)
psi = [y ;...
     y * x ;...
     y * x^2 ;...
     y * x^3 ;...
     - 0.5 * y^2] ;
psi = sparse(psi) ;

if param.verbose
    fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
        x, y, full(psi(1)), full(psi(2))) ;
end
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
w = model.w ;
z = w(1) + w(2) * x + w(3) * x.^2 + w(4) * x.^3 ;
yhat = [] ;
if w(5) > 0
    yhat = [z - 1, z + 1] / w(5) ;
    yhat = max(min(yhat, 1),-1) ;
end
yhat = [yhat, -1, 1] ;

aloss = @(y_) abs(y_ - y) + z * y_ - 0.5 * y_.^2 * w(5) ;
[drop, worse] = max(aloss(yhat)) ;
yhat = yhat(worse) ;
% yhat = rand(1);

if param.verbose
    fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
        model.w, x, y, yhat) ;
end
end
