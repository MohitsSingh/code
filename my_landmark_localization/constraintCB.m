function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
% disp('constraint');
% given an image representation, find the worst possible configuration for
% the locations y of the landmarks.

nPts = param.nPts;
w_unary = model.w(1:nPts);
w_edges = model.w(nPts+1:end);
w_prior = [];
% w_unary = w(1);
% w_edges = w(2);

[yhat] = apply_graphical_model(param,x,...
    w_unary,w_prior,w_edges,y);

end