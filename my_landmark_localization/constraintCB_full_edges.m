function yhat = constraintCB_full_edges(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
% disp('constraint');
% given an image representation, find the worst possible configuration for
% the locations y of the landmarks.

nPts = param.nPts;
n = (31*(param.windowSize/param.cellSize)^2)*nPts;
w_unary = model.w(1:n);
w_edges = model.w(n+1:end);

% w_unary = w(1);
% w_edges = w(2);

[yhat] = apply_graphical_model(param,x,...
    w_unary,w_edges,y);

end