function weights = ranklearning(x, C, optype, precision)

%%% This function solves an optimization problem (see below) with various
%%% conditions (L1/L2) and different solvers.

%%% Copyright @ Chunhui Gu, January 2008

% Inputs
% x is the difference-of-vectors matrix
% C is the tradeoff parameter
% The opt. problem is 
%     min r-norm(w) + C*sum(s(i))
%     st: s(i) >= 0, dot(w, x(:,i)) >= 1 - s(i), w >= 0

[m, n] = size(x);
p = [ones(m,1); C*ones(n,1)];
H = zeros(m+n); H(1:m,1:m) = eye(m);
f = [zeros(m,1); C*ones(n,1)];
A = [x',eye(n)];
b = ones(n,1);

switch optype
    case 'l1'
        
        [w_s,fval,exitflag] = linprog(p,-A,-b,[],[],zeros(m+n,1));
        %options = optimset('LargeScale','off','Simplex','on');
        %[w_s,fval,exitflag] = linprog(p,-A,-b,[],[],zeros(m+n,1),[],[],options);
        if exitflag == 1,
            weights = w_s(1:m);
        else
            weights = zeros(m,1);
            %error('opt. not converge!');
        end;
        
    case 'l2'
        
        weights = ranklearning_matlab(x,C,precision);
        
%         [w_s,fval,exitflag] = quadprog(H,f,-A,-b,[],[],zeros(m+n,1));
%         if exitflag == 1,
%             weights = w_s(1:m);
%         else
%             weights = 0;
%             error('opt. not converge!');
%         end;
        
%         cvx_begin
%             variable eta(m+n)
%             minimize( 1/2*eta'*H*eta + f'*eta )
%             subject to
%                 A*eta >= b
%                 eta >= zeros(m+n,1)
%         cvx_end
%         weights = eta(1:m);
            
    case 'l1cvx'
        
        weights = runcvx(x, C*ones(n,1), 'l1');

    case 'l2cvx'
        
        weights = runcvx(x, C*ones(n,1), 'l2');
        
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Andrea's L2 solver in Matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function weights = ranklearning_matlab(x,C,precision)

[m,n] = size(x);
MAXUPDATES = 2000000;
x_sq = sum(x.^2,1);

tau = rand(n,1);
tau_d = x*tau;
mu = max(0, -tau_d);
weights = max(0, tau_d);

numepochs = 0;
numupdates = 0;
epochupdates = 0;

index0 = 1:n;
index = index0;
pt = 1;
while(1)
    
    if isempty(index),
        if (numepochs > 0) && (epochupdates == 0),
            break;
        end;
        index = index0;
        pt = 1;
        numepochs = numepochs + 1;
        epochupdates = 0;
    elseif pt == length(index)+1,
        pt = 1;
    end;
    
    %fprintf(1,'%d/%d\n',pt,length(index));
    
    ind = index(pt);
    vec = x(:,ind);
    testval = weights'*vec;
    if (tau(ind) == 0 && testval >= 1 - precision) || ...
            (tau(ind) > 0 && tau(ind) < C && abs(testval - 1) <= precision) || ...
            (tau(ind) == C && testval <= 1 + precision),
        index(pt) = [];
        continue;
    end;
    
    tau_old = tau(ind);
    tau_d_other = tau_d - tau_old*vec;
    tau_new = (1-(tau_d_other + mu)'*vec)/x_sq(ind);
    tau_new = max(0,min(C,tau_new));
    
    tau_d = tau_d_other + tau_new*vec;
    tau(ind) = tau_new;
    
    mu = max(0,-tau_d);
    weights = max(0,tau_d);
    
    numupdates = numupdates + 1;
    epochupdates = epochupdates + 1;
    pt = pt + 1;
    
    if numupdates > MAXUPDATES,
        break;
    end;
end;

xi = max(0, 1-x'*weights);
primal = 0.5*weights'*weights + C*sum(xi);
dual = -0.5*weights'*weights + sum(tau);
%fprintf(1,'primal = %f, dual = %f.\n',primal, dual);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run L1/L2 norm optimization via CVX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = runcvx(distdiff, C, optype)

[nf,nt] = size(distdiff);   % nf: number of features; nt: number of triplets

cvx_begin
    variable w(nf)
    variable s(nt)
    switch optype
        case 'l1'
            minimize( sum(w) + C'*s )
        case 'l2'
            minimize( 1/2*w'*w + C'*s )          
    end;
    subject to
        s + distdiff'*w >= ones(nt,1);
        w >= zeros(nf,1);
        s >= zeros(nt,1);
cvx_end