
function rnn_demo()
data = fileread('big.txt');

chars = unique(data);
data_size = length(data);
vocab_size = length(chars);

char_to_ix = containers.Map();
for t = 1:length(chars)
    char_to_ix(chars(t)) = t;
end

ix_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
for t = 1:length(chars)
    ix_to_char(t) = chars(t);
end


peekIters = 100;
hidden_size = 100;
seq_length = 25;
learning_rate = 1e-1;

Wxh = randn(hidden_size,vocab_size)*.01;
Whh = randn(hidden_size,hidden_size)*.01;
Why = randn(vocab_size,hidden_size)*.01;
bh = zeros(hidden_size,1);
by = zeros(vocab_size,1);
    function [ param,mparam] = updateParam( param,dparam,mparam)
        %UPDATEPARAM Summary of this function goes here
        %   Detailed explanation goes here
        mparam = mparam + dparam.^2;
        param = param - learning_rate*dparam ./ ((mparam+1e-8).^.5);
        
    end


    function [loss, dWxh, dWhh, dWhy, dbh, dby, V] = lossFun( inputs,targets,hprev )
        xs = {};
        hs = {};
        ys = {};
        ps = {};
        
        loss = 0;
        V = [];
        %         hs{end+1} = hprev;
        for t = 1:length(inputs)
            curX = zeros(vocab_size,1);
            curX(inputs(t)) = 1;
            xs{t} = curX;
            if t == 1
                prev_h = hprev;
            else
                prev_h = hs{t-1};
            end
            curH = tanh(Wxh*curX + Whh*prev_h + bh);
            hs{t} = curH;
            curY = Why*curH+by;
            ys{t} = curY;
            ps{t} = exp(curY)/sum(exp(curY));
            curP = ps{t};
            loss = loss - log(curP(targets(t)));
        end
        V = curH;
        
        dWxh = zeros(size(Wxh));
        dWhh = zeros(size(Whh));
        dWhy = zeros(size(Why));
        dbh = zeros(size(bh));
        dby = zeros(size(by));
        dhnext = zeros(size(hprev));
        
        for t = length(inputs):-1:1
            dy = ps{t};
            dy(targets(t)) = dy(targets(t))-1;
            dWhy = dWhy + dy*hs{t}';
            dby = dby+ dy;
            dh = Why'*dy + dhnext;
            dhraw = (1-hs{t}.^2).*dh;
            dbh = dbh + dhraw;
            dWxh = dWxh + dhraw*xs{t}';
            
            if t > 1
                dWhh =dWhh+ dhraw*hs{t-1}';
            else
                dWhh =dWhh+ dhraw*hprev';
            end
            dhnext = Whh'*dhraw;
        end
        
% %         dWxh = clipValues(dWxh,-5,5);
% %         dWhh = clipValues(dWhh,-5,5);
% %         dWhy = clipValues(dWhy,-5,5);
% %         dbh = clipValues(dbh,-5,5);
% %         dby = clipValues(dby,-5,5);
        
        %         V = hs{
        
    end

    function ixes = sample(h,seed_ix,n)
        x = zeros(vocab_size,1);
        x(seed_ix) = 1;
        ixes = {};
        for t = 1:n
            h = tanh(Wxh*x + Whh * h + bh);
            y = Why*h + by;
            p_ = exp(y)/sum(exp(y));
            ix = randsample(vocab_size,1,true,p_);
            x = zeros(vocab_size,1);
            x(ix) = 1;
            ixes{t} = ix;
        end
        ixes = [ixes{:}];
    end

n = 0;
p = 1;

mWxh = zeros(hidden_size,vocab_size);
mWhh = zeros(hidden_size,hidden_size);
mWhy = zeros(vocab_size,hidden_size);
mbh = zeros(hidden_size,1);
mby = zeros(vocab_size,1);


smooth_loss = -log(1.0/vocab_size)*seq_length;

% Batch

while true
    if p+seq_length-1 >= length(data) || n == 0
        hprev = zeros(hidden_size,1);
        p = 1;
    end
    inputs = zeros(seq_length,1);
    targets = zeros(seq_length,1);
    curRange = p:p+seq_length-1;
    for ii = 1:length(curRange)
        inputs(ii) = char_to_ix(data(curRange(ii)));
        targets(ii) = char_to_ix(data(curRange(ii)+1));
    end
    txt = {};
    if mod(n,peekIters)==0
        sample_ix = sample(hprev,inputs(1),200);
        for r = 1:length(sample_ix)
            txt{r} = ix_to_char(sample_ix(r));
        end
        txt = cat(2,txt{:});
        fprintf('\n---------------\n%s\n-------------\n',txt)
    end
    [loss, dWxh, dWhh, dWhy, dbh, dby, hprev] = lossFun( inputs,targets,hprev );
    smooth_loss = smooth_loss * 0.999 + loss * 0.001;
    if mod(n,peekIters) == 0
        fprintf('iter %d, loss: %f\n' , n, smooth_loss); % print progress
        
    end
%     
%     pause
    % update all parameters
    [Wxh,mWxh] = updateParam(Wxh,dWxh,mWxh);
    [Whh,mWhh] = updateParam(Whh,dWhh,mWhh);
    [Why,mWhy] = updateParam(Why,dWhy,mWhy);
    [bh,mbh] = updateParam(bh,dbh,mbh);
    [by,mby] = updateParam(by,dby,mby);
    
    
    p = p+seq_length;
    n = n+1;
end


end


%LOSSFUN Summary of this function goes here
%   Detailed explanation goes here





