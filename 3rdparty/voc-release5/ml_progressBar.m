function ml_progressBar(k,n, prefixMessage)
% n: total
% k: current progress
% By: Minh Hoai Nguyen (minhhoai@robots.ox.ac.uk)
% Created: 24-Mar-2013
% Last modified: 24-Mar-2013

if ~exist('prefixMessage', 'var')
    prefixMessage = 'Progress';
end;

nDigit = length(sprintf('%d',n));
format = sprintf('%%%dd/%d (%%6.2f%%%%)', nDigit, n);
delFormat = repmat('\b', 1, 2*nDigit+11);

if k==1
    fprintf([prefixMessage ' ' format], k, 100*k/n);
elseif k == n
    fprintf([delFormat, format, '\n'], k, 100*k/n);
    %fprintf(['\n', prefixMessage ' ' format], k, 100);
else
    fprintf([delFormat, format], k, 100*k/n);
end;


% This function is from Yusuf
% if (k==1)
%     fprintf('\nProgress %05.2f %%',100*k/n);
% elseif (k==n)
%     fprintf('\b\b\b\b\b\b\b\b %05.2f %%\n',100*k/n);
% else    
%     fprintf('\b\b\b\b\b\b\b\b %05.2f %%',100*k/n);
% end
% 
% end
