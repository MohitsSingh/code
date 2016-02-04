function [ S ] = make_topo( S,xx,yy,nn)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    xx = [0 5000];
    yy = [0 5000];
    nn = 100;
end

for t = 1:length(S)
    xy = S(t).xyz(:,1:2);
    xy(:,1) = xy(:,1)-xx(1);
    xy(:,2) = xy(:,2)-yy(1);
    xy(:,1) = nn*xy(:,1)/xx(2);
    xy(:,2) = nn*xy(:,2)/yy(2);
    xy = round(xy);
    xy = max(1,xy);
    xy = min(nn,xy);
    z = zeros(nn);
    for r = 1:size(xy,1)
        z(xy(r,2),xy(r,1)) = max(z(xy(r,2),xy(r,1)),S(t).xyz(r,3));
    end
    S(t).topo = z;
end

end

