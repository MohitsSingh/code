function r = box2Region( bb ,sz, d, u)
%r = box2Region(bb,sz,d) creates masks the same shape as the given boxes in
%bb, bounded to an image if size sz. Results are double (rather than
%logical) if d is specified and true. 
% sz may be an (at least) 2d matrix in which sz in interpreted as size2(sz)

if nargin < 2
    sz = fliplr(ceil(bb(:,3:4)));
elseif numel(sz) > 2
    sz = size2(sz);
end
if nargin < 3
    d = false;
end
if nargin < 4
    u = false;
end
if size(bb,1) > 1
    r = {};
    for t = 1:size(bb,1)
        r{t} = box2Region(bb(t,:),sz,d);
    end
    
    if u && length(r) > 1
        rr = r{1};
        for t = 2:length(r)
            rr = rr | r{t};
        end
        r = rr;
    end
    return
end

r = poly2mask2(box2Pts(bb),sz);
if nargin == 3 && d
    r = double(cat(3,r,r,r));
end
end