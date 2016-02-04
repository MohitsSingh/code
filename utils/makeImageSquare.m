function I = makeImageSquare(I,padval)
    sz = size2(I);
    if nargin < 2
        padval = .5;
    end
    r = max(sz);
    sz = r-sz;
    sz_pre = floor(sz/2);
    sz_post = sz-sz_pre;
    I = padarray(I,sz_pre,padval,'pre');
    I = padarray(I,sz_post,padval,'post');
end
