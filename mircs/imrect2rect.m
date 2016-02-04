function r = imrect2rect(r)
if (iscell(r))
    r = cat(1,r{:});
end
r(:,3) = r(:,3)+r(:,1);
r(:,4) = r(:,2)+r(:,4);
end