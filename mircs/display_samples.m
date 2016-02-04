function display_samples(samples)
    for k = 1:size(samples,4)
        s = squeeze(samples(:,:,:,k));
        s = cat(3,s,zeros(dsize(s,1:2)));
        V = hogDraw(s,15,1);
        clf; imagesc(V); axis image; pause(.1);
    end
end