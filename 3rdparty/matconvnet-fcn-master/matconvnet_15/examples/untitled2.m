[z,iz] = sort(curScores,'descend');

for it = 1:length(z)
    it
    k = iz(it);
    curImg = uint8(test_images(:,:,:,k)+128);
    clf; imagesc2(curImg);
    dpc
end

