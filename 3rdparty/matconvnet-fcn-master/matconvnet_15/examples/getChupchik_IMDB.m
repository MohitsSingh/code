function imdb = getChupchik_IMDB()
% --------------------------------------------------------------------
% nSamples = 600;

B = 10;
nSamples = B*6;
nTrain = B*4;
nVal = B;
nTest = B;

set = [ones(1,nTrain) 3*ones(1,nVal)];
% create data of the following form : squares + noise + possibly a pixel
% "hanging" from the squares.
p_square = .5;
p_pixel_given_square = .5;
noise_magnitude = 24;
data_magnitude = 128;
imgSize = [28 28];
square_size = 5;
chupchik_size = 1;
chupchik_size_x = 1;
h = floor(chupchik_size/2);
data = zeros([imgSize,1,nSamples],'single');
rands = rand(2,nSamples);
labels = -1*ones(1,1,2,nSamples);
for t = 1:nSamples
    z = zeros(imgSize);
    if rands(1,t) >= p_square
        % square start coordinates:
        s0 = ceil(rand(1,2)*16);
        xmin = s0(1);
        ymin = s0(2);
        s1 = s0+4;
        xmax = s1(1);
        ymax = s1(2);
        z(ymin:ymax,xmin:xmax) = data_magnitude;
        labels(1,1,1,t) = 1;
        if rands(2,t) >= p_pixel_given_square
            z(ymin:ymin+h,xmax+7) = data_magnitude;
            labels(1,1,2,t) = 1;
            
            
        end
    end
    z = z+randn(imgSize)*noise_magnitude;
    if (all(rands(:,t)>.5))
%         figure(1);clf; imagesc2(z);drawnow
%         pause
    end
    data(:,:,:,t) = z;
    
end

set = 2*ones(1,nSamples);
set(1:nTrain) = 1;
set(nTrain+1:nTrain+nVal) = 3;
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = labels;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = {'box','box_w_chupchik'};