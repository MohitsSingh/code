function [ res ] =  compute_seg(net_conv,data_permute,im);


%
if (length(size(data_permute))==3)
    net_conv.blobs('data').reshape([size(data_permute) 1]);
else
    net_conv.blobs('data').reshape(size(data_permute));
end
net_conv.blobs('data').set_data(data_permute);
net_conv.forward_prefilled();
prob = net_conv.blobs('upsample1').get_data();
prob=permute(prob,[2 1 3 4]);
[m,res]=max(prob,[],3);
res=imResample(res,size2(im),'nearest');
end

