function feature_map_gen(nInput,nOutput)

--[[
Takes feature maps as an input and outputs downsampled flowmap (which will be used for loss function) 
along with a table with upsampled flowmap and feature map that will be concatenated with precedent feature map.
outputs will have this structure :
{flowmap,{upsampled flowmap, upsample feature map}}
]]--
  local a = nn.Sequential()
  local b = nn.ConcatTable()
  
  b:add(nn.Sequential():add(nn.SpatialConvolution(nInput,2,3,3,1,1,1,1)))
  b:add(nn.Identity())
  
  local c = nn.ConcatTable()
  local f = nn.Sequential()
  local d = nn.ParallelTable()
  d:add(nn.SpatialFullConvolution(2,2,4,4,2,2,1,1))
  local e = nn.Sequential()
  e:add(nn.SpatialFullConvolution(nInput,nOutput,4,4,2,2,1,1))
  e:add(nn.SpatialBatchNormalization(nOutput,1e-3))
  e:add(nn.LeakyReLU(0.1,true))
  d:add(e)
  
  f:add(d)
  f:add(nn.JoinTable(2))
  
  c:add(nn.SelectTable(1))
  c:add(f)
  
  a:add(b)
  a:add(c)
  
  return a
  
end

function advanced_concat(flowmaps_nb)
--[[
takes feature tables as an input and outputs reorganised with flowmaps along with concatenated featuremap from conv and upconv
input : { flows , {flow, upconv}, conv}
output : { flows, conv-upconv}
]]--
local a = nn.Sequential()
a:add(nn.FlattenTable())
local b = nn.ConcatTable()
b:add(nn.NarrowTable(1,flowmaps_nb))
local c = nn.Sequential()
c:add(nn.NarrowTable(flowmaps_nb+1,2))
c:add(nn.JoinTable(2))
b:add(c)
a:add(b)
return(a)

end

function createModel(nGPU)

   local net = nn.Sequential()
   net:add(nn.SpatialConvolution(6,64,7,7,2,2,3,3))
   net:add(nn.SpatialBatchNormalization(64,1e-3))
   net:add(nn.LeakyReLU(0.1,true))
   net:add(nn.SpatialConvolution(64,128,5,5,2,2,2,2))
   net:add(nn.SpatialBatchNormalization(128,1e-3))
   net:add(nn.LeakyReLU(0.1,true))
   local conv3 = nn.ConcatTable()
   net:add(conv3)
   local conv3_1 = nn.Sequential()
   local conv3_2 = nn.Identity()
   conv3:add(conv3_1)
   conv3:add(conv3_2)
   conv3_1:add(nn.SpatialConvolution(128,256,5,5,2,2,2,2))
   conv3_1:add(nn.SpatialBatchNormalization(256,1e-3))
   conv3_1:add(nn.LeakyReLU(0.1,true))
   conv3_1:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
   conv3_1:add(nn.SpatialBatchNormalization(256,1e-3))
   conv3_1:add(nn.LeakyReLU(0.1,true))
   local conv4 = nn.ConcatTable()
   conv3_1:add(conv4)
     local conv4_1 = nn.Sequential()
     local conv4_2 = nn.Identity()
     conv4:add(conv4_1)
     conv4:add(conv4_2)
     conv4_1:add(nn.SpatialConvolution(256,256,3,3,2,2,1,1))
     conv4_1:add(nn.SpatialBatchNormalization(256,1e-3))
     conv4_1:add(nn.LeakyReLU(0.1,true))
     conv4_1:add(nn.SpatialConvolution(256,512,3,3,1,1,1,1))
     conv4_1:add(nn.SpatialBatchNormalization(512,1e-3))
     conv4_1:add(nn.LeakyReLU(0.1,true))
     local conv5 = nn.ConcatTable()
     conv4_1:add(conv5)
       local conv5_1 = nn.Sequential()
       local conv5_2 = nn.Identity()
       conv5:add(conv5_1)
       conv5:add(conv5_2)
       conv5_1:add(nn.SpatialConvolution(512,512,3,3,2,2,1,1))
       conv5_1:add(nn.SpatialBatchNormalization(512,1e-3))
       conv5_1:add(nn.LeakyReLU(0.1,true))
       conv5_1:add(nn.SpatialConvolution(512,512,3,3,1,1,1,1))
       conv5_1:add(nn.SpatialBatchNormalization(512,1e-3))
       conv5_1:add(nn.LeakyReLU(0.1,true))
       local conv6 = nn.ConcatTable()
       conv5_1:add(conv6)
         local conv6_1 = nn.Sequential()
         local conv6_2 = nn.Identity()
         conv6:add(conv6_1)
         conv6:add(conv6_2)
         conv6_1:add(nn.SpatialConvolution(512,1024,3,3,2,2,1,1))
         conv6_1:add(nn.SpatialBatchNormalization(1024,1e-3))
         conv6_1:add(nn.LeakyReLU(0.1,true))
         conv6_1:add(nn.SpatialConvolution(1024,1024,3,3,1,1,1,1))
         conv6_1:add(nn.SpatialBatchNormalization(1024,1e-3))
         conv6_1:add(nn.LeakyReLU(0.1,true))
         conv6_1:add(feature_map_gen(1024,512))
       
       conv5_1:add(advanced_concat(1))
       local join6 = nn.ParallelTable()
       conv5_1:add(join6)
       
       join6:add(nn.Identity())
       join6:add(feature_map_gen(1026,256))
       
     conv4_1:add(advanced_concat(2))
     local join5 = nn.ParallelTable()
     conv4_1:add(join5)
     
     join5:add(nn.Identity())
     join5:add(feature_map_gen(770,128))

   conv3_1:add(advanced_concat(3))
   local join4 = nn.ParallelTable()
   join4:add(nn.Identity())
   join4:add(feature_map_gen(386,64))
   conv3_1:add(join4)
   
   net:add(advanced_concat(4))
   local final = nn.ParallelTable()
   final:add(nn.Identity())
   final:add(nn.SpatialConvolution(194,2,3,3,1,1,1,1))
   net:add(final)
   net:add(nn.FlattenTable())
   
   net.downSample = 4
   net.scales = 5
   net.imageSize = {512,384}
   net.imageCrop = {448,320}
   
   return net
end
