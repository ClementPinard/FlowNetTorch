require 'graph'
require 'nngraph'
nngraph.annotateNodes()

function createModel(...)

  local input = -nn.Identity()
  local conv1 = input - nn.SpatialConvolution(6,64,7,7,2,2,3,3)
          - nn.LeakyReLU(0.1, true)
  local conv2 = conv1 
          - nn.SpatialConvolution(64,128,5,5,2,2,2,2)
          - nn.LeakyReLU(0.1,true)
  local conv3 = conv2 
          - nn.SpatialConvolution(128,256,5,5,2,2,2,2)
          - nn.LeakyReLU(0.1,true)
  local conv3_1 = conv3 
          - nn.SpatialConvolution(256,256,3,3,1,1,1,1)
          - nn.LeakyReLU(0.1,true)
  local conv4 = conv3_1 
          - nn.SpatialConvolution(256,256,3,3,2,2,1,1)
          - nn.LeakyReLU(0.1,true)
  local conv4_1 = conv4 
            - nn.SpatialConvolution(256,512,3,3,1,1,1,1)
            - nn.LeakyReLU(0.1,true)
  local conv5 = conv4_1 
            - nn.SpatialConvolution(512,512,3,3,2,2,1,1)
            - nn.LeakyReLU(0.1,true)
  local conv5_1 = conv5
            - nn.SpatialConvolution(512,512,3,3,1,1,1,1)
            - nn.LeakyReLU(0.1,true)
  local conv6 = conv5_1 
            - nn.SpatialConvolution(512,1024,3,3,2,2,1,1)
            - nn.LeakyReLU(0.1,true)
  local conv6_1 = conv6
            - nn.SpatialConvolution(1024,1024,3,3,1,1,1,1)
            - nn.LeakyReLU(0.1,true)
  local predict_flow6 =conv6_1
            - nn.SpatialConvolution(1024,2,3,3,1,1,1,1)
  local deconv5 = conv6_1
            - nn.SpatialFullConvolution(1024,512,4,4,2,2,1,1)
            - nn.LeakyReLU(0.1,true)
  local upsampled_flow6_to_5 = predict_flow6
            - nn.SpatialFullConvolution(2,2,4,4,2,2,1,1)
  local concat5 = {conv5_1,deconv5,upsampled_flow6_to_5} - nn.JoinTable(1,3)
  local deconv4 = concat5
            - nn.SpatialFullConvolution(1026,256,4,4,2,2,1,1)
            - nn.LeakyReLU(0.1,true)
  local predict_flow5 = concat5
            - nn.SpatialConvolution(1026,2,3,3,1,1,1,1)
  local upsampled_flow5_to_4 = predict_flow5
            - nn.SpatialFullConvolution(2,2,4,4,2,2,1,1)
  local concat4 = {conv4_1,deconv4,upsampled_flow5_to_4} - nn.JoinTable(1,3)
  local deconv3 = concat4
            - nn.SpatialFullConvolution(770,128,4,4,2,2,1,1)
            - nn.LeakyReLU(0.1,true)
  local predict_flow4 = concat4
            - nn.SpatialConvolution(770,2,3,3,1,1,1,1)
  local upsampled_flow4_to_3 = predict_flow4
            - nn.SpatialFullConvolution(2,2,4,4,2,2,1,1)
  local concat3 = {conv3_1,deconv3,upsampled_flow4_to_3} - nn.JoinTable(1,3)
  local deconv2 = concat3
            - nn.SpatialFullConvolution(386,64,4,4,2,2,1,1)
            - nn.LeakyReLU(0.1,true)
  local predict_flow3 = concat3
            - nn.SpatialConvolution(386,2,3,3,1,1,1,1)
  local upsampled_flow3_to_2 = predict_flow3
            - nn.SpatialFullConvolution(2,2,4,4,2,2,1,1)
  local concat2 = {conv2,deconv2,upsampled_flow3_to_2} - nn.JoinTable(1,3)
  local predict_flow2 = concat2
            - nn.SpatialConvolution(194,2,3,3,1,1,1,1)
  
  local net = nn.gModule({input},{predict_flow6,predict_flow5,predict_flow4,predict_flow3,predict_flow2})
  
  
   net.downSample = 4
   net.scales = 5
   net.imageSize = {512,384}
   net.imageCrop = {448,320}
   
  return net
end
