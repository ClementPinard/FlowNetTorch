require 'nngraph'
require 'graph'
loadcaffe = require 'loadcaffe'

opt = lapp[[
     --deploy  (default 'deploy.tpl.prototxt')  prototxt file
     --model   (default 'flownets_smalldisp.caffemodel')  caffemodel file
     --multiOutput                          outputs only highest resolution flowmap (good for inference) or all resolutions (good for training)
     --save    (default 'FlowNet_pretrained')
     --drawGraph    save graph's drawing in svg format
     ]]
print(opt)

model = loadcaffe.load(opt.deploy,opt.model)
print(model)
input = - nn.Identity()
conv1 = input -model:get(1) -nn.LeakyReLU(0.1, true)
conv2 = conv1 -model:get(3) - nn.LeakyReLU(0.1,true)
conv3 = conv2 - model:get(5) - nn.LeakyReLU(0.1,true)
conv3_1 = conv3 -model:get(7) - nn.LeakyReLU(0.1,true)
conv4 = conv3_1 -model:get(9) - nn.LeakyReLU(0.1,true)
conv4_1 = conv4 -model:get(11) - nn.LeakyReLU(0.1,true)
conv5 = conv4_1 -model:get(13) - nn.LeakyReLU(0.1,true)
conv5_1 = conv5 -model:get(15) - nn.LeakyReLU(0.1,true)
conv6 = conv5_1 -model:get(17) - nn.LeakyReLU(0.1,true)
conv6_1 = conv6 -model:get(19) - nn.LeakyReLU(0.1,true)
predict_flow6 =conv6_1 - model:get(21)
deconv5 = conv6_1 - model:get(22) - nn.LeakyReLU(0.1,true)
upsampled_flow6_to_5 = predict_flow6 - model:get(24)
concat5 = {conv5_1,deconv5,upsampled_flow6_to_5} -nn.JoinTable(1,3)
deconv4 = concat5 -model:get(26) -nn.LeakyReLU(0.1,true)
predict_flow5 = concat5 -model:get(25)
upsampled_flow5_to_4 = predict_flow5 -model:get(28)
concat4 = {conv4_1,deconv4,upsampled_flow5_to_4} -nn.JoinTable(1,3)
deconv3 = concat4 -model:get(30) -nn.LeakyReLU(0.1,true)
predict_flow4 = concat4 -model:get(29)
upsampled_flow4_to_3 = predict_flow4 -model:get(32)
concat3 = {conv3_1,deconv3,upsampled_flow4_to_3} -nn.JoinTable(1,3)
deconv2 = concat3 -model:get(34) -nn.LeakyReLU(0.1,true)
predict_flow3 = concat3 -model:get(33)
upsampled_flow3_to_2 = predict_flow3 -model:get(36)
concat2 = {conv2,deconv2,upsampled_flow3_to_2} -nn.JoinTable(1,3)
predict_flow2 = concat2 -model:get(37)

if opt.multiOutput then
  g = nn.gModule({input},{predict_flow6,predict_flow5,predict_flow4,predict_flow3,predict_flow2})
else
  g = nn.gModule({input},{predict_flow2})
end


graph.dot(g.fg, opt.save, opt.save)

torch.save(opt.save..'.t7',g)