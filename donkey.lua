--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
require 'lfs'
paths.dofile('dataset.lua')
paths.dofile('util.lua')
flow_loader = dofile('flowFileLoader.lua')



-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {opt.noColor and 1 or 3, opt.imageSize[1], opt.imageSize[2]}
local cropSize = {opt.noColor and 1 or 3, opt.imageCrop[1], opt.imageCrop[2]}

local function loadImage(path)
   local input = image.load(path, loadSize[1], 'float')
   input = image.scale(input, loadSize[2], loadSize[3])
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local Hook = function(self, img1, img2, flow,augmentData)
   collectgarbage()
   
   local inputs = torch.FloatTensor(2,loadSize[1],loadSize[3],loadSize[2])
   inputs[1]:copy(loadImage(img1))
   inputs[2]:copy(loadImage(img2))
   local label = flow_loader.load(flow)/20
   
   if augmentData then
     local iW = inputs:size(4)
     local iH = inputs:size(3)
     -- do hflip and vflip with probability 0.5
     if torch.uniform() > 0.5 then 
       for i = 1,inputs:size(1) do
        inputs[i] = image.hflip(inputs[i]) 
       end
       label = image.hflip(label)
       label[1] = label[1]*(-1)
     end
     if torch.uniform() > 0.5 then
      for i = 1,inputs:size(1) do
        inputs[i] = image.vflip(inputs[i])
      end
      label = image.vflip(label)
      label[2] = label[2]*(-1)
     end
     
     --apply data augmentation : random zoom, translation and rotation
     local zoom = torch.uniform(0.9,1.0)
     local t1,t2 = 10*torch.rand(2),10*torch.rand(2)
     local r1,r2 = torch.uniform(-0.2,0.2),torch.uniform(-0.1,0.1)
     
     --generate flowamp from rotation between the 2 frames   
     local rotate_flow = torch.Tensor():resizeAs(label)
     for i=1,opt.width do
      rotate_flow[2][{{},i}]:fill((i-opt.width/2)*(-r2))
     end
     for i=1,opt.height do
      rotate_flow[1][i]:fill((i-opt.height/2)*(r2))
     end
     
     --data augmentation
     label:add(rotate_flow/20)
     
     label = image.rotate(label,r1)
     --rotate flow vectors
     label_ = label:clone()
     label[1] = math.cos(r1)*label_[1] + math.sin(r1)*label_[2]
     label[2] = -math.sin(r1)*label_[1] + math.cos(r1)*label_[2]

     inputs[1] = image.rotate(inputs[1],r1)
     inputs[2] = image.rotate(inputs[2],r1+r2)
     
     label = image.translate(label,t1[1],t1[2])
     inputs[1] = image.translate(inputs[1],t1[1],t1[2])
     inputs[2] = image.translate(inputs[2],t1[1] + t2[1],t1[1] + t2[2])

     label[1]:add((t2[1])/20)
     label[2]:add((t2[2])/20)
     
     --rescale and crop at desired size at the same time
     label = image.crop(image.scale(label,'*'..zoom), 'c', cropSize[2], cropSize[3])*zoom
     local cropped_inputs = torch.FloatTensor(2,loadSize[1],cropSize[3],cropSize[2])
     cropped_inputs[1] = image.crop(image.scale(inputs[1],'*'..zoom), 'c', cropSize[2], cropSize[3])
     cropped_inputs[2] = image.crop(image.scale(inputs[2],'*'..zoom), 'c', cropSize[2], cropSize[3])
     inputs = cropped_inputs
     
   end
   
   -- mean/std
   for j=1,inputs:size(1) do
     for i=1,loadSize[1] do -- channels
        if mean then inputs[{{j},{i},{},{}}]:add(-mean[i]) end
        if std then inputs[{{j},{i},{},{}}]:div(std[i]) end
     end
   end
   return {inputs, label}
end


if paths.filep(trainCache) and not opt.overWrite then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHook = Hook
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      path = opt.data,
      split = 90,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHook = Hook
end
collectgarbage()


-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) and not opt.overWrite then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 1000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {}
   for i=1,nSamples do
      xlua.progress(i,nSamples)
      local imgs = trainLoader:sample(1)[1]
      for j=1,imgs:size(2) do
         meanEstimate[j] = (meanEstimate[j] or 0) + imgs[{{},j}]:mean()
      end
   end
   trainLoader:resetSampler()
   for j=1,#meanEstimate do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {}
   for i=1,nSamples do
      xlua.progress(i,nSamples)
      local imgs = trainLoader:sample(1)[1]
      for j=1,imgs:size(2) do
         stdEstimate[j] = (stdEstimate[j] or 0) + imgs[{{},j}]:std()
      end
   end
   trainLoader:resetSampler()
   for j=1,#stdEstimate do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate
   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end

opt.overWrite = false
