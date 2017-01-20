--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
local names = {}
local style ={}
if opt.logAllScales then
for i =1,opt.scales do
  table.insert(names,'train EPE pyr'..(opt.scales + 1 - i))
  table.insert(style,'-')
end
for i =1,opt.scales do
  table.insert(names,'test EPE pyr'..(opt.scales + 1 - i))
  table.insert(style,'-')
end
else
  names = {'avg EPE for HighRes (train set, normal data)', 'avg EPE for HighRes (test set)'}
  style = {'-','-'}
end
testLogger:setNames(names)
testLogger:style(style)
testLogger.showPlot = false


local batchNumber
local mean_dist, train_mean_dist
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   mean_dist = {}
   train_mean_dist = {}
   loss = 0
   i = 1
   while(i< testSize) do
    local indexStart = i
    local indexEnd = math.min(indexStart + opt.batchSize-1, testSize)
      donkeys:addjob(
           -- work to be done by donkey thread
           function()
              local inputs, labels = trainLoader:get(indexStart, indexEnd)
              inputs:resize(inputs:size(1),inputs:size(2)*inputs:size(3),inputs:size(4),inputs:size(5))
              
              local trainInputs, trainLabels = trainLoader:sample(opt.batchSize,false)
              trainInputs:resize(trainInputs:size(1),trainInputs:size(2)*trainInputs:size(3),trainInputs:size(4),trainInputs:size(5))
              
              
              return inputs, labels, trainInputs, trainLabels
           end,
           -- callback that is run in the main thread once the work is done
           testBatch
      )
    
    i = indexEnd +1
   end

   donkeys:synchronize()
   cutorch.synchronize()
   
   for i=1,#mean_dist do
    train_mean_dist[i] = 20*train_mean_dist[i]*opt.batchSize / testSize
    mean_dist[i] = 20*mean_dist[i]*opt.batchSize / testSize
   end
   if opt.logAllScales then
      log = {}
      
      for i =1,opt.scales do
        table.insert(log,train_mean_dist[i])
      end
      for i =1,opt.scales do
        table.insert(log,mean_dist[i])
      end
      testLogger:add(log)
   else
      testLogger:add{train_mean_dist[5], mean_dist[5]}
   end
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average accuracy : %.5f , (px) \t ',
                       epoch, timer:time().real, mean_dist[5]))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local multiScaleNet = multiScale(opt.downSample,opt.scales,net):cuda()

function testBatch(inputsCPU, labelsCPU, trainInputsCPU, trainLabelsCPU)
   batchNumber = batchNumber + opt.batchSize
   local err,err2 = {},{}
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = model:forward(inputs)
   local multiLabels = multiScaleNet:forward(labels) 
   criterion:forward(outputs, multiLabels)
   err={}
   for i,v in ipairs(criterion.criterions) do
    table.insert(err,v.output)
   end
   
   
   inputs:resize(trainInputsCPU:size()):copy(trainInputsCPU)
   labels:resize(trainLabelsCPU:size()):copy(trainLabelsCPU)
   
   local outputs = model:forward(inputs)
   local multiLabels = multiScaleNet:forward(labels)
   testCriterion:forward(outputs, multiLabels)
      err2={}
      for i,v in ipairs(criterion.criterions) do
        table.insert(err2,v.output)
      end
   
   cutorch.synchronize()
   for i= 1,#err2 do
    mean_dist[i] = (mean_dist[i] or 0) + err[i]
    train_mean_dist[i] = (train_mean_dist[i] or 0) + err2[i]
   end
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, testSize))
   end
end
