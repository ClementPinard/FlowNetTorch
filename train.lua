--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'FlowCriterion'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk


local optimState = {
         learningRate = opt.LR,
         beta1 = opt.momentum,
         weightDecay = opt.weightDecay,
         beta2 = 0.999
      }

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     10,   opt.LR,   4e-4 },
        { 11,     20,   opt.LR/2,   4e-4  },
        { 21,     30,   opt.LR/4,   4e-4 },
        { 31,     50,   opt.LR/10,   4e-4 },
        { 50,    1e8,   opt.LR/20,   4e-4 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
local batchNumber
local loss_epoch
local mean_dist
local multiScaleNet = multiScale(opt.downSample,opt.scales,net):cuda()
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
trainLogger:setNames{'avg loss (train set)'}
trainLogger:style{'-'}
trainLogger.showPlot = false

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   
   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         beta1 = opt.momentum,
         weightDecay = params.weightDecay,
         beta2 = 0.999
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   loss_epoch = 0
   mean_dist = {}
   mean_train_error = {}
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize,true)
            inputs:resize(inputs:size(1),inputs:size(2)*inputs:size(3),inputs:size(4),inputs:size(5))
            
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()
   loss_epoch = loss_epoch / opt.epochSize
   for i= 1,#mean_dist do
    mean_dist[i] = mean_dist[i] / opt.epochSize
   end
   trainLogger:add{loss_epoch}
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average error (px): %.5f \t ',
                       epoch, tm:time().real, 20*mean_dist[5]))
   print('\n')

   -- save model
   collectgarbage()

   saveDataParallel(paths.concat(opt.save, 'model.t7'), model:clearState()) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   local err, err2, outputs, multiLabels
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      multiLabels = multiScaleNet:forward(labels)
      err = criterion:forward(outputs, multiLabels)
      err2={}
      for i,v in ipairs(criterion.criterions) do
        table.insert(err2,v.output)
      end
      local gradOutputs = criterion:backward(outputs, multiLabels)
      model:backward(inputs, gradOutputs)
      
      return err, gradParameters
   end
   optim.adam(feval, parameters, optimState)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   for i,v in ipairs(err2) do
    mean_dist[i] = (mean_dist[i] or 0)  + v
   end

   print(('Epoch: [%d][%d/%d]\tTime %.3f Overall Loss %.4f\t HighRes EPE: %.5f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, 20*err2[5],
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end
