--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'FlowCriterion'
require 'nngraph'
require 'graph'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating criterion from file: models/' .. opt.netType .. '.lua')
   _, criterion = createModel(opt.nGPU)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU):cuda() -- defined in util.lua
else
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model, criterion = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
   model = model:cuda()
   criterion=criterion:cuda()
   
   function MSRinit(net)
     local function init(name)
       for k,v in pairs(net:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if v.bias then v.bias:zero() end
       end
     end
        
     init'nn.SpatialConvolution'
     init'nn.SpatialFullConvolution'
   end
      
  MSRinit(model)
  cudnn.convert(model, cudnn)
   
end

print('=> Model')
--print(model)



opt.downSample = model.downSample or opt.downSample
opt.scales = model.scales or opt.scales

criterion = criterion or multiImgCriterion(opt.scales,opt.loss,{0.32,0.08,0.02,0.01,0.005}):cuda()

testCriterion = multiImgCriterion(opt.scales,'Abs'):cuda()


print('=> Criterion')
--print(criterion)
-- save the configuration in the test.log
print('Will save at '..opt.save)
file = io.open(paths.concat(opt.save, 'network_summary.log'),'w')
file:write(tostring(model))
file:write('\n')
file:write('\n')
file:close()

collectgarbage()
