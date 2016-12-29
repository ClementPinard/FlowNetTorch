--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'
require 'math'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------


-- a cache file of the training metadata (if doesnt exist, will be created)
trainCache = paths.concat(opt.cache, 'trainCache_.t7')
meanstdCache = paths.concat(opt.cache, 'meanstdCache_'..(opt.noColor and 1 or 3)..'.t7')

if opt.overWrite or (not paths.filep(trainCache)) or (not paths.filep(meanstdCache)) then paths.dofile('donkey.lua') end



do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      local cache_files = {trainCache,meanstdCache}
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
            require 'FlowCriterion'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            trainCache = cache_files[1]
            meanstdCache = cache_files[2]
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

donkeys:addjob(function() return trainLoader:train_size() end, function(s) trainSize = s end)
donkeys:synchronize()
assert(trainSize, "Failed to get train size")

if opt.epochSize ==0 then
  opt.epochSize = math.floor(trainSize/opt.batchSize)
end




donkeys:addjob(function() return trainLoader:test_size() end, function(s) testSize = s end)
donkeys:synchronize()
assert(testSize, "Failed to get test size")
testSize = math.min(testSize,opt.epochSize*opt.batchSize)

print('train size: '.. opt.epochSize*opt.batchSize..' samples')
print('test size: '.. testSize..' samples')
