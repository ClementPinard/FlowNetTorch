require 'torch'
local Thread = require 'threads'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'


local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for flying chairs image pairs.
     Optimized for large datasets.
]],
   {name="path",
    type="string",
    help="path of flyingchairs directory"},
    
   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced, balanced will sample the whole dataset with a random permutation",
    default = "random"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end
   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end
   self.index = 1
   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image + corresponding flowmap in dataset
   
   local imgSequences = {}
   local maxPathLength = 0
   local length = 0
   --last update of official flying chairs dataset doesn't have a list file anymore. we keep the functionnality for outdated and custom datasets
   if paths.filep(paths.concat(self.path,'FlyingChairs_release.list')) then
      print('loading ' .. paths.concat(self.path,'FlyingChairs_release.list'))
      local list_file = io.open(paths.concat(self.path,'FlyingChairs_release.list'))

      for line in list_file:lines() do
        length = length + 1
        local img1,img2,flow = unpack(line:split("\t"))
        maxPathLength = math.max(maxPathLength,paths.concat(self.path,img1):len())
        maxPathLength = math.max(maxPathLength,paths.concat(self.path,img2):len())
        maxPathLength = math.max(maxPathLength,paths.concat(self.path,flow):len())
        table.insert(imgSequences,{paths.concat(self.path,img1),paths.concat(self.path,img2),paths.concat(self.path,flow)})
      end
   else
     local data_dir = paths.concat(self.path,'FlyingChairs_release','data')
     maxPathLength =  paths.concat(data_dir,'00001_img1.ppm'):len()
     for i = 1,22872 do
       length = length + 1
       table.insert(imgSequences,{paths.concat(data_dir,('%05d_img1.ppm'):format(i)),paths.concat(data_dir,('%05d_img2.ppm'):format(i)),paths.concat(data_dir,('%05d_flow.flo'):format(i))})
     end
  end
   maxPathLength = maxPathLength + 1
   self.imagePath:resize(length, 3,maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   
   for i,imgs in ipairs(imgSequences) do
      count = count +1
      ffi.copy(s_data, imgs[1])
      s_data = s_data + maxPathLength
      ffi.copy(s_data, imgs[2])
      s_data = s_data + maxPathLength
      ffi.copy(s_data, imgs[3])
      s_data = s_data + maxPathLength
      
     if self.verbose then
      xlua.progress(count,length)
     end
   end
   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
  
  

    print('Splitting training and test sets to a ratio of '
             .. self.split .. '/' .. (100-self.split))
    local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
    self.ListTrain = torch.randperm(splitidx)
    self.perm = self.ListTrain
    if splitidx == count then -- all samples were allocated to train set
        self.ListTest  = torch.LongTensor()
    else
       self.ListTest  = torch.range(splitidx,count)
   end
   
   
end


function dataset:train_size()
  return self.ListTrain:size(1)
end

function dataset:test_size()
  return self.ListTest:size(1)
end

-- get img pair for training
function dataset:getASample(augmentData)
   local index
   if self.samplingMode == 'random' then
    index = math.max(1, math.ceil(torch.uniform() * self.ListTrain:size(1)))
   else
    index = self.perm[self.index]
    self.index  = self.index % self.perm:size(1) + 1
   end
   local imgpath1 = ffi.string(torch.data(self.imagePath[self.ListTrain[index]][1]))
   local imgpath2 = ffi.string(torch.data(self.imagePath[self.ListTrain[index]][2]))
   local flowpath = ffi.string(torch.data(self.imagePath[self.ListTrain[index]][3]))
   return self:sampleHook(imgpath1,imgpath2,flowpath,augmentData)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, flowTable)
   local data, flowData
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)
   data = torch.Tensor(quantity, dataTable[1]:size(1),
		       dataTable[1]:size(2), dataTable[1]:size(3), dataTable[1]:size(4))
   flowData = torch.Tensor(quantity, 2, flowTable[1]:size(2), flowTable[1]:size(3))
   for i=1,quantity do
     flowData[i]:copy(flowTable[i])
     data[i]:copy(dataTable[i])
   end
   
   return data, flowData
end

function dataset:resetSampler()
  self.index = 1
end

-- sampler, samples from the training set.
function dataset:sample(quantity,augmentData)
   
   assert(quantity)
   local dataTable = {}
   local flowTable = {}
   for i=1,quantity do
      if self.samplingMode == 'balanced' then
        if self.index == 1 then
          self.perm = torch.randperm(self.ListTrain:size(1))
        end
      end
      local out = self:getASample()
      table.insert(dataTable, out[1])
      table.insert(flowTable, out[2])
      
   end
   local data, flow = tableToOutput(self, dataTable, flowTable)
   return data, flow
   
end

function dataset:get(i1, i2)
   local indices = torch.range(i1, i2)
   local quantity = i2 - i1 + 1
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local flowTable = {}
   for i=1,quantity do
      -- load the sample
      local index = self.ListTest[indices[i]]
      local imgpath1 = ffi.string(torch.data(self.imagePath[index][1]))
      local imgpath2 = ffi.string(torch.data(self.imagePath[index][2]))
      local flowpath = ffi.string(torch.data(self.imagePath[index][3]))
      local out = self:sampleHook(imgpath1,imgpath2,flowpath,false)
      table.insert(dataTable, out[1])
      table.insert(flowTable, out[2])
   end
   local data, flow = tableToOutput(self, dataTable, flowTable)
   return data, flow
end

return dataset
