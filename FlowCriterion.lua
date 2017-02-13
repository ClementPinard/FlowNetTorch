require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
dofile 'EPECriterion.lua'

function multiScale(initScale,scaleNb,net)
  if not net then
    net = nn.ConcatTable()
    for i=scaleNb,1,-1 do
      local scale = initScale*2^(i-1)
      net:add(nn.SpatialAveragePooling(scale,scale,scale,scale))
    end
  end
  return net
end

function multiImgCriterion(nb, loss, weights)
  local criterion = nn.ParallelCriterion()
  weights = weights and  torch.Tensor(weights) or torch.Tensor(nb):fill(1)
  
  for i=1,nb do
    criterion:add(nn.EPECriterion(loss),weights[i])
  end
  
  return criterion
end