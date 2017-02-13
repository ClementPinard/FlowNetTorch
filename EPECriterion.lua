local EPECriterion, parent = torch.class('nn.EPECriterion', 'nn.Criterion')

function EPECriterion:__init(scales,criterion)
   parent.__init(self)
   self.EPE = torch.Tensor()
   if criterion == 'SmoothL1' then
    self.criterion = nn.SmoothL1Criterion():cuda()
   elseif criterion == 'MSE' then
    self.criterion = nn.MSECriterion():cuda()
   end
end

function EPECriterion:updateOutput(input, target)
   
   local diffMap = input-target
   diffMap = torch.pow(diffMap,2)
   assert(input:nDimension() == 4 or input:nDimension() == 3)
   if input:nDimension() == 4 then
     self.EPE = torch.pow(diffMap[{{},1}] + diffMap[{{},2}], 0.5)
   else
     self.EPE = torch.pow(diffMap[1] + diffMap[2], 0.5)
   end
   if self.criterion then
    self.output = self.criterion:forward(self.EPE, torch.CudaTensor:resizeAs(self.EPE):fill(0))
   else
    self.output = torch.mean(self.EPE)
   end
   return self.output
end

function EPECriterion:updateGradInput(input, target)
   self.gradInput = input-target
   assert(self.gradInput:nDimension() == 4 or gradInput:nDimension() == 3)
   if self.gradInput:nDimension() == 4 then
     self.gradInput[{{},1}]:cdiv(self.EPE)
     self.gradInput[{{},2}]:cdiv(self.EPE)
   else
     self.gradInput[1]:cdiv(self.EPE)
     self.gradInput[2]:cdiv(self.EPE)
   end
   if self.criterion then
    self.gradInput = self.gradInput*self.criterion:backward(self.EPE,torch.CudaTensor:resizeAs(self.EPE):fill(0))
   end
   return self.gradInput
end