local EPECriterion, parent = torch.class('nn.EPECriterion', 'nn.Criterion')

function EPECriterion:__init(scales,criterion)
   parent.__init(self)
   self.EPE = torch.Tensor()
   if criterion == 'SmoothL1' then
    self.criterion = nn.SmoothL1Criterion():cuda()
   elseif criterion == 'MSE' then
    self.criterion = nn.MSECriterion():cuda()
   else --AbsCriterions
    self.criterion = nn.AbsCriterion():cuda()
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
   self.zeroEPE = torch.zeros(self.EPE:size()):cuda():fill(0)
   self.output = self.criterion:forward(self.EPE, self.zeroEPE)
   return self.output
end

function EPECriterion:updateGradInput(input, target)
   self.gradInput = input-target
   local gradOutput = torch.cdiv(self.criterion:backward(self.EPE,self.zeroEPE),self.EPE)
   assert(self.gradInput:nDimension() == 4 or gradInput:nDimension() == 3)
   if self.gradInput:nDimension() == 4 then
     self.gradInput[{{},1}]:cmul(gradOutput)
     self.gradInput[{{},2}]:cmul(gradOutput)
   else
     self.gradInput[1]:cmul(gradOutput)
     self.gradInput[2]:cmul(gradOutput)
   end
   return self.gradInput
end