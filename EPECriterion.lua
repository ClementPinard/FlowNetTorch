local EPECriterion, parent = torch.class('nn.EPECriterion', 'nn.Criterion')

function EPECriterion:__init(scales,initialDownScale,weights,criterion)
   parent.__init(self)
   self.EPE = torch.ensor()
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
   self.output = torch.mean(EPE)
   return self.output
end

function EPECriterion:updateGradInput(input, target)
   self.gradInput = input-target
   assert(gradInput:nDimension() == 4 or gradInput:nDimension() == 3)
   if gradInput:nDimension() == 4 then
     self.gradInput[{{},1}]:cdiv(self.EPE)
     self.gradInput[{{},2}]:cdiv(self.EPE)
   else
     self.gradInput[1]:cdiv(self.EPE)
     self.gradInput[2]:cdiv(self.EPE)
   end
   return self.gradInput
end