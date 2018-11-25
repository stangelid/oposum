require 'torch'
require 'nn'

local function _assertTensor(t)
  assert(torch.isTensor(t), "This module only works on tensor")
end

-- This module will work for non-contiguous inputs
Unsqueeze_nc, _ = torch.class('nn.Unsqueeze_nc', 'nn.Unsqueeze')
function Unsqueeze_nc:updateGradInput(input, gradOutput)
  _assertTensor(input)
  _assertTensor(gradOutput)
  assert(input:nElement() == gradOutput:nElement())

  if self.gradInput:isContiguous() and gradOutput:isContiguous() then
    self.gradInput:view(gradOutput, input:size())
  else
    self.gradInput:reshape(gradOutput, input:size())
  end

  return self.gradInput
end
