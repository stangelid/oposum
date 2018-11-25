local MixtureTableVarLen, parent = torch.class('nn.MixtureTableVarLen', 'nn.MixtureTable')

function MixtureTableVarLen:updateOutput(input) 
   local gaterInput, expertInputs = table.unpack(input)
   
   -- buffers 
   self._gaterView = self._gaterView or input[1].new()
   self._expert = self._expert or input[1].new()
   self._expertView = self._expertView or input[1].new()
   
   self.dimG = 2
   local batchSize = gaterInput:size(1)
   if gaterInput:dim() < 2 then
      self.dimG = 1
      self.dim = self.dim or 1
      batchSize = 1
   end
   self.dim = self.dim or 2
      
   if self.table or torch.type(expertInputs) == 'table' then 
      -- expertInputs is a Table
      self.table = true
      if gaterInput:size(self.dimG) ~= #expertInputs then
         error"Should be one gater output per expert"
      end
      local expertInput = expertInputs[1]
      if self.batchSize ~= batchSize then
         self.size:resize(expertInput:dim()+1):fill(1)
         if self.dimG > 1 then 
            self.size[1] = gaterInput:size(1)
         end
         self.size[self.dim] = gaterInput:size(self.dimG)
         self.output:resizeAs(expertInput)
         self.backwardSetup = false
         self.batchSize = batchSize
      end
      self._gaterView:view(gaterInput, self.size)
      self.output:zero()
      -- multiply accumulate gater outputs by their commensurate expert
      for i,expertInput in ipairs(expertInputs) do
         local gate = self._gaterView:select(self.dim,i):expandAs(expertInput)
         self.output:addcmul(expertInput, gate)
      end
   else
      -- expertInputs is a Tensor :
      if self.batchSize ~= batchSize or self.size[self.dim] ~= gaterInput:size(self.dimG) then
         self.size:resize(expertInputs:dim()):fill(1)
         if self.dimG > 1 then
            self.size[1] = gaterInput:size(1)
         end
         self.size[self.dim] = gaterInput:size(self.dimG)
         self.output:resizeAs(expertInputs:select(self.dim, 1))
         self.batchSize = batchSize
         self.backwardSetup = false
      end
      self._gaterView:view(gaterInput, self.size)
      self._expert:cmul(self._gaterView:expandAs(expertInputs), expertInputs)
      self.output:sum(self._expert, self.dim)
      self.output:resizeAs(expertInputs:select(self.dim, 1))
   end

   return self.output
end
