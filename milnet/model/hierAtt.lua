require 'nn'
require 'nngraph'
require 'dpnn'
require 'rnn'
require 'model.Unsqueeze_nc'
require 'model.MixtureTableVarLen'

local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:__init(w2v)
  self.w2v = w2v
end

function ModelBuilder:make_net()
  if opt.cudnn == 1 then
    require 'cudnn'
    require 'cunn'
  end

  sentEnc = self:getSentenceEncoder()

  local docencInputSize = (#opt.kernels) * opt.num_feat_maps

  local fwd = nn.GRU(docencInputSize, opt.doc_rnn_hiddensize, opt.doc_rho, opt.doc_gru_dropout)
  fwd:maskZero(1)
  local fullFwd = nn.Sequencer(fwd)

  local bwd = nn.GRU(docencInputSize, opt.doc_rnn_hiddensize, opt.doc_rho, opt.doc_gru_dropout)
  bwd:maskZero(1)
  local seqBwd = nn.Sequencer(bwd)
  local fullBwd = nn.Sequential()
  fullBwd:add(nn.SeqReverseSequence(1))
  fullBwd:add(seqBwd)
  fullBwd:add(nn.SeqReverseSequence(1))

  local biConcat = nn.Concat(3)
  biConcat:add(fullFwd)
  biConcat:add(fullBwd)
  
  local model = nn.Sequential()
  model:add(sentEnc)
  model:add(nn.Transpose({1,2}))
  model:add(biConcat)

  if opt.doc_att == 0 then
    if opt.pool == 'max' then
      model:add(nn.Max(1))
    elseif opt.pool == 'mean' then
      model:add(nn.Mean(1))
    end

    local flin = nn.Linear(2 * opt.doc_rnn_hiddensize, opt.num_classes)
    flin.name = 'linear'

    model:add(nn.Dropout(opt.dropout_p))
    model:add(flin)
    if opt.cudnn == 1 then
      model:add(cudnn.LogSoftMax())
    else
      model:add(nn.LogSoftMax())
    end
  else
    local att = nn.Sequential()
    local att_lin_idx = 1
    if opt.doc_att_dropout >= 0 then
      att:add(nn.Dropout(opt.doc_att_dropout))
      att_lin_idx = att_lin_idx + 1
    end
    att:add(nn.Linear(2*opt.doc_rnn_hiddensize, opt.doc_att_size))
    att:add(nn.Tanh())
    att:add(nn.LinearNoBias(opt.doc_att_size, 1))
    att.modules[att_lin_idx].name = 'linear'

    local par_att = nn.Parallel(1,2)
    par_att:add(att)

    for i=2,opt.max_doc do
      cloned_att = att:sharedClone()
      cloned_att.modules[att_lin_idx].name = 'lin_clone'
      par_att:add(cloned_att)
    end

    local full_att = nn.Sequential()
    full_att:add(par_att)
    if opt.cudnn == 1 then
      full_att:add(cudnn.SoftMax())
    else
      full_att:add(nn.SoftMax())
    end

    local concat_att = nn.ConcatTable()
    concat_att.name = 'concat_att'
    concat_att:add(full_att)
    concat_att:add(nn.Transpose({1,2}))

    local head = nn.Sequential()
    head.name = 'head'
    head:add(nn.MixtureTableVarLen())
    head:add(nn.Dropout(opt.dropout_p))
    head:add(nn.Linear(2 * opt.doc_rnn_hiddensize, opt.num_classes))
    head.modules[3].name = 'linear'
    if opt.cudnn == 1 then
      head:add(cudnn.LogSoftMax())
    else
      head:add(nn.LogSoftMax())
    end

    model:add(concat_att)
    model:add(head)
  end

  return model
end

function ModelBuilder:getSentenceEncoder()
  local lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
  if opt.model_type ~= 'rand' then
    lookup.weight:copy(self.w2v)
  else
    lookup.weight:uniform(-0.25, 0.25)
  end
  lookup.weight[1]:zero()

  local kernels = opt.kernels
  local kconcat = nn.ConcatTable()
  for i = 1, #kernels do
    local conv
    if opt.cudnn == 1 then
      conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kernels[i])
      conv.weight:uniform(-0.01, 0.01)
      conv.bias:zero()

      local single_conv = nn.Sequential()
      single_conv:add(conv)
      if opt.bn == 1 then
        single_conv:add(nn.SpatialBatchNormalization(opt.num_feat_maps))
      end
      single_conv:add(nn.Squeeze(3,3))
      single_conv:add(cudnn.ReLU(true))
      single_conv:add(nn.Max(2,2))
      single_conv:add(nn.Unsqueeze(1,1))

      kconcat:add(single_conv)
    end
  end

  local sent_conv = nn.Sequential()
  sent_conv:add(lookup)
  sent_conv:add(nn.Unsqueeze(1,2))
  sent_conv:add(kconcat)
  sent_conv:add(nn.JoinTable(3))

  local par = nn.Parallel(2, 2)
  par:add(sent_conv)

  for i=2,opt.max_doc do
    local cloned_conv = sent_conv:sharedClone()
    par:add(cloned_conv)
  end

  return par
end

return ModelBuilder
