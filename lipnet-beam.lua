--[[

LipNet: End-to-End Sentence-level Lipreading. arXiv preprint arXiv:1611.01599 (2016).

Copyright (C) 2017 Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

]] --

--
-- Dependencies
--

require 'io'
require 'sys'
require 'nn'
require 'nngraph'
require 'optim'
require 'hdf5'
require 'paths'

require 'cutorch'
require 'cunn'
require 'cunnx'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true

require 'modules.TemporalJitter'

require 'warp_ctc'
assert(gpu_ctc, 'requires Baidu unmodified warp_ctc (not my fork), built with GPU support')

require 'nnx'

local log = require 'util.log'
log.level = "debug"

require 'pprint'


--
-- Configuration
--

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-beam_width', 200, '')
cmd:option('-seed', 123, 'initial random seed')
cmd:option('-debug', 0, '')
cmd:option('-verbose', 0, '')
cmd:option('-beam_alpha', 1, '')
cmd:option('-beam_beta', 0, '')
cmd:option('-datapath', 'data', 'video data path')
cmd:option('-bs', 100, 'batch size')
cmd:option('-checkpoint', '', '')
cmd:option('-rnn_size', 256, 'rnn size')
cmd:option('-dropout', 0.5, '')
cmd:option('-threads', 8, 'number of torch built-in threads')
cmd:option('-exp', 'exp.0001', 'lua file that returns model, experiment name, optimization settings, ...')
cmd:option('-exp_i', 1, 'if exp returns more than one model, uses this index')
cmd:option('-ignore_checkpoint', 1, 'do not continue from checkpoint')
cmd:option('-print_every', 1, 'iterations between printing')
cmd:option('-test_every', 1, 'iterations between testing')
cmd:option('-checkpoint_every', 1, 'iterations between saving checkpoints')
cmd:text()
local opt = cmd:parse(arg)
assert(opt.exp ~= '', 'exp lua file required')
for k, v in pairs(opt) do
    log.infof('opt: %s=%s', k, tostring(v))
end

--
-- Initialisation
--
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

--
-- CUDA
--

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

--
-- Load model
--
local checkpoint = torch.load(opt.checkpoint)

local model = checkpoint.model
local opt_ch = checkpoint.opt
opt_ch.debug = 0
opt_ch.bs = opt.bs
local stats = checkpoint.stats
local iter = checkpoint.iter

pprint(opt_ch)

-- Load exp
local exp = require(opt_ch.exp)(opt_ch)
pprint(exp.data_loader.vocab)

function editDistance(prediction, target)
    local d = torch.Tensor(#target + 1, #prediction + 1):zero()
    for i = 1, #target + 1 do
        for j = 1, #prediction + 1 do
            if (i == 1) then
                d[1][j] = j - 1
            elseif (j == 1) then
                d[i][1] = i - 1
            end
        end
    end

    for i = 2, #target + 1 do
        for j = 2, #prediction + 1 do
            if (target[i - 1] == prediction[j - 1]) then
                d[i][j] = d[i - 1][j - 1]
            else
                local substitution = d[i - 1][j - 1] + 1
                local insertion = d[i][j - 1] + 1
                local deletion = d[i - 1][j] + 1
                d[i][j] = torch.min(torch.Tensor({ substitution, insertion, deletion }))
            end
        end
    end
    local errorRate = d[#target + 1][#prediction + 1] / #target
    return errorRate
end

-- load data
local softmax = nnob.TimeBatchWrapper { mod = nn.SoftMax() }:cuda()

local stats = { beam_cer = 0, beam_wer = 0 }

-- Test
model.pred:evaluate()

local bs_count = 0

for it = 1, #exp.data_loader.dataset_val, opt.bs do

    xlua.progress(it, #exp.data_loader.dataset_val)

    -- batch indexes
    local idx = {}
    for i = 0, opt.bs - 1 do
        if it + i <= #exp.data_loader.dataset_val then
            table.insert(idx, it + i)
        end
    end

    -- load data
    local x, y, lengths = unpack(exp.data_loader:forward(idx, true))

    -- predict data
    local logits = model.pred:forward(x)
    local prob = softmax:forward(logits):float()

    -- save data
    local filename = '/tmp/pctc.h5'
    local pctcFile = hdf5.open(filename, 'w')
    pctcFile:write('/pctc', prob)
    pctcFile:close()

    -- Beam Search
    local handle = io.popen('python lipnet-beam.py ' .. opt.beam_width .. ' ' .. filename .. ' ' .. opt.beam_alpha .. ' ' .. opt.beam_beta)
    local beam = handle:read("*a"):split("\n")
    handle:close()

    -- Compute error
    for b = 1, #idx do
        bs_count = bs_count + 1

        -- Target
        local tok_y = {}
        for t = 1, #y[b] do
            table.insert(tok_y, exp.data_loader.vocab[y[b][t]])
        end

        if opt.verbose and it == 1 then
            print(b, 'y', table.concat(tok_y, ""))
        end

        -- Split in chars
        local tok = {}
        for c in beam[b]:gmatch "." do
            table.insert(tok, c)
        end


        -- Compute WER
        local tok_w = table.concat(tok, ""):split(' ')
        local tok_y_w = table.concat(tok_y, ""):split(' ')
        local wer = editDistance(tok_w, tok_y_w)
        stats.beam_wer = stats.beam_wer + wer

        -- Compute CER
        local cer = editDistance(tok, tok_y)
        stats.beam_cer = stats.beam_cer + cer
        if opt.verbose and it == 1 then
            print(b, 'b' .. opt.beam_width, table.concat(tok, ""), cer, wer)
        end

        -- Output
        local tok = {}
        local p = prob:narrow(2, b, 1):squeeze()
        for t = 1, lengths[b] do
            local _, c = torch.max(p[t], 1)
            c = c:squeeze()
            table.insert(tok, exp.data_loader.vocab[c - 1] or '_')
        end
        if it == 1 then
            print(b, 'o', table.concat(tok, ""))

            if opt.verbose then
                print('')
            end
        end

        -- break
    end
end


print(opt.beam_width, opt.beam_alpha, opt.beam_beta, 'beam_width CER', stats.beam_cer / bs_count)
print(opt.beam_width, opt.beam_alpha, opt.beam_beta, 'beam_width WER', stats.beam_wer / bs_count)
