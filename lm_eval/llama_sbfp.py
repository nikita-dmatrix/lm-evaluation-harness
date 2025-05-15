import numpy as np
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import scipy

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from transformers import Cache
from qtorch import quant


def quantize_bfp(tensor, bits = 4, block_size = 128, along_rows=False):
    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = quant.block_quantize(tensor, bits, dim = 0, rounding='nearest')
        tensor = tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape
        if (shape[-1] % block_size) != 0:
            #print(shape)
            tensor = F.pad(tensor, (0, block_size - (shape[-1] % block_size )), "constant", 0)
        new_shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = quant.block_quantize(tensor, bits, dim = 0, rounding='nearest')
        tensor = tensor.reshape(*new_shape)
        tensor = tensor[..., : shape[-1]]
        tensor = tensor.transpose(-1,-2)
    return tensor



low_cutoff = 1./128
high_cutoff = 496



def extract_4bit_rounded_mantissa_and_exp(x):
    x32  = x.float().view(dtype=torch.int32)
    exp = (x32 & 0x7F800000 ) >> 23
    man8 = (((x32 & 0x007C0000) >> 17) | ((x32 & 0x0003FFFF) != 0)).to(torch.int8)
    man8[torch.logical_or(((man8 & 0x3) == 0x3), ((man8 & 0x6) == 0x6))] += 0x4
    man8 >>= 2
    exp[(man8 & 0x10) != 0] += 1
    man8[(man8 & 0x10) != 0] = 0

    return man8, exp


def float_to_e4m4(x, ebias):
    man8, exp = extract_4bit_rounded_mantissa_and_exp(x)
    exp = exp - 127 + ebias
    man8[exp < 0] = 0
    exp[exp < 0] = 0

    man8[exp > 15] = 15
    exp[exp > 15] = 15

    if x.dtype == torch.float32:
        answ = ((man8.to(torch.int32) << 19) | ((exp + 127 - ebias).to(torch.int32) << 23)).view(dtype=torch.float32)
    elif x.dtype == torch.bfloat16:
        answ = ((man8.to(torch.int16) << 3) | ((exp + 127 - ebias).to(torch.int16) << 7)).view(dtype=torch.bfloat16)
    else:
        raise NotImplementedError
    return answ

def float_to_fp4(x):
    mask_done = torch.zeros_like(x, dtype=torch.bool)
    sign = x < 0
    x = torch.abs(x)
    mask_6 = x > 5.0
    mask_4 = x >= 3.5
    mask_3 = x > 2.5
    mask_2 = x >= 1.75
    mask_15 = x > 1.25
    mask_1 = x >= 0.75
    mask_05 = x > 0.25

    x[mask_6] = 6.0
    mask_done |= mask_6
    x[torch.logical_and(mask_4,  torch.logical_not(mask_done))] = 4.0
    mask_done |= mask_4
    x[torch.logical_and(mask_3,  torch.logical_not(mask_done))] = 3.0
    mask_done |= mask_3
    x[torch.logical_and(mask_2,  torch.logical_not(mask_done))] = 2.0
    mask_done |= mask_2
    x[torch.logical_and(mask_15,  torch.logical_not(mask_done))] = 1.5
    mask_done |= mask_15
    x[torch.logical_and(mask_1,  torch.logical_not(mask_done))] = 1.0
    mask_done |= mask_1
    x[torch.logical_and(mask_05,  torch.logical_not(mask_done))] = 0.5
    mask_done |= mask_05
    x[torch.logical_not(mask_done)] = 0.0
    x[sign] *= -1.0
    return x

def getSBFPEbias(x):
    x = torch.max(torch.abs(x))/7.0
    man8, exp = extract_4bit_rounded_mantissa_and_exp(x)
    exp -= 127
    exp = exp.detach().cpu().item()
    ebias = 15 - (exp if exp > 0 else 0)
    ebias = ebias if ebias > 1 else 1
    return ebias


def quantize_sort(tensor, bits = 4, block_size = 16, along_rows=True, ebias=None, use_sfp=False, wrap_bfp=True):
    if along_rows:
        norm = torch.norm(tensor, dim=0)
        order = torch.argsort(norm)
        #order = torch.arange(len(norm), device=norm.device)
        reverse = torch.empty_like(order)
        reverse[order] = torch.arange(len(order), device=order.device)
        tensor = tensor[:,order]
        tensor = quantize_sbfp(tensor, bits, block_size, along_rows, ebias, use_sfp)
        tensor = quantize_bfp(tensor, 8, 64, along_rows)
        tensor = tensor[:,reverse]
        return tensor
    else:
        raise NotImplementedError

def quantize_sbfp(tensor, bits = 4, block_size = 128, along_rows=False, ebias=None, use_sfp=False):
    if ebias is None:
        ebias = getSBFPEbias(tensor)
        print("ebias", ebias)

    maxrepr = (2 ** (bits - 1) - 1.)
    if use_sfp:
        assert bits == 4
        maxrepr = 6.0

    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        maxx = torch.max(torch.abs(tensor), dim = -1, keepdims=True).values
        scale = maxx / maxrepr

        #print("before", scale)
        scale = float_to_e4m4(scale, ebias)
        #print("after", scale)
        if use_sfp:
            tensor = float_to_fp4(tensor/scale)
        else:
            tensor = torch.round(tensor / scale)
            tensor[tensor > maxrepr] = maxrepr
            tensor[tensor < -maxrepr] = -maxrepr

        #print ("tensormax", tensor.max())
        #print(tensor)
        tensor *= scale
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        return tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape

        #assert shape[-1] % block_size == 0, f"shape {shape} not divisible by block_size"
        if (shape[-1] % block_size) != 0:
            #print(shape)
            tensor = F.pad(tensor, (0, block_size - (shape[-1] % block_size )), "constant", 0)
        new_shape = tensor.shape

        tensor = tensor.reshape(-1, block_size)


        maxx = torch.max(torch.abs(tensor), dim = -1, keepdims=True).values
        #maxx [ maxx < 6e-8] = 1.
        scale = maxx / maxrepr
        scale = float_to_e4m4(scale, ebias)
        #print(tensor)
        #print(scale.shape)

        #print(scale.min(), scale.max())
        if use_sfp:
            tensor = float_to_fp4(tensor/scale)
        else:
            tensor = torch.round(tensor / scale)
            tensor[tensor > maxrepr] = maxrepr
            tensor[tensor < -maxrepr] = -maxrepr

        #print(tensor)

        tensor *= scale

        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = tensor.reshape(*new_shape)
        #print(tensor.shape)
        tensor = tensor[..., : shape[-1]]
        #print(tensor.shape)
        #print("=====")
        return tensor.transpose(-1,-2)

Last login: Mon May 12 13:06:14 on ttys001
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % ssh coder.nikita-ml.main 
ntrukhanov@ntrukhanov-macbook-air ~ % cat ~/.ssh/id_ed25519.pub 
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINN2HRiBrWLpEh6qtDcYk5y5f/dB25Flh0yb1LYXOXCa ntrukhanov@d-matrix.ai
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % ssh coder.nikita-ml.main 
==> ⧗ The workspace agent lost connection
2025-05-15 15:51:56.555+01:00 Wait for it to reconnect or restart your workspace.
2025-05-15 15:51:56.555+01:00 For more information and troubleshooting, see https://coder.com/docs/v2/latest/templates#agent-connection-issues and https://coder.com/docs/coder-oss/latest/templates#troubleshooting-templates


^C
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % ssh coder.nikita-ml.main 
Encountered an error running "coder ssh"
workspace "nikita-ml" has no agents
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % ssh coder.nikita-ml.main 
==> ⧗ The workspace agent lost connection
2025-05-15 16:13:51.586+01:00 Wait for it to reconnect or restart your workspace.
2025-05-15 16:13:51.586+01:00 For more information and troubleshooting, see https://coder.com/docs/v2/latest/templates#agent-connection-issues and https://coder.com/docs/coder-oss/latest/templates#troubleshooting-templates
^C
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % 
ntrukhanov@ntrukhanov-macbook-air ~ % ssh coder.nikita-ml.main 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ ./startup.sh 
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
  libevent-core-2.1-7 libutempter0
The following NEW packages will be installed:
  libevent-core-2.1-7 libutempter0 tmux
0 upgraded, 3 newly installed, 0 to remove and 166 not upgraded.
Need to get 531 kB of archives.
After this operation, 1365 kB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libevent-core-2.1-7 amd64 2.1.12-stable-1build3 [93.9 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 libutempter0 amd64 1.2.1-2build2 [8848 B]
Get:3 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 tmux amd64 3.2a-4ubuntu0.2 [428 kB]
Fetched 531 kB in 1s (401 kB/s)
debconf: delaying package configuration, since apt-utils is not installed
Selecting previously unselected package libevent-core-2.1-7:amd64.
(Reading database ... 85815 files and directories currently installed.)
Preparing to unpack .../libevent-core-2.1-7_2.1.12-stable-1build3_amd64.deb ...
Unpacking libevent-core-2.1-7:amd64 (2.1.12-stable-1build3) ...
Selecting previously unselected package libutempter0:amd64.
Preparing to unpack .../libutempter0_1.2.1-2build2_amd64.deb ...
Unpacking libutempter0:amd64 (1.2.1-2build2) ...
Selecting previously unselected package tmux.
Preparing to unpack .../tmux_3.2a-4ubuntu0.2_amd64.deb ...
Unpacking tmux (3.2a-4ubuntu0.2) ...
Setting up libevent-core-2.1-7:amd64 (2.1.12-stable-1build3) ...
Setting up libutempter0:amd64 (1.2.1-2build2) ...
Setting up tmux (3.2a-4ubuntu0.2) ...
Processing triggers for man-db (2.10.2-1) ...
Processing triggers for libc-bin (2.35-0ubuntu3.1) ...
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ ls
Capsaicin			   eleuther_tasks.txt  keval	  kveval     lm-evaluation-harness  miniconda3	   notebooks	 startup.sh  transformers
Miniconda3-latest-Linux-x86_64.sh  fullevalpre	       kevalpost  llama.cpp  lost+found		    my_perplexity  perp_eval.py  tmpeval     weval
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~$ cd lm-evaluation-harness/lm_eval
(main) coder@coder-nikita-dmatrix-nikita-ml:~/lm-evaluation-harness/lm_eval$ git diff
diff --git a/lm_eval/evaluator.py b/lm_eval/evaluator.py
diff --git a/lm_eval/evaluator.py b/lm_eval/evaluator.py
index 9c94fa54..b8eaec3a 100644
--- a/lm_eval/evaluator.py
+++ b/lm_eval/evaluator.py
@@ -9,6 +9,8 @@ from typing import TYPE_CHECKING, List, Optional, Union
 import numpy as np
 import torch
 
+from .llama_sbfp import wrap_model, wrap_model_weights
+
 import lm_eval.api.metrics
 import lm_eval.api.registry
 import lm_eval.api.task
@@ -293,9 +295,9 @@ def simple_evaluate(
     print(type(lm))
     print(type(lm.model))
     print(type(lm.model.model.layers[0].self_attn))
-    from .llama_hack_sbfp_new import wrap_model, wrap_model_weights
+    #from .opt_hack_sbfp_8 import wrap_model, wrap_model_weights
     wrap_model_weights(lm.model)
-    wrap_model(lm.model, quantize=True, block_size=128)
+    #wrap_model(lm.model, quantize=True, block_size=16)
 
     if evaluation_tracker is not None:
         evaluation_tracker.general_config_tracker.log_experiment_args(
diff --git a/lm_eval/llama_hack_sbfp_new.py b/lm_eval/llama_hack_sbfp_new.py
index 6c02d321..24dcbd0c 100644
--- a/lm_eval/llama_hack_sbfp_new.py
+++ b/lm_eval/llama_hack_sbfp_new.py
@@ -9,6 +9,7 @@ import scipy
 import torch
 import torch.utils.checkpoint
 from torch import nn
+import torch.nn.functional as F
 from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
 import transformers
 from transformers import Cache
@@ -50,17 +51,41 @@ def quantize_sbfp(tensor, bits = 4, block_size = 128, along_rows=False):
     else:
         tensor = tensor.transpose(-1,-2)
         shape = tensor.shape
+
+        #assert shape[-1] % block_size == 0, f"shape {shape} not divisible by block_size"
+        if (shape[-1] % block_size) != 0:
+            #print(shape)
+            tensor = F.pad(tensor, (0, block_size - (shape[-1] % block_size )), "constant", 0)
+        new_shape = tensor.shape
+
         tensor = tensor.reshape(-1, block_size)
+
+
+        maxx = torch.max(torch.abs(tensor), dim = -1, keepdims=True).values
+        maxx [ maxx < 6e-8] = 1.
+
(main) coder@coder-nikita-dmatrix-nikita-ml:~/lm-evaluation-harness/lm_eval$ q
bash: q: command not found
(main) coder@coder-nikita-dmatrix-nikita-ml:~/lm-evaluation-harness/lm_eval$ 
(main) coder@coder-nikita-dmatrix-nikita-ml:~/lm-evaluation-harness/lm_eval$ ls
__init__.py  caching		 falcon_fc_hack.py  llama_hack.py   llama_hack5.py   llama_hack_ln.py	     llama_hack_sbfp_norm.py  opt_hack.py	  output	ufpbfp
__main__.py  decontamination	 filters	    llama_hack2.py  llama_hack6.py   llama_hack_sbfp.py      llama_sbfp.py	      opt_hack_ln.py	  prompts	util.py
__pycache__  evaluator.py	 gemma_hack.py	    llama_hack3.py  llama_hack6v.py  llama_hack_sbfp_8.py    loggers		      opt_hack_sbfp_8.py  sbfp_results	utils.py
api	     evaluator_utils.py  llama_fc_hack.py   llama_hack4.py  llama_hack7.py   llama_hack_sbfp_new.py  models		      opteval		  tasks
(main) coder@coder-nikita-dmatrix-nikita-ml:~/lm-evaluation-harness/lm_eval$ vim evaluator.py 
(main) coder@coder-nikita-dmatrix-nikita-ml:~/lm-evaluation-harness/lm_eval$ vim llama_sbfp.py 



        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.quantize:
            #query_states = quantize_bfp(query_states.float(), 8, self.block_size, True).to(self.dtype)
            #key_states = quantize_sbfp(key_states.float(), 4, self.block_size, True).to(self.dtype)
            #value_states = quantize_sbfp(value_states.float(), 4, self.block_size, False).to(self.dtype)
            #key_states = quantize_sbfp(key_states.float(), 4, self.block_size, True)
            key_states = quantize_bfp(key_states.float(),8,64,True).to(self.dtype)
            value_states = quantize_sbfp(value_states.float(), 4, self.block_size, False)
            value_states = quantize_bfp(value_states.float(),8,64,False).to(self.dtype)
            query_states = quantize_bfp(query_states.float(), 8, 64, True).to(self.dtype)


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



def wrap_model(model, quantize = False, reorder=False, pnorm=2, block_size=128):
    for idx in range(model.config.num_hidden_layers):
        model.model.layers[idx].self_attn = LlamaSDPAAttentionWrapper(model.model.layers[idx].self_attn, quantize, reorder, pnorm, block_size)


def quantize_weight(W):
    #return quantize_sbfp(W.float(), 4, 16, True).to(W.dtype).to(W.device)
    #return quantize_bfp(W.float(), 8, 64, True).to(W.dtype).to(W.device)
    return quantize_sort(W.float(), 4, 16, True, use_sfp=False).to(W.dtype).to(W.device)



def wrap_model_weights(model):
    for idx in range(model.config.num_hidden_layers):
        model.model.layers[idx].self_attn.q_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.q_proj.weight.data)
        model.model.layers[idx].self_attn.k_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.k_proj.weight.data)
        model.model.layers[idx].self_attn.v_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.v_proj.weight.data)
        model.model.layers[idx].self_attn.o_proj.weight.data = quantize_weight(model.model.layers[idx].self_attn.o_proj.weight.data)
        model.model.layers[idx].mlp.gate_proj.weight.data = quantize_weight(model.model.layers[idx].mlp.gate_proj.weight.data)
        model.model.layers[idx].mlp.up_proj.weight.data = quantize_weight(model.model.layers[idx].mlp.up_proj.weight.data)
        model.model.layers[idx].mlp.down_proj.weight.data = quantize_weight(model.model.layers[idx].mlp.down_proj.weight.data)



