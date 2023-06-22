# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decoder self-attention layer definition."""
from typing import Optional, Tuple

import torch
from torch import nn


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
            If `None` is passed, Inter-attention is not used, such as
            CIF, GPT, and other decoder only model.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: Optional[nn.Module],
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        # size=256默认值
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.norm3 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # 这里在i=1，也就是trt只有sos的时候是none，其余的时候不是none，而是一个之前decoder六层的输出的list
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            # 这里-1表示的是之前的cache所以每次都是比现在的少一个
            # 这里每次只放入一个字，这个字就是最后一个字，也就是decode的最新的一个字
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]
        file_name='/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/attention_matrix.txt'
        with open(file_name, 'a') as file:
                file.write('            self attention'+'\n')
        x = residual + self.dropout(
            self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])
        if not self.normalize_before:
            x = self.norm1(x)


        file_name='/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/attention_matrix.txt'
        with open(file_name, 'a') as file:
                file.write('            src attention'+'\n')
        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm2(x)
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask)[0])
            # 所以这里的实际上是只拿encoder的输出来做attention，而不是拿encoder的每一层来做attention
            if not self.normalize_before:
                x = self.norm2(x)


        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        # print('x')
        # print(x.shape)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
            # print('chach')
            # print(cache.shape)
        
        

        return x, tgt_mask, memory, memory_mask
