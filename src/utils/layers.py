# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn
import math


class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, kq_same=False, bias=True, attention_d=-1):
		super().__init__()
		"""
		It has projection layer for getting keys, queries and values. Followed by attention.
		"""
		self.d_model = d_model
		self.h = n_heads
		if attention_d<0:
			self.attention_d = self.d_model
		else:
			self.attention_d = attention_d

		self.d_k = self.attention_d // self.h
		self.kq_same = kq_same

		if not kq_same:
			self.q_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.k_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.v_linear = nn.Linear(d_model, self.attention_d, bias=bias)

	def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
		new_x_shape = x.size()[:-1] + (self.h, self.d_k)
		return x.view(*new_x_shape).transpose(-2, -3)

	def forward(self, q, k, v, mask=None):
		origin_shape = q.size()

		# perform linear operation and split into h heads
		if not self.kq_same:
			q = self.head_split(self.q_linear(q))
		else:
			q = self.head_split(self.k_linear(q))
		k = self.head_split(self.k_linear(k))
		v = self.head_split(self.v_linear(v))

		# calculate attention using function we will define next
		output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

		# concatenate heads and put through final linear layer
		output = output.transpose(-2, -3).reshape(list(origin_shape)[:-1]+[self.attention_d]) # modified
		return output

	@staticmethod
	def scaled_dot_product_attention(q, k, v, d_k, mask=None):
		"""
		This is called by Multi-head attention object to find the values.
		"""
		scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -np.inf)
		scores = (scores - scores.max()).softmax(dim=-1)
		scores = scores.masked_fill(torch.isnan(scores), 0)
		output = torch.matmul(scores, v)  # bs, head, q_len, d_k
		return output

class AttLayer(nn.Module):
	"""Calculate the attention signal(weight) according the input tensor.
	Reference: RecBole https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py#L236
	Args:
		infeatures (torch.FloatTensor): An input tensor with shape of[batch_size, XXX, embed_dim] with at least 3 dimensions.

	Returns:
		torch.FloatTensor: Attention weight of input. shape of [batch_size, XXX].
	"""

	def __init__(self, in_dim, att_dim):
		super(AttLayer, self).__init__()
		self.in_dim = in_dim
		self.att_dim = att_dim
		self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
		self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

	def forward(self, infeatures):
		att_signal = self.w(infeatures)  # [batch_size, XXX, att_dim]
		att_signal = fn.relu(att_signal)  # [batch_size, XXX, att_dim]

		att_signal = torch.mul(att_signal, self.h)  # [batch_size, XXX, att_dim]
		att_signal = torch.sum(att_signal, dim=-1)  # [batch_size, XXX]
		att_signal = fn.softmax(att_signal, dim=-1)  # [batch_size, XXX]

		return att_signal

class TransformerLayer(nn.Module):
	def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
		super().__init__()
		"""
		This is a Basic Block of Transformer. It contains one Multi-head attention object. 
		Followed by layer norm and position wise feedforward net and dropout layer.
		"""
		# Multi-Head Attention Block
		self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

		# Two layer norm layer and two dropout layer
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, seq, mask=None):
		context = self.masked_attn_head(seq, seq, seq, mask)
		context = self.layer_norm1(self.dropout1(context) + seq)
		output = self.linear1(context).relu()
		output = self.linear2(output)
		output = self.layer_norm2(self.dropout2(output) + context)
		return output

class FrequencyLayer(nn.Module):
    def __init__(self, dropout, emb_size, c):
        super(FrequencyLayer, self).__init__()
        # 初始化Dropout层和LayerNorm层
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-12)
        # 设置频率截断参数
        self.c = c // 2 + 1
        # 初始化sqrt_beta参数
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, input_tensor):
        # 获取输入张量的形状
        batch, seq_len, hidden = input_tensor.shape
        # 对输入张量进行快速傅里叶变换
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        # 低通滤波
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        # 高通滤波
        high_pass = input_tensor - low_pass
        # 组合低通和高通滤波结果
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        # 进行Dropout和LayerNorm
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
	
class BSARecLayer(nn.Module):
	def __init__(self, d_model, n_heads, alpha, c,  dropout=0, kq_same=False):
		super().__init__()
        # 初始化频率层和多头注意力层
		self.filter_layer = FrequencyLayer(dropout, d_model, c)
		self.attention_layer = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)
        # 设置alpha参数
		self.alpha = alpha

	def forward(self, input_tensor, attention_mask):
        # 通过频率层
		dsp = self.filter_layer(input_tensor)
        # 通过多头注意力层
		gsp = self.attention_layer(input_tensor, input_tensor, input_tensor, attention_mask)
        # 组合频率层和多头注意力层的输出
		hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp

		return hidden_states

class BSARecBlock(nn.Module):
	def __init__(self, d_model, d_ff, n_heads, alpha, c,  dropout=0, kq_same=False):
		super(BSARecBlock, self).__init__()
        # 初始化BSARec层和前馈层
		self.layer = BSARecLayer(d_model, n_heads, alpha, c,  dropout, kq_same)

		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def gelu(self, x):
		"""Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
		return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


	def forward(self, seq, mask=None):
		context = self.layer(seq, mask)
		context = self.layer_norm1(self.dropout1(context) + seq)
		output = self.linear1(context)
		output = self.gelu(output)
		output = self.linear2(output)
		output = self.layer_norm2(self.dropout2(output) + context)
		return output
	
		
class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        self.select_index = torch.LongTensor(idx).cuda()
        self.zeros = torch.zeros((self.ksize-1, input_dim)).cuda()

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x) # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:,-1,:]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize*l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class LocalRNNLayer(nn.Module):
	"Encoder is made up of attconv and feed forward (defined below)"
	def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
		super(LocalRNNLayer, self).__init__()
		self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        #self.connection = SublayerConnection(output_dim, dropout)
		self.norm = nn.LayerNorm(output_dim)
		self.dropout = nn.Dropout(dropout)


	def forward(self, x):
		"Follow Figure 1 (left) for connections."
		return x + self.dropout(self.local_rnn(self.norm(x)))

class RBSARecLayer(nn.Module):
	def __init__(self, d_model, n_heads, alpha, c, rnn_type, ksize=6 , dropout=0,  kq_same=False):
		super().__init__()
        # 初始化频率层和多头注意力层
		self.filter_layer = FrequencyLayer(dropout, d_model, c)
		# 初始化RNN
		self.rnn = LocalRNNLayer(d_model, d_model, rnn_type, ksize, dropout)
		self.attention_layer = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)
        # 设置alpha参数
		self.alpha = alpha

	def forward(self, input_tensor, attention_mask):
        # 通过频率层
		filter = self.filter_layer(input_tensor)
		rnn = self.rnn(input_tensor)
		hidden_states = self.alpha * filter + (1 - self.alpha) * rnn
        # 通过多头注意力层
		hidden_states = self.attention_layer(hidden_states, hidden_states, hidden_states, attention_mask)
        
		return hidden_states

class RBSARecBlock(nn.Module):
	def __init__(self, d_model, d_ff, n_heads, alpha, c,  rnn_type,ksize,dropout=0, kq_same=False):
		super(RBSARecBlock, self).__init__()
        # 初始化BSARec层和前馈层
		self.layer = RBSARecLayer(d_model, n_heads, alpha, c, rnn_type ,ksize, dropout, kq_same)

		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def gelu(self, x):
		"""Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
		return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


	def forward(self, seq, mask=None):
		context = self.layer(seq, mask)
		context = self.layer_norm1(self.dropout1(context) + seq)
		output = self.linear1(context)
		output = self.gelu(output)
		output = self.linear2(output)
		output = self.layer_norm2(self.dropout2(output) + context)
		return output
	
class MultiHeadTargetAttention(nn.Module):
    '''
    Reference: FuxiCTR, https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/attentions/target_attention.py
    '''
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True):
        super(MultiHeadTargetAttention, self).__init__()
        if not use_qkvo:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        # linear projection
        if self.use_qkvo:
            query = self.W_q(target_item)
            key = self.W_k(history_sequence)
            value = self.W_v(history_sequence)
        else:
            query, key, value = target_item, history_sequence, history_sequence

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1) # 700*1*1*1

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        return output

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention


class MLP_Block(nn.Module):
	'''
	Reference: FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/blocks/mlp_block.py
	'''
	def __init__(self, input_dim, hidden_units=[], 
				 hidden_activations="ReLU", output_dim=None, output_activation=None, 
				 dropout_rates=0.0, batch_norm=False, 
				 layer_norm=False, norm_before_activation=True,
				 use_bias=True):
		super(MLP_Block, self).__init__()
		dense_layers = []
		if not isinstance(dropout_rates, list):
			dropout_rates = [dropout_rates] * len(hidden_units)
		if not isinstance(hidden_activations, list):
			hidden_activations = [hidden_activations] * len(hidden_units)
		hidden_activations = [getattr(nn, activation)()
                    if activation != "Dice" else Dice(emb_size) for activation, emb_size in zip(hidden_activations, hidden_units)]
		hidden_units = [input_dim] + hidden_units
		for idx in range(len(hidden_units) - 1):
			dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
			if norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if hidden_activations[idx]:
				dense_layers.append(hidden_activations[idx])
			if not norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if dropout_rates[idx] > 0:
				dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
		if output_dim is not None:
			dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
		if output_activation is not None:
			dense_layers.append(getattr(nn, output_activation)())
		self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
	
	def forward(self, inputs):
		return self.mlp(inputs)


class Dice(nn.Module):
	"""The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

	Input shape:
		- 2 dims: [batch_size, embedding_size(features)]
		- 3 dims: [batch_size, num_features, embedding_size(features)]

	Output shape:
		- Same shape as input.

	References
		- [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
		- https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
	"""

	def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
		super(Dice, self).__init__()
		assert dim == 2 or dim == 3

		self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
		self.sigmoid = nn.Sigmoid()
		self.dim = dim

		# wrap alpha in nn.Parameter to make it trainable
		if self.dim == 2:
			self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
		else:
			self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

	def forward(self, x):
		assert x.dim() == self.dim
		if self.dim == 2:
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
		else:
			x = torch.transpose(x, 1, 2)
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
			out = torch.transpose(out, 1, 2)
		return out
	
