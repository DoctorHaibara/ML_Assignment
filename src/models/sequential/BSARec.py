import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

class BSARecBase(object):
    @staticmethod
    def parse_model_args(parser):
        # 添加模型参数
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--c', type=int, default=5,
                            help='frequency set.')
        parser.add_argument('--alpha', type=float, default=0.5)
        return parser		
    
    def _base_init(self, args, corpus):
        # 初始化模型参数
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.c = args.c
        self.alpha = args.alpha
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        # 定义模型参数
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        # 定义BSARecBlock
        self.BSARecBlock = nn.ModuleList([
            layers.BSARecBlock(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    alpha=self.alpha, c=self.c, dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])


    def forward(self, feed_dict):
        # 前向传播
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        # 获取有效的历史记录
        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # 位置嵌入
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
        # 自注意力机制
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        for block in self.BSARecBlock:
            his_vectors = block(his_vectors, attn_mask)

        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 获取历史向量
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]

        # 获取物品向量
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)

        # 用户和物品向量
        u_v = his_vector.repeat(1, i_ids.shape[1]).view(i_ids.shape[0], i_ids.shape[1], -1)
        i_v = i_vectors

        return {'prediction': prediction.view(batch_size, -1), 'u_v': u_v, 'i_v': i_v}
    
class BSARec(SequentialModel, BSARecBase):
    # 定义读取器和运行器
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads','c','alpha']

    @staticmethod
    def parse_model_args(parser):
        # 解析模型参数
        parser = BSARecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        # 初始化模型
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # 前向传播
        out_dict = BSARecBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}
    
class SASRecImpression(ImpressionSeqModel, BSARecBase):
    # 定义读取器和运行器
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads','c','alpha']

    @staticmethod
    def parse_model_args(parser):
        # 解析模型参数
        parser = BSARecBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        # 初始化模型
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # 前向传播
        return BSARecBase.forward(self, feed_dict)