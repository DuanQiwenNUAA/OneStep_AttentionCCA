import torch
import torch.optim as optim
import numpy as np
from cca import CCA
from self_attention import SelfAttention, MultiHeadSelfAttention, apply_self_attention
from cross_attention import CrossAttention, apply_cross_attention
from data_preprocessing import (
    load_multi_view_data,
    normalize_data,
    prepare_for_attention,
    convert_to_tensor,
    split_train_test,
    batch_data
)
from evaluation import evaluate_attention_effect, evaluate_kmeans_clustering
from complexity_metrics import compute_pds, compute_mnc, calculate_dataset_complexity
import scipy.io as sio
import csv
import os


class AttentionCCA:
    """
    注意力机制结合CCA的主类
    先对每个视图数据进行自注意力机制处理，得到新的向量表示
    然后可以进行后续的CCA处理
    """
    def __init__(self, config=None):
        """
        初始化AttentionCCA模型
        
        参数:
            config: 配置字典，包含模型参数
        """
        # 默认配置
        self.config = {
            'view1_input_dim': 100,  # 第一个视图的输入维度
            'view2_input_dim': 100,  # 第二个视图的输入维度
            'view1_output_dim': None,  # 第一个视图的输出维度，默认为输入维度
            'view2_output_dim': None,  # 第二个视图的输出维度，默认为输入维度
            'attention_type': 'multihead',  # 'single' 或 'multihead'
            'num_heads': 4,  # 多头自注意力的头数
            'hidden_dim': 128,  # 隐藏层维度
            'use_gpu': True,  # 是否使用GPU
            'enable_cross_attention': True,  # 是否执行交叉注意力环节
            'enable_complexity_analysis': True,  # 是否执行复杂度分析
        }
        
        # 更新配置
        if config is not None:
            self.config.update(config)
        
        # 初始化设备
        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        
        # 初始化自注意力模型
        self._init_attention_models()
        
    def _init_attention_models(self):
        """
        初始化注意力模型
        """
        if self.config['attention_type'] == 'single':
            # 单头自注意力
            self.view1_attention = SelfAttention(
                input_dim=self.config['view1_input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view1_output_dim']
            )
            self.view2_attention = SelfAttention(
                input_dim=self.config['view2_input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view2_output_dim']
            )
        else:
            # 多头自注意力
            self.view1_attention = MultiHeadSelfAttention(
                input_dim=self.config['view1_input_dim'],
                num_heads=self.config['num_heads'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view1_output_dim']
            )
            self.view2_attention = MultiHeadSelfAttention(
                input_dim=self.config['view2_input_dim'],
                num_heads=self.config['num_heads'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view2_output_dim']
            )
            
        # 初始化交叉注意力模型
        self.cross_attention1 = CrossAttention(
            input_dim1=self.config['view1_output_dim'] or self.config['view1_input_dim'],
            input_dim2=self.config['view2_output_dim'] or self.config['view2_input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['view1_output_dim'] or self.config['view1_input_dim']
        )
        self.cross_attention2 = CrossAttention(
            input_dim1=self.config['view2_output_dim'] or self.config['view2_input_dim'],
            input_dim2=self.config['view1_output_dim'] or self.config['view1_input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['view2_output_dim'] or self.config['view2_input_dim']
        )
            
    def process_views(self, view1_data, view2_data, sequence_length1=None, sequence_length2=None):
        """
        处理两个视图数据，应用自注意力机制和交叉注意力机制
        
        参数:
            view1_data: 第一个视图的数据
            view2_data: 第二个视图的数据
            sequence_length1: 第一个视图的序列长度
            sequence_length2: 第二个视图的序列长度
        
        返回:
            tuple: (processed_view1, processed_view2)，处理后的两个视图数据
        """
        # 准备数据格式
        prepared_view1 = prepare_for_attention(view1_data, sequence_length1)
        prepared_view2 = prepare_for_attention(view2_data, sequence_length2)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(prepared_view1)
        tensor_view2 = convert_to_tensor(prepared_view2)
        
        # 应用自注意力机制
        processed_view1 = apply_self_attention(tensor_view1, self.view1_attention, self.device)
        processed_view2 = apply_self_attention(tensor_view2, self.view2_attention, self.device)

        # 计算自注意力模型输出数据的结构复杂度
        # # 计算PDS分数
        pds_view1 = compute_pds(torch.squeeze(processed_view1,dim = 1).detach().cpu().numpy())
        pds_view2 = compute_pds(torch.squeeze(processed_view2,dim = 1).detach().cpu().numpy())
        
        # 应用交叉注意力机制（如果启用）和结构复杂性度量
        if self.config['enable_complexity_analysis']: # 使用绝对值的倒数作为权重
            cross_view1 = apply_cross_attention(1.0 / abs(pds_view1) * processed_view1, 1.0 / abs(pds_view2)  * processed_view2, self.cross_attention1, self.device)
            cross_view2 = apply_cross_attention(1.0 / abs(pds_view2) * processed_view2, 1.0 / abs(pds_view1)  * processed_view1, self.cross_attention2, self.device)
            
            # 将结果转换回numpy数组（如果需要）
            if not isinstance(view1_data, torch.Tensor):
                cross_view1 = cross_view1.cpu().numpy()
                cross_view2 = cross_view2.cpu().numpy()
            
            return cross_view1, cross_view2
        else:
            # 如果不启用结构复杂性度量，则不进行加权
            cross_view1 = apply_cross_attention(processed_view1, processed_view2, self.cross_attention1, self.device)
            cross_view2 = apply_cross_attention(processed_view2, processed_view1, self.cross_attention2, self.device)

            if not isinstance(view1_data, torch.Tensor):
                cross_view1 = cross_view1.cpu().numpy()
                cross_view2 = cross_view2.cpu().numpy()
            
            return cross_view1, cross_view2

    def save_models(self, view1_path, view2_path, cross_view1_path=None, cross_view2_path=None):
        """
        保存自注意力模型和交叉注意力模型
        
        参数:
            view1_path: 第一个视图的自注意力模型保存路径
            view2_path: 第二个视图的自注意力模型保存路径
            cross_view1_path: 第一个视图的交叉注意力模型保存路径(可选)
            cross_view2_path: 第二个视图的交叉注意力模型保存路径(可选)
        """
        torch.save(self.view1_attention.state_dict(), view1_path)
        torch.save(self.view2_attention.state_dict(), view2_path)
        
        if cross_view1_path is not None and self.config['enable_cross_attention']:
            torch.save(self.cross_attention1.state_dict(), cross_view1_path)
        if cross_view2_path is not None and self.config['enable_cross_attention']:
            torch.save(self.cross_attention2.state_dict(), cross_view2_path)
        
    def load_models(self, view1_path, view2_path, cross_view1_path=None, cross_view2_path=None):
        """
        加载自注意力模型和交叉注意力模型
        
        参数:
            view1_path: 第一个视图的模型加载路径
            view2_path: 第二个视图的模型加载路径
            cross_view1_path: 第一个视图的交叉注意力模型加载路径(可选)
            cross_view2_path: 第二个视图的交叉注意力模型加载路径(可选)
        """
        self.view1_attention.load_state_dict(torch.load(view1_path, map_location=self.device))
        self.view2_attention.load_state_dict(torch.load(view2_path, map_location=self.device))
        
        # 加载交叉注意力模型（如果启用且提供了路径）
        if self.config['enable_cross_attention']:
            if cross_view1_path is not None:
                self.cross_attention1.load_state_dict(torch.load(cross_view1_path, map_location=self.device))
            if cross_view2_path is not None:
                self.cross_attention2.load_state_dict(torch.load(cross_view2_path, map_location=self.device))
        
        # 设置为评估模式
        self.view1_attention.eval()
        self.view2_attention.eval()
        if self.config['enable_cross_attention']:
            self.cross_attention1.eval()
            self.cross_attention2.eval()
        
    def _correlation_loss(self, view1_features, view2_features):
        """
        计算两个视图特征之间的相关性损失
        目标是最大化或最小化两个视图之间的相关性
        
        参数:
            view1_features: 第一个视图处理后的特征
            view2_features: 第二个视图处理后的特征
        
        返回:
            loss: 相关性损失值
        """
        # 对特征进行平均池化，减少序列维度
        view1_mean = torch.mean(view1_features, dim=1)  # [batch_size, output_dim]
        view2_mean = torch.mean(view2_features, dim=1)  # [batch_size, output_dim]
        
        # 计算协方差矩阵
        batch_size = view1_mean.size(0)
        centered_view1 = view1_mean - torch.mean(view1_mean, dim=0, keepdim=True)
        centered_view2 = view2_mean - torch.mean(view2_mean, dim=0, keepdim=True)
        
        # 归一化特征以计算相关性
        view1_norm = torch.norm(centered_view1, dim=1, keepdim=True) + 1e-8
        view2_norm = torch.norm(centered_view2, dim=1, keepdim=True) + 1e-8
        
        normalized_view1 = centered_view1 / view1_norm
        normalized_view2 = centered_view2 / view2_norm
        
        # 计算视图间的相关性
        correlation = torch.mean(torch.sum(normalized_view1 * normalized_view2, dim=1))
        
        # 如果我们想最大化相关性，使用1 - correlation作为损失
        # 如果我们想最小化相关性，直接使用correlation作为损失
        # 这里我们选择最大化视图间的相关性
        loss = 1 - correlation
        
        return loss
        
    def train_model(self, train_data, num_epochs=100, batch_size=32, learning_rate=0.001):
        """
        训练AttentionCCA模型
        
        参数:
            train_data: 训练数据，包含(view1_data, view2_data)元组
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        
        返回:
            loss_history: 训练过程中的损失历史
            processed_view1: 视图1处理后的特征
            processed_view2: 视图2处理后的特征
        """
        # 解包训练数据
        view1_train, view2_train = train_data
        
        # 准备数据格式
        view1_data = prepare_for_attention(view1_train)
        view2_data = prepare_for_attention(view2_train)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(view1_data)
        tensor_view2 = convert_to_tensor(view2_data)
        
        # 创建批次数据并转换为列表以便获取长度
        train_batches = list(batch_data(tensor_view1, tensor_view2, batch_size))
        
        # 设置优化器参数，同时优化自注意力和交叉注意力模块
        params = list(self.view1_attention.parameters()) + list(self.view2_attention.parameters())
        if self.config['enable_cross_attention']:
            params += list(self.cross_attention1.parameters()) + list(self.cross_attention2.parameters())
            
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # 记录损失历史
        loss_history = []
        
        # 开始训练循环
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in train_batches:
                # 解包批次数据
                batch_view1, batch_view2 = batch
                
                # 移动到指定设备
                batch_view1 = batch_view1.to(self.device)
                batch_view2 = batch_view2.to(self.device)
                
                # 前向传播 - 使用训练模式
                # 自注意力处理
                self_view1 = apply_self_attention(batch_view1, self.view1_attention, self.device, train_mode=True)
                self_view2 = apply_self_attention(batch_view2, self.view2_attention, self.device, train_mode=True)
                
                # 结构复杂性加权（如果启用）
                if self.config['enable_complexity_analysis']:
                    # 计算自注意力模型输出数据的结构复杂度
                    # # 计算PDS分数
                    pds_view1 = compute_pds(torch.squeeze(self_view1,dim = 1).detach().cpu().numpy())
                    pds_view2 = compute_pds(torch.squeeze(self_view2,dim = 1).detach().cpu().numpy())
                   
                    # 使用结构复杂度加权作为交叉注意力层的输入
                    processed_view1 = apply_cross_attention(abs(pds_view1) * self_view1, abs(pds_view2) * self_view2, self.cross_attention1, self.device, train_mode=True)
                    processed_view2 = apply_cross_attention(abs(pds_view2) * self_view2, abs(pds_view1) * self_view1, self.cross_attention2, self.device, train_mode=True)
                    
                else:
                    # 使用自注意力层的输出作为交叉注意力层的输入
                    processed_view1 = apply_cross_attention(self_view1, self_view2, self.cross_attention1, self.device, train_mode=True)
                    processed_view2 = apply_cross_attention(self_view2, self_view1, self.cross_attention2, self.device, train_mode=True)
                
                # 计算损失
                loss = self._correlation_loss(processed_view1, processed_view2)
                
                # 反向传播和参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / len(train_batches)
            loss_history.append(avg_epoch_loss)
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return loss_history, processed_view1, processed_view2

# 示例用法函数
def demo_attention_cca():
    """
    演示OneStep_AttentionCCA的使用方法，包括模型训练过程
    """
    # 创建结果文件
    results_file = os.path.join(os.path.dirname(__file__), r'D:\硕士\AttentionCCA\OneStep_AttentionCCA\Results\results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视图', '指标名称', '指标值', '阶段'])

    # 四个视图，分别为：（400，512）、（400，59）、（400，864）、（400，254）
    mat_data = sio.loadmat("D:\本科毕业设计\Python_Projects\DataSets\数据集\ORL.mat")
    view1_data = mat_data['fea'][0][0]
    view2_data = mat_data['fea'][0][1]
    labels = mat_data['gt'].squeeze()

    n_clusters = 40  # 聚类数量
    
    # 创建配置
    config = {
        'view1_input_dim': view1_data.shape[1],
        'view2_input_dim': view2_data.shape[1],
        'view1_output_dim': 50,  # 指定降维后的输出维度
        'view2_output_dim': 50,  # 指定降维后的输出维度
        'attention_type': 'multihead',
        'num_heads': 4,
        'hidden_dim': 128,
        'use_gpu': True,
        'enable_cross_attention': True,
        'enable_complexity_analysis': True,
    }
    
    # 初始化模型
    model = AttentionCCA(config)
    # 是否启用结构复杂度加权
    model.config['enable_cross_attention'] = True
    
    # 使用未训练的模型处理数据
    print("\n===== 未训练模型的处理结果 =====")
    model.config['enable_cross_attention'] = True
    untrained_view1, untrained_view2 = model.process_views(view1_data, view2_data)

    # 使用CCA进行降维
    cca0 = CCA(n_components=model.config['view1_output_dim'])
    cca0.fit(view1_data, view2_data)
    view1_cca, view2_cca = cca0.transform(view1_data, view2_data)

    print(f"\n测试数据形状:")
    print(f"  视图1形状: {view1_data.shape}")
    print(f"  视图2形状: {view2_data.shape}")
    print(f"训练后处理结果形状:")
    print(f"  视图1形状: {untrained_view1.squeeze().shape}")
    print(f"  视图2形状: {untrained_view2.squeeze().shape}")

    # 评估原始视图的Kmeans聚类效果
    print("\n原始视图的Kmeans聚类效果:")
    original_kmeans_result = evaluate_kmeans_clustering(view1_data, view2_data, n_clusters, labels, labels)
    print(f"  视图1轮廓系数: {original_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图1Calinski-Harabasz指数: {original_kmeans_result['view1_calinski_harabasz_score']:.4f}")
    print(f"  视图1Davies-Bouldin指数: {original_kmeans_result['view1_davies_bouldin_score']:.4f}")
    print(f"  视图1调整兰德指数: {original_kmeans_result['view1_adjusted_rand_score']:.4f}")
    print(f"  视图1调整互信息分数: {original_kmeans_result['view1_adjusted_mutual_info_score']:.4f}")
    print(f"  视图2轮廓系数: {original_kmeans_result['view2_silhouette']:.4f}")
    print(f"  视图2Calinski-Harabasz指数: {original_kmeans_result['view2_calinski_harabasz_score']:.4f}")
    print(f"  视图2Davies-Bouldin指数: {original_kmeans_result['view2_davies_bouldin_score']:.4f}")
    print(f"  视图2调整兰德指数: {original_kmeans_result['view2_adjusted_rand_score']:.4f}")
    print(f"  视图2调整互信息分数: {original_kmeans_result['view2_adjusted_mutual_info_score']:.4f}")

        # 记录原始视图评估结果
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视图1', '轮廓系数', original_kmeans_result['view1_silhouette'], '直接对原始视图聚类'])
        writer.writerow(['视图1', 'Calinski-Harabasz指数', original_kmeans_result['view1_calinski_harabasz_score'], '直接对原始视图聚类'])
        writer.writerow(['视图1', 'Davies-Bouldin指数', original_kmeans_result['view1_davies_bouldin_score'], '直接对原始视图聚类'])
        writer.writerow(['视图1', '调整兰德指数', original_kmeans_result['view1_adjusted_rand_score'], '直接对原始视图聚类'])
        writer.writerow(['视图1', '调整互信息分数', original_kmeans_result['view1_adjusted_mutual_info_score'], '直接对原始视图聚类'])
        writer.writerow(['视图2', '轮廓系数', original_kmeans_result['view2_silhouette'], '直接对原始视图聚类'])
        writer.writerow(['视图2', 'Calinski-Harabasz指数', original_kmeans_result['view2_calinski_harabasz_score'], '直接对原始视图聚类'])
        writer.writerow(['视图2', 'Davies-Bouldin指数', original_kmeans_result['view2_davies_bouldin_score'], '直接对原始视图聚类'])
        writer.writerow(['视图2', '调整兰德指数', original_kmeans_result['view2_adjusted_rand_score'], '直接对原始视图聚类'])
        writer.writerow(['视图2', '调整互信息分数', original_kmeans_result['view2_adjusted_mutual_info_score'], '直接对原始视图聚类'])

    print("\n处理后视图的Kmeans聚类效果:")
    processed_kmeans_result = evaluate_kmeans_clustering(untrained_view1, untrained_view2, n_clusters, labels, labels)
    print(f"  视图1轮廓系数: {processed_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图1Calinski-Harabasz指数: {processed_kmeans_result['view1_calinski_harabasz_score']:.4f}")
    print(f"  视图1Davies-Bouldin指数: {processed_kmeans_result['view1_davies_bouldin_score']:.4f}")
    print(f"  视图1调整兰德指数: {processed_kmeans_result['view1_adjusted_rand_score']:.4f}")
    print(f"  视图1调整互信息分数: {processed_kmeans_result['view1_adjusted_mutual_info_score']:.4f}")
    print(f"  视图2轮廓系数: {processed_kmeans_result['view2_silhouette']:.4f}")
    print(f"  视图2Calinski-Harabasz指数: {processed_kmeans_result['view2_calinski_harabasz_score']:.4f}")
    print(f"  视图2Davies-Bouldin指数: {processed_kmeans_result['view2_davies_bouldin_score']:.4f}")
    print(f"  视图2调整兰德指数: {processed_kmeans_result['view2_adjusted_rand_score']:.4f}")
    print(f"  视图2调整互信息分数: {processed_kmeans_result['view2_adjusted_mutual_info_score']:.4f}")

        # 记录处理后视图评估结果
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视图1', '轮廓系数', processed_kmeans_result['view1_silhouette'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图1', 'Calinski-Harabasz指数', processed_kmeans_result['view1_calinski_harabasz_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图1', 'Davies-Bouldin指数', processed_kmeans_result['view1_davies_bouldin_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图1', '调整兰德指数', processed_kmeans_result['view1_adjusted_rand_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图1', '调整互信息分数', processed_kmeans_result['view1_adjusted_mutual_info_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图2', '轮廓系数', processed_kmeans_result['view2_silhouette'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图2', 'Calinski-Harabasz指数', processed_kmeans_result['view2_calinski_harabasz_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图2', 'Davies-Bouldin指数', processed_kmeans_result['view2_davies_bouldin_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图2', '调整兰德指数', processed_kmeans_result['view2_adjusted_rand_score'], '用未训练的模型处理后视图聚类'])
        writer.writerow(['视图2', '调整互信息分数', processed_kmeans_result['view2_adjusted_mutual_info_score'], '用未训练的模型处理后视图聚类'])

    print("\n CCA处理后视图的Kmeans聚类效果:")
    processed_kmeans_result = evaluate_kmeans_clustering(view1_cca, view2_cca, n_clusters, labels, labels)
    print(f"  视图1轮廓系数: {processed_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图1Calinski-Harabasz指数: {processed_kmeans_result['view1_calinski_harabasz_score']:.4f}")
    print(f"  视图1Davies-Bouldin指数: {processed_kmeans_result['view1_davies_bouldin_score']:.4f}")
    print(f"  视图1调整兰德指数: {processed_kmeans_result['view1_adjusted_rand_score']:.4f}")
    print(f"  视图1调整互信息分数: {processed_kmeans_result['view1_adjusted_mutual_info_score']:.4f}")
    print(f"  视图2轮廓系数: {processed_kmeans_result['view2_silhouette']:.4f}")
    print(f"  视图2Calinski-Harabasz指数: {processed_kmeans_result['view2_calinski_harabasz_score']:.4f}")
    print(f"  视图2Davies-Bouldin指数: {processed_kmeans_result['view2_davies_bouldin_score']:.4f}")
    print(f"  视图2调整兰德指数: {processed_kmeans_result['view2_adjusted_rand_score']:.4f}")
    print(f"  视图2调整互信息分数: {processed_kmeans_result['view2_adjusted_mutual_info_score']:.4f}")

    # 记录处理后视图评估结果
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视图1', '轮廓系数', processed_kmeans_result['view1_silhouette'], '用CCA处理后视图聚类'])
        writer.writerow(['视图1', 'Calinski-Harabasz指数', processed_kmeans_result['view1_calinski_harabasz_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图1', 'Davies-Bouldin指数', processed_kmeans_result['view1_davies_bouldin_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图1', '调整兰德指数', processed_kmeans_result['view1_adjusted_rand_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图1', '调整互信息分数', processed_kmeans_result['view1_adjusted_mutual_info_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图2', '轮廓系数', processed_kmeans_result['view2_silhouette'], '用CCA处理后视图聚类'])
        writer.writerow(['视图2', 'Calinski-Harabasz指数', processed_kmeans_result['view2_calinski_harabasz_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图2', 'Davies-Bouldin指数', processed_kmeans_result['view2_davies_bouldin_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图2', '调整兰德指数', processed_kmeans_result['view2_adjusted_rand_score'], '用CCA处理后视图聚类'])
        writer.writerow(['视图2', '调整互信息分数', processed_kmeans_result['view2_adjusted_mutual_info_score'], '用CCA处理后视图聚类'])

    # 准备训练数据
    print("\n===== 开始训练模型 =====")
    # 分割训练和测试数据
    view1_train, view1_test, view2_train, view2_test, labels_train, labels_test = split_train_test(view1_data, view2_data, labels, test_ratio=0.2)
    train_data = (view1_train, view2_train)
    test_data = (view1_test, view2_test)
    
    # 训练注意力模型
    print("\n===== 同时训练自注意力和交叉注意力模型 =====")
    self_loss_history, processed_view1, processed_view2 = model.train_model(
        train_data=train_data,
        num_epochs=300,  # 训练轮数
        batch_size=view1_train.shape[0],  # 批次大小
        learning_rate=0.001  # 学习率
    )

    # 保存训练好的模型
    model.save_models(
        view1_path="view1_attention_model.pth",
        view2_path="view2_attention_model.pth",
        cross_view1_path="cross_view1_attention_model.pth",
        cross_view2_path="cross_view2_attention_model.pth"
    )
    print("\n模型已保存到当前目录")
    
    # 使用训练后的模型处理数据
    print("\n===== 使用训练好的自注意力模型和交叉注意力模型对测试集处理 =====")
    trained_view1, trained_view2 = model.process_views(view1_test, view2_test)
    
    # 打印结果形状
    print(f"\n测试数据形状:")
    print(f"  视图1形状: {view1_test.shape}")
    print(f"  视图2形状: {view2_test.shape}")
    print(f"\n模型处理结果形状:")
    print(f"  视图1形状: {trained_view1.squeeze().shape}")
    print(f"  视图2形状: {trained_view2.squeeze().shape}")
    
    # # 评估原始视图的Kmeans聚类效果
    # print("\n原始视图的Kmeans聚类效果:")
    # original_kmeans_result = evaluate_kmeans_clustering(view1_test, view2_test, n_clusters, labels_test, labels_test)
    # print(f"  视图1轮廓系数: {processed_kmeans_result['view1_silhouette']:.4f}")
    # print(f"  视图1Calinski-Harabasz指数: {processed_kmeans_result['view1_calinski_harabasz_score']:.4f}")
    # print(f"  视图1Davies-Bouldin指数: {processed_kmeans_result['view1_davies_bouldin_score']:.4f}")
    # print(f"  视图1调整兰德指数: {processed_kmeans_result['view1_adjusted_rand_score']:.4f}")
    # print(f"  视图1调整互信息分数: {processed_kmeans_result['view1_adjusted_mutual_info_score']:.4f}")
    # print(f"  视图2轮廓系数: {processed_kmeans_result['view2_silhouette']:.4f}")
    # print(f"  视图2Calinski-Harabasz指数: {processed_kmeans_result['view2_calinski_harabasz_score']:.4f}")
    # print(f"  视图2Davies-Bouldin指数: {processed_kmeans_result['view2_davies_bouldin_score']:.4f}")
    # print(f"  视图2调整兰德指数: {processed_kmeans_result['view2_adjusted_rand_score']:.4f}")
    # print(f"  视图2调整互信息分数: {processed_kmeans_result['view2_adjusted_mutual_info_score']:.4f}")
        
    # # 记录原始视图测试集评估结果
    # with open(results_file, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['视图1', '轮廓系数', processed_kmeans_result['view1_silhouette'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图1', 'Calinski-Harabasz指数', processed_kmeans_result['view1_calinski_harabasz_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图1', 'Davies-Bouldin指数', processed_kmeans_result['view1_davies_bouldin_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图1', '调整兰德指数', processed_kmeans_result['view1_adjusted_rand_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图1', '调整互信息分数', processed_kmeans_result['view1_adjusted_mutual_info_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图2', '轮廓系数', processed_kmeans_result['view2_silhouette'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图2', 'Calinski-Harabasz指数', processed_kmeans_result['view2_calinski_harabasz_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图2', 'Davies-Bouldin指数', processed_kmeans_result['view2_davies_bouldin_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图2', '调整兰德指数', processed_kmeans_result['view2_adjusted_rand_score'], '对原始视图测试集聚类'])
    #     writer.writerow(['视图2', '调整互信息分数', processed_kmeans_result['view2_adjusted_mutual_info_score'], '对原始视图测试集聚类'])
    
    # 使用CCA对测试集处理并聚类
    cca = CCA(n_components=model.config['view1_output_dim'])
    cca.fit(view1_test, view2_test)
    view1_cca, view2_cca = cca.transform(view1_test, view2_test)

    print("\n CCA处理后视图的Kmeans聚类效果:")
    processed_kmeans_result = evaluate_kmeans_clustering(view1_cca, view2_cca, n_clusters, labels_test, labels_test)
    print(f"  视图1轮廓系数: {processed_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图1Calinski-Harabasz指数: {processed_kmeans_result['view1_calinski_harabasz_score']:.4f}")
    print(f"  视图1Davies-Bouldin指数: {processed_kmeans_result['view1_davies_bouldin_score']:.4f}")
    print(f"  视图1调整兰德指数: {processed_kmeans_result['view1_adjusted_rand_score']:.4f}")
    print(f"  视图1调整互信息分数: {processed_kmeans_result['view1_adjusted_mutual_info_score']:.4f}")
    print(f"  视图2轮廓系数: {processed_kmeans_result['view2_silhouette']:.4f}")
    print(f"  视图2Calinski-Harabasz指数: {processed_kmeans_result['view2_calinski_harabasz_score']:.4f}")
    print(f"  视图2Davies-Bouldin指数: {processed_kmeans_result['view2_davies_bouldin_score']:.4f}")
    print(f"  视图2调整兰德指数: {processed_kmeans_result['view2_adjusted_rand_score']:.4f}")
    print(f"  视图2调整互信息分数: {processed_kmeans_result['view2_adjusted_mutual_info_score']:.4f}")

    # 记录处理后视图评估结果
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视图1', '轮廓系数', processed_kmeans_result['view1_silhouette'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图1', 'Calinski-Harabasz指数', processed_kmeans_result['view1_calinski_harabasz_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图1', 'Davies-Bouldin指数', processed_kmeans_result['view1_davies_bouldin_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图1', '调整兰德指数', processed_kmeans_result['view1_adjusted_rand_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图1', '调整互信息分数', processed_kmeans_result['view1_adjusted_mutual_info_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图2', '轮廓系数', processed_kmeans_result['view2_silhouette'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图2', 'Calinski-Harabasz指数', processed_kmeans_result['view2_calinski_harabasz_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图2', 'Davies-Bouldin指数', processed_kmeans_result['view2_davies_bouldin_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图2', '调整兰德指数', processed_kmeans_result['view2_adjusted_rand_score'], '用CCA处理后视图测试集聚类'])
        writer.writerow(['视图2', '调整互信息分数', processed_kmeans_result['view2_adjusted_mutual_info_score'], '用CCA处理后视图测试集聚类'])
    
    # 评估处理后视图的Kmeans聚类效果
    print("\n处理后视图的Kmeans聚类效果:")
    processed_kmeans_result = evaluate_kmeans_clustering(trained_view1, trained_view2, n_clusters, labels_test, labels_test)
    print(f"  视图1轮廓系数: {processed_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图1Calinski-Harabasz指数: {processed_kmeans_result['view1_calinski_harabasz_score']:.4f}")
    print(f"  视图1Davies-Bouldin指数: {processed_kmeans_result['view1_davies_bouldin_score']:.4f}")
    print(f"  视图1调整兰德指数: {processed_kmeans_result['view1_adjusted_rand_score']:.4f}")
    print(f"  视图1调整互信息分数: {processed_kmeans_result['view1_adjusted_mutual_info_score']:.4f}")
    print(f"  视图2轮廓系数: {processed_kmeans_result['view2_silhouette']:.4f}")
    print(f"  视图2Calinski-Harabasz指数: {processed_kmeans_result['view2_calinski_harabasz_score']:.4f}")
    print(f"  视图2Davies-Bouldin指数: {processed_kmeans_result['view2_davies_bouldin_score']:.4f}")
    print(f"  视图2调整兰德指数: {processed_kmeans_result['view2_adjusted_rand_score']:.4f}")
    print(f"  视图2调整互信息分数: {processed_kmeans_result['view2_adjusted_mutual_info_score']:.4f}")

        # 记录处理后视图评估结果
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视图1', '轮廓系数', processed_kmeans_result['view1_silhouette'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图1', 'Calinski-Harabasz指数', processed_kmeans_result['view1_calinski_harabasz_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图1', 'Davies-Bouldin指数', processed_kmeans_result['view1_davies_bouldin_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图1', '调整兰德指数', processed_kmeans_result['view1_adjusted_rand_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图1', '调整互信息分数', processed_kmeans_result['view1_adjusted_mutual_info_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图2', '轮廓系数', processed_kmeans_result['view2_silhouette'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图2', 'Calinski-Harabasz指数', processed_kmeans_result['view2_calinski_harabasz_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图2', 'Davies-Bouldin指数', processed_kmeans_result['view2_davies_bouldin_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图2', '调整兰德指数', processed_kmeans_result['view2_adjusted_rand_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
        writer.writerow(['视图2', '调整互信息分数', processed_kmeans_result['view2_adjusted_mutual_info_score'], '用自注意力和交叉注意力模型处理后视图测试集聚类'])
    
    return trained_view1, trained_view2


if __name__ == "__main__":
    # 运行演示
    demo_attention_cca()