## Implementation of MRF-based predictors for joint multi-agent trajectory distribution modeling.

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import lightning.pytorch as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import coalesce


from modules import QCNetAgentEncoder, MLPPotential, DistancePotential
from layers import FourierEmbedding, MLPLayer
from utils.geometry import polar_to_cartesian
from transforms.normalizer import TrajNorm
from losses import FocalLoss, FocalPairwiseLoss
from metrics import ADE, FDE, minADE, minJADE, minFDE, minJFDE, Brier, SocialCollisionRate
from metrics.collision_env import EnvironmentalCollisionRate, EnvironmentalCollisionRate_SDD
from metrics.coverage import Coverage 

import pickle

class BeliefPropagator(nn.Module):
    # Belief propagation module to update the unary potentials based on the pairwise potentials
    def __init__(self, num_iters:int=3) -> None:
        super(BeliefPropagator, self).__init__()
        self.num_iters = num_iters

    def forward(self, unary_logit: torch.Tensor, pairwise_potential: torch.Tensor, edge_index: torch.Tensor, mode: str = "sum") -> torch.Tensor:
        """
        Perform belief propagation to update the unary potentials.
        Args:
            unary_logit: torch.Tensor, unary potentials of shape (N, num_modes)
            pairwise_potential: torch.Tensor, pairwise potentials of shape (E, num_modes, num_modes)
            edge_index: torch.Tensor, edge index of shape (2, E)
            mode: str, either "sum" (sum-product) or "max" (max-product)
        Returns:
            updated_unary_logit: torch.Tensor, updated unary potentials of shape (N, num_modes)
        """
        N, K = unary_logit.shape
        E = edge_index.shape[1]
        device = unary_logit.device

        messages = torch.zeros(E, K, device=device)  # E, K

        for _ in range(self.num_iters):
            src = edge_index[0]  # (E,)
            # Gather unary logits for all sources: (E, K)
            src_unary = unary_logit[src]  # (E, K)
            src_unary_exp = src_unary.unsqueeze(1)  # (E, 1, K)
            msg_exp = messages.unsqueeze(1)  # (E, 1, K)
            # Compute message matrix: (E, K, K)
            msg_matrix = src_unary_exp + msg_exp + pairwise_potential  # (E, K, K)
            if mode == "sum":
                msg_out = torch.logsumexp(msg_matrix, dim=1)  # (E, K)
            elif mode == "max":
                msg_out, _ = torch.max(msg_matrix, dim=1)  # (E, K)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            msg_norm = msg_out - torch.logsumexp(msg_out, dim=1, keepdim=True)  # (E, K)
            messages = msg_norm

        updated_unary_logit = unary_logit.clone()
        updated_unary_logit[edge_index[1]] += messages
        return updated_unary_logit


class UnarySelector(nn.Module):
    # Unary potential function also used to select and refine trajectory prototypes
    def __init__(self, num_out:int, hidden_dim:int, num_freq_bands:int, env_hist_fuse:bool=False, qc_encoding_only:bool=False, apply_env_filtering:bool=False, stop_grad:bool=False) -> None:
        super(UnarySelector, self).__init__()
        # hyperparameters
        self.num_out = num_out
        self.hidden_dim = hidden_dim
        self.env_hist_fuse = env_hist_fuse # whether to fuse env with historical features or anchor features default to anchor features for backward compatibility
        self.qc_encoding_only = qc_encoding_only
        self.apply_env_filtering = apply_env_filtering
        self.stop_grad = stop_grad
        # modules
        self.obs_encoder = nn.LSTM(input_size=2,
                                    hidden_size=hidden_dim,
                                    num_layers=1,
                                    batch_first=True)
        self.obs_affine = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.obs_project = MLPLayer(hidden_dim, hidden_dim*2, hidden_dim)
        self.obs_social_fuser = MLPLayer(hidden_dim*2, hidden_dim, hidden_dim)
        self.fut_encoder = nn.LSTM(input_size=2,
                                    hidden_size=hidden_dim,
                                    num_layers=1,
                                    batch_first=True)
        self.fut_decoder = nn.LSTMCell(input_size=hidden_dim,
                                   hidden_size=hidden_dim,)
        self.fut_projector = MLPLayer(hidden_dim, hidden_dim//2, 2)
        self.agent_env_encoder = FourierEmbedding(input_dim=2, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.anchor_env_fuser = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1,batch_first=True)
        self.normalize = F.normalize

    def decode_traj(self, fut_len, init_state) -> torch.Tensor:
        """
        Decode the future trajectory given the initial state and the future length
        Args:
            fut_len: int, the length of the future trajectory
            init_state: torch.Tensor, the initial state of the future trajectory
        Returns:
            fut_traj: torch.Tensor, the future trajectory
        """
        A,S,D = init_state.shape
        init_state = init_state.view(A*S,D)
        fut_traj = torch.zeros(A*S, fut_len, D).to(init_state.device)
        h_t, c_t = self.fut_decoder(init_state)
        fut_traj[:,0,:] = h_t
        for t in range(1, fut_len):
            h_t, c_t = self.fut_decoder(init_state, (h_t, c_t))
            fut_traj[:,t,:] = h_t
        fut_traj = fut_traj.view(A,S,fut_len,D)
        fut_traj = self.fut_projector(fut_traj)
        return fut_traj

    def train_forward(self, norm_obs:torch.tensor, 
                      hss_feat:torch.tensor, 
                      env_view:torch.tensor, 
                      anchors:torch.tensor,
                      gt_fut:torch.tensor=None,
                      env_compliant_mask:torch.tensor=None,
                      anchor_label:torch.tensor=None) -> Dict[str, torch.Tensor]:
        """
        Compute the unary potential for each agent given the observed trajectory, HSSE feature, environment view, and anchors
        Args:
            norm_obs: torch.Tensor, normalized observed trajectory of shape (B, obs_len, 2)
            hss_feat: torch.Tensor, HSSE feature of shape (B, hidden_dim)
            env_view: torch.Tensor, environment view of shape (B, env_len, 2)
            anchors: torch.Tensor, trajectory prototypes of shape (num_modes, fut_len, 2)
            anchor_label: torch.Tensor, ground truth anchor labels of shape (B,)
            gt_fut: torch.Tensor, ground truth future trajectory of shape (B, fut_len, 2)
        Returns:
            unary_potentials: torch.Tensor, unary potentials of shape (B, num_modes)
            refined_anchors: torch.Tensor, refined trajectory prototypes of shape (num_modes, fut_len, 2)
        """
        output = self.forward(norm_obs, hss_feat, env_view, anchors, env_compliant_mask=env_compliant_mask, training=True)
        if gt_fut is not None:
            _,(_, fut_embed) = self.fut_encoder(gt_fut)
            fut_embed = fut_embed.permute(1,0,2) # N,1,D
            if self.env_hist_fuse:
                pass # do nothing, already fused in observation embedding
            else:
                fut_env_embed,_ = self.anchor_env_fuser(fut_embed, output['env_emb'], output['env_emb'], need_weights=False)
                fut_embed = fut_embed + fut_env_embed
            fut_embed = self.normalize(fut_embed, dim=-1)
            fut_recon = self.decode_traj(output['T'], fut_embed)
            output['fut_recon_norm'] = fut_recon.squeeze(1)
            output['fut_embed'] = fut_embed.squeeze(1)
        if anchor_label is not None:
            gt_anchor_embed = (anchor_label.unsqueeze(-1) * output['anchor_embed']).sum(dim=1, keepdim=True)
            gt_anchor_embed = gt_anchor_embed + output['obs_res']
            output['gt_anchor_embed'] = gt_anchor_embed.squeeze(1)
            gt_anchor_recon = self.decode_traj(output['T'], gt_anchor_embed)
            output['gt_anchor_recon_norm'] = gt_anchor_recon.squeeze(1)

        return output

    def forward(self, norm_obs:torch.tensor, 
                hss_feat:torch.tensor, 
                env_view:torch.tensor, 
                anchors:torch.tensor,
                env_compliant_mask:torch.tensor=None,
                training:bool=False,
                cache_embeddings=False) -> Dict[str, torch.Tensor]:
        """
        Compute the unary potential for each agent given the observed trajectory, HSSE feature, environment view, and anchors
        Args:
            norm_obs: torch.Tensor, normalized observed trajectory of shape (B, obs_len, 2)
            hss_feat: torch.Tensor, HSSE feature of shape (B, hidden_dim)
            env_view: torch.Tensor, environment view of shape (B, env_len, 2)
            anchors: torch.Tensor, trajectory prototypes of shape (num_modes, fut_len, 2)
        Returns:
            unary_potentials: torch.Tensor, unary potentials of shape (B, num_modes)
            refined_anchors: torch.Tensor, refined trajectory prototypes of shape (num_modes, fut_len, 2)
        """
        S,N,T,D = anchors.shape
        # encode observed trajectory and fuse with ego status and social feature
        if self.qc_encoding_only:
            obs_embed = hss_feat.unsqueeze(1)  # N,1,D
            obs_affine = self.obs_affine(obs_embed)  # N,1,D
            # obs_affine = self.obs_social_fuser(torch.cat([obs_affine.permute(1,0,2), hss_feat.unsqueeze(1)], dim=-1))
        else:
            _, (_, obs_embed) = self.obs_encoder(norm_obs)
            obs_affine = self.obs_affine(obs_embed) # N,D
            obs_affine = self.obs_social_fuser(torch.cat([obs_affine.permute(1,0,2), hss_feat.unsqueeze(1)], dim=-1))
        # prep anchor embedding
        _,(_,anchor_embed) = self.fut_encoder(anchors[:,0,:,:]) # 1,S,D
        anchor_embed = anchor_embed.repeat(N,1,1)
        env_emb = self.agent_env_encoder(env_view.reshape(-1,2)).reshape(N,360,-1)
        anchor_env_embed, _ = self.anchor_env_fuser(anchor_embed, env_emb, env_emb, need_weights=False)
        if self.env_hist_fuse:
            obs_affine, _ = self.anchor_env_fuser(obs_affine, env_emb, env_emb, need_weights=False)
        else:
            anchor_embed = anchor_embed + anchor_env_embed
        # normalize the embeddings
        obs_affine = self.normalize(obs_affine, dim=-1)
        anchor_embed = self.normalize(anchor_embed, dim=-1) 

        # compute the dot product between the two embeddings
        if self.stop_grad:
            anchor_logit = torch.sum(obs_affine * anchor_embed.detach(), dim=-1)/0.1 # detach the anchor embedding to prevent gradients flowing into the anchor prototypes, force the learning of observation embedding to align with anchor embeddings (use GT fut embedding to supervise the anchor embeddings such as auxiliary losses)
        else:
            anchor_logit = torch.sum(obs_affine * anchor_embed, dim=-1)/0.1
        anchor_prob = F.softmax(anchor_logit, dim=-1)
        if self.apply_env_filtering and (env_compliant_mask is not None):
            # mask out non-compliant anchor logits
            large_neg = -1e6
            anchor_logit = anchor_logit + (1.0 - env_compliant_mask.float()) * large_neg
            anchor_prob = F.softmax(anchor_logit, dim=-1)
        # top-k prediction
        pred_prob_k, pred_label_k = anchor_prob.topk(self.num_out, dim=1) # n_ped, num_out, 1
        pred_prob_k = pred_prob_k / torch.sum(pred_prob_k, dim=1, keepdim=True)
        pred_anchors_embed = torch.gather(anchor_embed, 1, pred_label_k.unsqueeze(-1).repeat(1,1,anchor_embed.shape[-1]))
        pred_anchors_logit = torch.gather(anchor_logit, 1, pred_label_k)
        # compute the future trajectory
        if self.stop_grad:
            obs_res = self.obs_project(obs_affine.detach()) # N,1,D detach the obs_affine to prevent gradients flowing into the obs encoder, force the learning projection layer to transform the obs embedding to correct anchor embedding (as residual to anchor embedding)
        else:
            obs_res = self.obs_project(obs_affine) # N,1,D
        fut_pred = self.decode_traj(T, pred_anchors_embed)

        output = {}
        output['anchor_prob'] = anchor_prob # N,S
        output['anchor_logit'] = anchor_logit # N,S
        output['fut_pred_norm'] = fut_pred.permute(1,0,2,3) # S,N,T,2
        output['pred_prob_k'] = pred_prob_k # N,num_out
        output['unary_logit'] = pred_anchors_logit # N,num_out

        if cache_embeddings:
            output['obs_embed'] = obs_embed # N,1,D
            output['obs_affine'] = obs_affine # N,1,D
            output['anchor_embed'] = anchor_embed # N,S,D

        if training: # for training
            training_output = {}
            training_output['obs_res'] = obs_res # N,1,D
            training_output['obs_affine'] = obs_affine # N,1,D
            training_output['obs_embed'] = obs_embed.permute(1,0,2) # N,1,D
            training_output['anchor_embed'] = anchor_embed # N,S,D
            training_output['env_emb'] = env_emb # N,360,D
            training_output['T'] = T
            output.update(training_output)
        return output


class PairwiseSelector(nn.Module):
    # Pairwise potential function to model the interaction between agents
    def __init__(self,
                 num_historical_steps:int,
                 num_modes:int,
                 radius_thresh:float,
                 collision_thresh:float,
                 hidden_dim:int=64,
                 BP_iters:int=3,
                 pairwise_potential_type:str='mlp', # distance, mlp
                 distance_type:str='cosine',
                 distance_order:int=2,
                 apply_collision_filtering:bool=False,
                 stop_grad:bool=False
                 ) -> None:
        super(PairwiseSelector, self).__init__()
        self.num_historical_steps = num_historical_steps
        self.num_modes = num_modes
        self.radius_thresh = radius_thresh
        self.collision_thresh = collision_thresh
        self.apply_collision_filtering = apply_collision_filtering
        self.stop_grad = stop_grad
        self.normalizer = TrajNorm(ori=True, rot=True, sca=False) # normalize and rotate based on last observed position

        self.norm_pred_encoder = nn.LSTM(input_size=2,
                                    hidden_size=hidden_dim,
                                    num_layers=1,
                                    batch_first=True)
        self.pred_projector = MLPLayer(hidden_dim, hidden_dim//2, hidden_dim)
        if pairwise_potential_type == 'distance':
            self.pairwise_potential = DistancePotential(type=distance_type, order=distance_order)
        elif pairwise_potential_type == 'mlp':
            self.pairwise_potential = MLPPotential(input_dim=hidden_dim*2, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown pairwise potential type: {pairwise_potential_type}")
        self.bp = BeliefPropagator(num_iters=BP_iters)

    def forward(self, data:HeteroData, unary_pred:Dict[str, torch.Tensor], batch_vec:torch.Tensor, mode:str='sum', training:bool=False) -> Dict[str, torch.Tensor]:
        """
        Compute the pairwise potential for each agent given the unary predictions and the social graph
        Args:
            data: HeteroData, input data containing the social graph
            unary_pred: Dict[str, torch.Tensor], output from the unary selector
            batch_vec: torch.Tensor, batch vector indicating the batch index for each agent
            mode: str, either "sum" (sum-product) or "max" (max-product)
        Returns:
            pairwise_potentials: torch.Tensor, pairwise potentials of shape (num_edges,)
        """
        # Define Social Graph based on predicted trajectories from unary selector
        # Baseline strategy: choosing the most-likely trajectory for each agent and define edge if future distance < threshold at any time step
        # Implementation: using torch geometric to define distance graph for each time step and then combine the graphs

        unary_liklihood = unary_pred['pred_prob_k'].clone().detach() # [N,num_out]
        unary_predictions = unary_pred['fut_pred'].clone().detach() # [K,N,T_f,2]
        unary_pred_norm = unary_pred['fut_pred_norm'].clone().detach() # [K,N,T_f,2]
        most_likely_mode = unary_liklihood.argmax(dim=1) # [N,]
        most_likely_pred = unary_predictions[most_likely_mode, torch.arange(unary_liklihood.shape[0])] # [N,T_f,2]

        K, N, T_f, _ = unary_predictions.shape
        edge_index = []
        for t in range(T_f):
            edge_idx_t = radius_graph(most_likely_pred[:,t], r=self.radius_thresh, batch=batch_vec, max_num_neighbors=32, loop=False)
            edge_index.append(edge_idx_t)
        edge_index = torch.cat(edge_index, dim=1)
        edge_index = coalesce(edge_index) # remove duplicate edges

        E = edge_index.shape[1]
        if E == 0: # no edges in graph, therefore no interaction
            output = {}
            output['pairwise_potential'] = torch.empty((0, self.num_modes, self.num_modes)).to(most_likely_pred.device)
            output['updated_unary_logit'] = unary_pred['unary_logit']
            output['edge_index'] = edge_index
            if training:
                output['collision_label'] = torch.empty((0, self.num_modes, self.num_modes)).to(most_likely_pred.device)
            return output
        
        
        # Compute pairwise potential for each edge in the graph
        # Suppose each agent has K predicted trajectories, then for each edge, we have KxK possible trajectory pairs
        # Initialize pairwise potential to zero for now [E x K x K]
        pairwise_potential = torch.zeros(len(edge_index), self.num_modes, self.num_modes).to(most_likely_pred.device)
        full_agents = data['agent']['category'] == 1
        source_hist = data['agent']['position'][full_agents][edge_index[0], :self.num_historical_steps] # [E, T_o, 2]
        self.normalizer.calculate_params(source_hist)
        source_pred_norm = torch.zeros(K, E, T_f, 2).to(most_likely_pred.device)
        target_pred_norm = torch.zeros(K, E, T_f, 2).to(most_likely_pred.device)
        for k in range(K):
            source_pred_norm[k] = self.normalizer.normalize(unary_predictions[k, edge_index[0]])
            target_pred_norm[k] = self.normalizer.normalize(unary_predictions[k, edge_index[1]])
        
        # encode the ego and neighbor trajectories
        _, (_, source_embed) = self.norm_pred_encoder(source_pred_norm.reshape(-1, T_f, 2))
        _, (_, target_embed) = self.norm_pred_encoder(target_pred_norm.reshape(-1, T_f, 2))
        source_embed = source_embed.squeeze(0).reshape(K, E, -1) # K,E,D
        target_embed = target_embed.squeeze(0).reshape(K, E, -1) # K,E,D
        source_feat = self.pred_projector(source_embed) # K,E,D
        target_feat = self.pred_projector(target_embed) # K,E,D
        pairwise_potential = self.pairwise_potential(source_feat, target_feat) # K,K,E
        pairwise_potential = pairwise_potential.permute(2,0,1) # E,K,K

        # get label of whether two prototypes collide for each edge
        dist = torch.norm(source_pred_norm.unsqueeze(1) - target_pred_norm.unsqueeze(0), dim=-1).min(dim=-1)[0].permute(2,0,1) # E,K,K
        collision_label = (dist < self.collision_thresh).float()
        collision_label = 1 - collision_label # 1 if no collision, 0 if collision

        if self.apply_collision_filtering:
            large_neg = -1e6
            pairwise_potential = pairwise_potential + (1.0 - collision_label) * large_neg
        
        # update the unary potentials using belief propagation
        updated_unary_logit = self.bp(unary_pred['unary_logit'], pairwise_potential, edge_index, mode=mode)

        output = {}
        output['pairwise_potential'] = pairwise_potential
        output['updated_unary_logit'] = updated_unary_logit
        output['edge_index'] = edge_index
        output['collision_label'] = collision_label


        return output

    def sampling(self, unary_pred:Dict[str, torch.Tensor], pair_wise_potential:Dict[str, torch.Tensor], sampling_mode:str) -> Dict[str, torch.Tensor]:
        """
        Sample the joint trajectory distribution for all agents given the unary predictions and the pairwise potentials
        Args:
            unary_pred: Dict[str, torch.Tensor], output from the unary selector
            pairwise_potential: Dict[str, torch.Tensor], output from the pairwise selector
            sampling_mode: str reranking, gibbs 
        Returns:
            joint_samples: Dict[str, torch.Tensor], sampled joint trajectories and their probabilities
        """
        updated_unary_logit = pair_wise_potential['updated_unary_logit']
        original_unary_logit = unary_pred['unary_logit']
        edge_index = pair_wise_potential['edge_index']
        pairwise_potential = pair_wise_potential['pairwise_potential']
        N, K = updated_unary_logit.shape
        device = updated_unary_logit.device

        if edge_index.shape[1] == 0: # no edges in graph, therefore no interaction
            joint_samples = {}
            joint_samples['MRF_sample'] = unary_pred['fut_pred'] # num_out,N,T_f,2
            joint_samples['MRF_sample_prob'] = F.softmax(updated_unary_logit, dim=-1) # N,num_out
            return joint_samples

        if sampling_mode == "reranking":
            # get top-k trajectories for each agent based on the updated unary potentials
            pred_prob_k, pred_label_k = F.softmax(updated_unary_logit, dim=-1).topk(self.num_modes, dim=-1) # N,num_out,1
            pred_prob_k = pred_prob_k / torch.sum(pred_prob_k, dim=1, keepdim=True)
            fut_pred = torch.gather(unary_pred['fut_pred'], 0, pred_label_k.permute(1,0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,unary_pred['fut_pred'].shape[2],unary_pred['fut_pred'].shape[3])) # num_out,N,T_f,2

            # rerank the joint trajectory samples based on the pairwise potentials
            joint_samples = {}
            joint_samples['MRF_sample'] = fut_pred # num_out,N,T_f,2
            joint_samples['MRF_sample_prob'] = pred_prob_k # N,num_out
            return joint_samples

        elif sampling_mode == "gibbs":
            # Gibbs sampling implementation 
            num_samples = self.num_modes
            num_burn_in = 10
            samples = torch.zeros(num_samples, N, dtype=torch.long, device=device)
            # Initialize current sample randomly
            curr_sample = torch.randint(0, K, (N,), device=device)
            for i in range(1, num_samples + num_burn_in):
                for j in range(N):
                    unary_logit = original_unary_logit[j].clone()  # K
                    neighbors = edge_index[1][edge_index[0] == j]
                    if len(neighbors) > 0:
                        pairwise_terms = torch.zeros(K, device=device)
                        for neighbor in neighbors:
                            neighbor_mode = curr_sample[neighbor]
                            neighbor_edge = (edge_index[0] == j) & (edge_index[1] == neighbor)
                            assert(neighbor_edge.sum() == 1)
                            pairwise_terms += pairwise_potential[neighbor_edge][0, :, neighbor_mode]
                        unary_logit += pairwise_terms
                    prob = F.softmax(unary_logit, dim=-1)
                    curr_sample[j] = torch.multinomial(prob, 1).squeeze(0)
                if i >= num_burn_in:
                    samples[i - num_burn_in] = curr_sample.clone()
            joint_samples = {}
            # Vectorized gather per-agent mode for each sample
            agent_idx = torch.arange(N, device=device).unsqueeze(0).expand(num_samples, -1)  # [S, N]
            joint_samples['MRF_sample'] = unary_pred['fut_pred'][samples, agent_idx]  # [S, N, T_f, 2]
            joint_samples['MRF_sample_prob'] = None  # Probabilities are not straightforward to compute in Gibbs sampling
            return joint_samples 
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")


class MRF(pl.LightningModule):
    def __init__(
            self,
            dataset: str,
            root: str,
            num_historical_steps:int,
            num_future_steps:int,
            num_modes:int,
            bp_iter:int,
            input_dim:int,
            hidden_dim:int,
            pl2a_radius:float,
            a2a_radius:float,
            num_freq_bands:int,
            num_agent_layers:int,
            num_heads:int,
            head_dim:int,
            dropout:float,
            lr:float,
            lr_min:float,
            weight_decay:float,
            T_max:int,
            unary_only_until:int=25,
            debug:bool=False,
            pairwise_potential_type:str='mlp', # distance, mlp
            distance_type:str='cosine',
            distance_order:int=2,
            pairwise_loss_fn:str='cross_entropy', # cross_entropy, focal
            pairwise_by_row:bool=False,
            env_hist_fuse:bool=False, # whether to fuse env with historical features or anchor features in unary selector, default to anchor features for backward compatibility
            unary_loss_fn:str='focal', # cross_entropy, focal, cosine
            unary_recon_loss='wta', # 'wta', 'gt_anchor'
            qc_encoding_only:bool=False, # whether to use only the QC encoding without the RNN hist encoding
            apply_env_penalty:bool=True,
            no_reg_losses:bool=False, # whether to skip the regularization losses,
            unary_lambda:float=100.0,
            collision_thresh:float=0.2,
            apply_collision_penalty:bool=False,
            apply_env_filtering:bool=False,
            apply_collision_filtering:bool=False,
            stop_grad:bool=False, # weather stop gradient in the unary loss component
            sampling_mode:str='reranking', # 'reranking', 'gibbs'
            save_output:bool=False,
            save_path:str=None,
            **kwargs
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_agent_layers = num_agent_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.num_modes = num_modes
        self.num_out = num_modes # predict top-k trajectories
        self.unary_only_until = unary_only_until
        # for debugging pairwise potential
        # self._debug_pairwise = debug
        self.pairwise_loss_fn = pairwise_loss_fn
        self.pairwise_by_row = pairwise_by_row
        self.env_hist_fuse = env_hist_fuse
        self.unary_recon_loss = unary_recon_loss
        self.qc_encoding_only = qc_encoding_only
        self.apply_env_penalty = apply_env_penalty
        self.no_reg_losses = no_reg_losses
        self.unary_lambda = unary_lambda
        self.collision_thresh = collision_thresh
        self.apply_collision_penalty = apply_collision_penalty
        self.apply_env_filtering = apply_env_filtering
        self.apply_collision_filtering = apply_collision_filtering
        self.sampling_mode = sampling_mode
        self.save_output = save_output
        self.save_path = save_path
        self.stop_grad = stop_grad

        self.had_env_filtering = self.apply_env_filtering
        self.had_collision_filtering = self.apply_collision_filtering

        # Load anchor data from file (common for all agents)
        anchor_path = os.path.join(root, 'anchors', 'trajectory_prototypes.npy')
        # add this to the model buffer not part of the model parameters
        self.register_buffer('prototypes', torch.from_numpy(np.load(anchor_path)).float())

        # Define trajectory normalizer
        self.normalizer = TrajNorm(ori=True, rot=True, sca=False) # normalize and rotate based on last observed position

        # Define Historical Ego Status and Social Encoder (HSSE)
        self.HSSE = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=self.num_historical_steps,
            time_span=self.num_historical_steps,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            )

        self.agent_env_encoder = FourierEmbedding(
            input_dim=2, 
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands
            )

        # Define Unary Potential Selector
        self.unary_selector = UnarySelector(
            num_out=self.num_out,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            env_hist_fuse=self.env_hist_fuse,
            qc_encoding_only=self.qc_encoding_only,
            apply_env_filtering=self.apply_env_filtering,
            )
        
        # Define Pairwise Potential Selector
        self.pairwise_selector = PairwiseSelector(
            num_historical_steps=num_historical_steps,
            num_modes=num_modes,
            radius_thresh=a2a_radius,
            collision_thresh=collision_thresh,
            hidden_dim=hidden_dim,
            BP_iters=bp_iter,
            pairwise_potential_type=pairwise_potential_type,
            distance_type=distance_type,
            distance_order=distance_order,
            apply_collision_filtering=apply_collision_filtering,
            )
        
        # define loss functions
        self.unary_loss_fn_name = unary_loss_fn
        if unary_loss_fn == 'cosine':
            self.unary_loss_fn = nn.CosineEmbeddingLoss(margin=0.0)
        else:
            self.unary_loss_fn = FocalLoss() if unary_loss_fn == 'focal' else nn.functional.cross_entropy
        self.focal_loss = FocalLoss()
        self.focal_loss_pairwise = FocalLoss(alpha=0.25, gamma=2.0)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.env_penalty = nn.BCEWithLogitsLoss()

        # fix 1 focal pairwise loss
        self.focal_pairwise_loss = FocalPairwiseLoss(
            num_modes=self.num_modes,
            alpha=0.25,
            gamma=2.0
        )
        self.a2a_collision_penalty = nn.BCEWithLogitsLoss()

        # define metrics
        self.minADE = minADE(max_guesses=self.num_modes)
        self.minFDE = minFDE(max_guesses=self.num_modes)

        self.minJADE = minJADE(max_guesses=self.num_modes)
        self.minJFDE = minJFDE(max_guesses=self.num_modes)
        self.brier_score = Brier(max_guesses=self.num_modes)

        self.ADE = ADE()
        self.FDE = FDE()

        self.coverage = Coverage(max_guesses=self.num_modes)

        
        if dataset == 'sdd':
            self.envCollision = EnvironmentalCollisionRate_SDD(
                dataset_root=root,
                scene_name=None, # for SDD, scene_name is not needed, since there is not one fixed scene
                max_guesses=self.num_modes,
                )
        else: # ETH/UCY
            scene_name = root.split('/')[-1]
            dataset_root = '/'.join(root.split('/')[:-1])
            self.envCollision = EnvironmentalCollisionRate(
                dataset_root=dataset_root,
                scene_name=scene_name,
                max_guesses=self.num_modes,
                )
        self.socialCollision = SocialCollisionRate(
            max_guesses=self.num_modes,
            min_distance=collision_thresh
        )

        if self.save_output:
            os.makedirs(self.save_path, exist_ok=True)
            self.saved_outputs = {'pred':[], 'gt':[], 'scene_id':[]} # to store the outputs during testing, 'pred': List[torch.Tensor(num_agents, num_modes, T_total, 2)], 'gt': List[torch.Tensor(num_agents, T_total, 2)]

    def on_train_start(self):
        """For training mode do not apply filtering
        """
        if self.apply_env_filtering:
            print("Detected applying environment filtering, disabling it during training.")
            self.apply_env_filtering = False
            self.unary_selector.apply_env_filtering = False

        if self.apply_collision_filtering:
            print("Detected applying collision filtering, disabling it during training.")
            self.apply_collision_filtering = False
            self.pairwise_selector.apply_collision_filtering = False
        # clear CUDA cache to prevent OOM
        torch.cuda.empty_cache()

    def on_validation_start(self):
        # clear cache
        torch.cuda.empty_cache()
    #     if self.had_env_filtering:
    #         print("Enabling environment filtering for validation.")
    #         self.apply_env_filtering = True
    #         self.unary_selector.apply_env_filtering = True

    #     if self.had_collision_filtering:
    #         print("Enabling collision filtering for validation.")
    #         self.apply_collision_filtering = True
    #         self.pairwise_selector.apply_collision_filtering = True

    def on_validation_end(self):
        # clear cache
        torch.cuda.empty_cache()
    #     if self.had_env_filtering:
    #         print("Disabling environment filtering after validation.")
    #         self.apply_env_filtering = False
    #         self.unary_selector.apply_env_filtering = False

    #     if self.had_collision_filtering:
    #         print("Disabling collision filtering after validation.")
    #         self.apply_collision_filtering = False
    #         self.pairwise_selector.apply_collision_filtering = False

    def forward(self, data:HeteroData, batch_idx:int, training:bool=False, pairwise:bool=False, pairwise_bp_mode:str='sum') -> Dict[str, torch.Tensor]:
        """Forward pass for the MRF model.

        Args:
            data (HeteroData): The input data for the model.
            batch_idx (int): The index of the current batch.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        """
        
        full_agent = data['agent']['category'] == 1
        full_agent_obs = data['agent']['position'][full_agent][:, :self.num_historical_steps].clone() # [A_f,T_o,2]
        self.normalizer.calculate_params(full_agent_obs)
        full_agent_obs_norm = self.normalizer.normalize(full_agent_obs.clone()) # [A_f,T_o,2]
        full_agent_anchor = self.prototypes.clone().unsqueeze(1).repeat(1, full_agent_obs.shape[0], 1, 1) # [S,A_f,T_f,2]

        hss_feat = self.HSSE(data, map_enc={})['x_a'][:,-1] # [A,D] take last observed step token

        headings = data['agent']['heading'][full_agent, :self.num_historical_steps].clone() # [A_f,T_o]
        range_map = data['agent']['range_map'][full_agent, :self.num_historical_steps].clone() # [A_f,T_o,360,2]
        range_map[...,0] = range_map[...,0] - headings.unsqueeze(-1) # rotate to agent-centric frame
        env_view = polar_to_cartesian(range_map.reshape(-1, 2), r_first=False) # [A_f*T_o*360,2]
        env_view = env_view.reshape(-1, self.num_historical_steps, 360, 2)
        if self.apply_env_filtering:
            full_agent_anchor_env_compliant_score = data['agent']['anchor_complaint_score'][full_agent] # [A_f,S]
            env_compliant_mask = (full_agent_anchor_env_compliant_score == 1.0) # [A_f,S] 1 if compliant, 0 otherwise

        else:
            env_compliant_mask = None
        if training:
            unary_pred = self.unary_selector.train_forward(
                norm_obs=full_agent_obs_norm,
                hss_feat=hss_feat[full_agent], 
                env_view=env_view[:, -1], # [A_f,360,2]
                anchors=full_agent_anchor,
                anchor_label=data['agent']['anchor_label'][full_agent],
                gt_fut=self.normalizer.normalize(data['agent']['position'][full_agent][:, self.num_historical_steps:].clone()),
                env_compliant_mask=env_compliant_mask,
            )
        else:
            unary_pred = self.unary_selector(
                norm_obs=full_agent_obs_norm,
                hss_feat=hss_feat[full_agent],
                env_view=env_view[:, -1],
                anchors=full_agent_anchor,
                training=training,
                env_compliant_mask=env_compliant_mask,
            )

        # unnormalize the predicted trajectory
        fut_pred = torch.zeros_like(unary_pred['fut_pred_norm'])
        for i in range(fut_pred.shape[0]):
            fut_pred[i] = self.normalizer.denormalize(unary_pred['fut_pred_norm'][i].clone())
        unary_pred['fut_pred'] = fut_pred

        if training:
            gt_anchor_pred = torch.zeros_like(unary_pred['gt_anchor_recon_norm'])
            gt_anchor_pred = self.normalizer.denormalize(unary_pred['gt_anchor_recon_norm'].clone())
            unary_pred['gt_anchor_pred'] = gt_anchor_pred
        
        if pairwise:
            batch_vec = data['agent']['batch'][full_agent]
            pairwise_pred = self.pairwise_selector(data, unary_pred, batch_vec, mode=pairwise_bp_mode, training=training)
            joint_pred = self.pairwise_selector.sampling(unary_pred, pairwise_pred, sampling_mode=self.sampling_mode)
            output = {}
            output.update(unary_pred)
            output.update(joint_pred)
            output.update(pairwise_pred)
        else:
            output = {}
            output.update(unary_pred)
        return output
    
    def pairwise_loss(self, pairwise_potential:torch.Tensor, edge_index:torch.Tensor, best_pred_idx:torch.Tensor, best_row_only: bool=False) -> torch.Tensor:
        """Compute the pairwise loss for the MRF model.

        Args:
            pairwise_potential (torch.Tensor): The pairwise potentials between agents.
            edge_index (torch.Tensor): The edge indices of the social graph.
            best_pred_idx (torch.Tensor): The indices of the best predicted trajectories for each agent.
            best_row_only (bool, optional): Whether to compute the pairwise loss only for the row corresponding to the best mode. Defaults to False.
        Returns:
            torch.Tensor: The computed pairwise loss.
        """
        # debugging pairwise potential
        # from debug_scripts.debug_pairwise import check_pairwise_loss_inputs
        if best_row_only:
            if self.pairwise_loss_fn == 'focal':
                source_idx = edge_index[0]
                target_idx = edge_index[1]
                E, K, _ = pairwise_potential.shape
                gt_pairwise_label = torch.zeros(E, K).to(pairwise_potential.device)
                gt_pairwise_label[torch.arange(E), best_pred_idx[target_idx]] = 1
                pairwise_loss = self.focal_loss_pairwise(pairwise_potential[torch.arange(E), best_pred_idx[source_idx]], gt_pairwise_label)
            else: # cross_entropy
                source_idx = edge_index[0]
                target_idx = edge_index[1]
                E, K, _ = pairwise_potential.shape
                gt_pairwise_label = torch.zeros(E, K).to(pairwise_potential.device)
                gt_pairwise_label[torch.arange(E), best_pred_idx[target_idx]] = 1
                pairwise_loss = self.cross_entropy_loss(pairwise_potential[torch.arange(E), best_pred_idx[source_idx]], gt_pairwise_label.argmax(dim=1))
        else:
            # if self._debug_pairwise:
            #     check_pairwise_loss_inputs(pairwise_potential, edge_index, best_pred_idx)
            if self.pairwise_loss_fn == 'focal':
                pairwise_loss = self.focal_pairwise_loss(pairwise_potential, edge_index, best_pred_idx)
            else: # cross_entropy
                source_idx = edge_index[0]
                target_idx = edge_index[1]
                gt_pairwise_label = torch.zeros_like(pairwise_potential)
                gt_pairwise_label[:, best_pred_idx[source_idx], best_pred_idx[target_idx]] = 1
                pairwise_loss = self.cross_entropy_loss(pairwise_potential.reshape(-1, self.num_modes*self.num_modes), gt_pairwise_label.reshape(-1, self.num_modes*self.num_modes))
        return pairwise_loss

        
    def _compute_losses(self, data: HeteroData, MRF_pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute losses for the MRF model.
        
        Args:
            data: Input data for the model
            MRF_pred: Predictions from the forward pass
            
        Returns:
            Dictionary containing all losses
        """
        full_agent = data['agent']['category'] == 1
        full_agent_obs = data['agent']['position'][full_agent][:, :self.num_historical_steps]
        self.normalizer.calculate_params(full_agent_obs)
        full_agent_fut = data['agent']['position'][full_agent][:, self.num_historical_steps:]
        full_agent_fut_norm = self.normalizer.normalize(full_agent_fut.clone())
        full_agent_anchor_label = data['agent']['anchor_label'][full_agent]
        full_agent_anchor_env_compliant_score = data['agent']['anchor_complaint_score'][full_agent]
        full_agent_anchor_env_compliant_label = (full_agent_anchor_env_compliant_score == 1.0)
        
        # Compute base losses
        if self.unary_loss_fn_name == 'cosine':
            # for cosine similarity loss, directly compute the cosine embedding loss between obs embedding and GT trajectory embedding
            obs_affine = MRF_pred['obs_affine'].squeeze(1) # [A_f, D]
            fut_embed = MRF_pred['fut_embed'].squeeze(1) # [A_f, D]
            unary_focal_loss = self.unary_loss_fn(obs_affine, fut_embed, torch.ones(obs_affine.shape[0]).to(obs_affine.device))
        else:
            unary_focal_loss = self.unary_loss_fn(MRF_pred['anchor_logit'], full_agent_anchor_label)
        fut_recon_loss = self.mse_loss(MRF_pred['fut_recon_norm'], full_agent_fut_norm)
        if self.stop_grad:
            embedding_reg_loss = self.mse_loss(MRF_pred['fut_embed'], MRF_pred['gt_anchor_embed'].detach()) # encourage the predicted trajectory embedding to be close to the GT anchor embedding by detaching the GT anchor embedding to prevent collapse
        else:
            embedding_reg_loss = self.mse_loss(MRF_pred['fut_embed'], MRF_pred['gt_anchor_embed']) # encourage the predicted trajectory embedding to be close to the GT anchor embedding
        # Compute environment penalty loss if applicable or set to 0 if not applicable
        if self.apply_env_penalty:
            # penalize non-environment-compliant anchors when GT label is 0 (non-compliant)
            env_penalty_logits = MRF_pred['anchor_logit'][~full_agent_anchor_env_compliant_label]
            if env_penalty_logits.shape[0] == 0:
                env_penalty_loss = torch.tensor(0.0).to(full_agent_fut.device)
            else:
                env_penalty_loss = self.env_penalty(env_penalty_logits, torch.zeros_like(env_penalty_logits))
        else:
            env_penalty_loss = torch.tensor(0.0).to(full_agent_fut.device)


        if self.unary_recon_loss == 'wta':
            fut_pred_regress_loss = (MRF_pred['fut_pred_norm'] - full_agent_fut_norm.unsqueeze(0)).norm(dim=-1).mean(-1).min(0)[0].mean()
        elif self.unary_recon_loss == 'gt_anchor':
            fut_pred_regress_loss = self.mse_loss(MRF_pred['gt_anchor_recon_norm'], full_agent_fut_norm)
        else:
            raise ValueError(f"Unknown unary reconstruction loss type: {self.unary_recon_loss}")
        # Compute pairwise loss if applicable
        if self.unary_only_until < self.current_epoch:
            pairwise_potential = MRF_pred['pairwise_potential']
            edge_index = MRF_pred['edge_index']
            
            if edge_index.shape[1] == 0:
                pairwise_loss = torch.tensor(0.0).to(full_agent_fut.device)
            else:
                best_pred_idx = (MRF_pred['fut_pred_norm'] - full_agent_fut_norm.unsqueeze(0)).norm(dim=-1).mean(-1).argmin(0)
                pairwise_loss = self.pairwise_loss(pairwise_potential, edge_index, best_pred_idx, self.pairwise_by_row)
        else:
            pairwise_loss = torch.tensor(0.0).to(full_agent_fut.device)

        # compute total loss with different weighting for different components, and optionally skip regularization losses
        if self.no_reg_losses:
            loss = unary_focal_loss * self.unary_lambda + fut_pred_regress_loss + pairwise_loss
        else:
            loss = unary_focal_loss * self.unary_lambda + fut_pred_regress_loss + pairwise_loss + fut_recon_loss + embedding_reg_loss 

        # add env penalty and collision penalty loss if applicable
        if self.apply_env_penalty:
            loss += env_penalty_loss

        if self.apply_collision_penalty and self.unary_only_until < self.current_epoch:
            pairwise_potential = MRF_pred['pairwise_potential']
            edge_index = MRF_pred['edge_index']
            collision_label = MRF_pred['collision_label']
            if edge_index.shape[1] == 0:
                collision_penalty_loss = torch.tensor(0.0).to(full_agent_fut.device)
            else:
                collision_penalty_potential = pairwise_potential[collision_label==0] # only consider pairs that collide
                if collision_penalty_potential.shape[0] == 0:
                    collision_penalty_loss = torch.tensor(0.0).to(full_agent_fut.device)
                else:
                    collision_penalty_label = torch.zeros_like(collision_penalty_potential).to(full_agent_fut.device)
                    collision_penalty_loss = self.a2a_collision_penalty(collision_penalty_potential.reshape(-1), collision_penalty_label.reshape(-1))
            loss += collision_penalty_loss

        # collect all loss components
        losses = {}
        losses['loss'] = loss
        losses['unary_focal_loss'] = unary_focal_loss
        losses['fut_recon_loss'] = fut_recon_loss
        losses['embedding_reg_loss'] = embedding_reg_loss
        losses['fut_pred_regress_loss'] = fut_pred_regress_loss

        if self.unary_only_until < self.current_epoch:
            losses['pairwise_loss'] = pairwise_loss

        if self.apply_env_penalty:
            losses['env_penalty_loss'] = env_penalty_loss
        
        if self.apply_collision_penalty and self.unary_only_until < self.current_epoch:
            losses['collision_penalty_loss'] = collision_penalty_loss
        
        return losses
    
    def _compute_metrics(self, data: HeteroData, MRF_pred: Dict[str, torch.Tensor], stage: str) -> None:
        """Compute and log metrics for validation/testing.
        
        Args:
            data: Input data for the model
            MRF_pred: Predictions from the forward pass
            stage: 'Val' or 'Test' for logging prefix
        """
        full_agent = data['agent']['category'] == 1
        full_agent_fut = data['agent']['position'][full_agent][:, self.num_historical_steps:]
        fut_mask = data['agent']['predict_mask'][:, self.num_historical_steps:][full_agent]
        
        pred_prob = MRF_pred['pred_prob_k']
        pred_traj = MRF_pred['fut_pred'].permute(1, 0, 2, 3)
        
        self.minADE.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        self.minFDE.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        
        self.log(f'{stage}/minADE', self.minADE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log(f'{stage}/minFDE', self.minFDE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)

        unary_anchor_prob = MRF_pred['anchor_prob'] # [A,S]
        GT_anchor_label = data['agent']['anchor_label'][full_agent] # [A,S]
        self.coverage.update(pred=unary_anchor_prob, target=GT_anchor_label)
        self.log(f'{stage}/Coverage', self.coverage, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)

        # compute the GT anchor's ADE/FDE for reference
        GT_anchor_traj = MRF_pred['gt_anchor_pred'] # [A,T_f,2]
        self.ADE.update(pred=GT_anchor_traj, target=full_agent_fut, valid_mask=fut_mask)
        self.FDE.update(pred=GT_anchor_traj, target=full_agent_fut, valid_mask=fut_mask)
        self.log(f'{stage}/GT_Anchor_ADE', self.ADE, on_epoch=True, prog_bar=False, batch_size=full_agent.sum(), sync_dist=True)
        self.log(f'{stage}/GT_Anchor_FDE', self.FDE, on_epoch=True, prog_bar=False, batch_size=full_agent.sum(), sync_dist=True)
        
        # Compute joint metrics if pairwise is enabled
        if self.unary_only_until < self.current_epoch and 'MRF_sample' in MRF_pred:
            joint_traj = MRF_pred['MRF_sample'].permute(1, 0, 2, 3)
            joint_prob = MRF_pred['MRF_sample_prob']
            
            self.minJADE.update(pred=joint_traj, target=full_agent_fut, valid_mask=fut_mask, prob=joint_prob)
            self.minJFDE.update(pred=joint_traj, target=full_agent_fut, valid_mask=fut_mask, prob=joint_prob)
            self.brier_score.update(pred=joint_traj, target=full_agent_fut, valid_mask=fut_mask, prob=joint_prob)
            
            self.log(f'{stage}/minJADE', self.minJADE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
            self.log(f'{stage}/minJFDE', self.minJFDE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
            self.log(f'{stage}/Brier', self.brier_score, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)

    def training_step(self, data: HeteroData, batch_idx: int) -> torch.Tensor:
        """Training step for the MRF model.

        Args:
            data: The input data for the model
            batch_idx: The index of the current batch

        Returns:
            The loss for the current batch
        """
        # use_pairwise = self.unary_only_until < self.current_epoch
        use_pairwise = True # alway use pairwise here, loss will be zero if before the threshold epoch so that the model can learn to predict good unary potentials first before learning pairwise interactions
        bp_mode = 'sum'
        
        MRF_pred = self(data, batch_idx, training=True, pairwise=use_pairwise, pairwise_bp_mode=bp_mode)
        losses = self._compute_losses(data, MRF_pred)
        
        # Log individual losses
        for key, value in losses.items():
            if 'loss' in key and key != 'loss':
                self.log(f'Train/{key}', value, on_epoch=True, batch_size=1, sync_dist=True)
        
        self.log('Train/Total_Loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True, batch_size=1, sync_dist=True)
        
        # # Debug pairwise potential if enabled
        # if hasattr(self, '_debug_pairwise') and self._debug_pairwise:
        #     from debug_scripts.debug_pairwise import debug_pairwise_loss_step
        #     debug_pairwise_loss_step(self, data, MRF_pred, self.current_epoch, batch_idx)
        
        return losses['loss']
    
    def validation_step(self, data: HeteroData, batch_idx: int) -> None:
        """Validation step for the MRF model.

        Args:
            data: The input data for the model
            batch_idx: The index of the current batch
        """
        use_pairwise = self.unary_only_until < self.current_epoch
        # use_pairwise = True
        bp_mode = 'max'
        
        MRF_pred = self(data, batch_idx, training=True, pairwise=use_pairwise, pairwise_bp_mode=bp_mode)
        losses = self._compute_losses(data, MRF_pred)
        
        # Log individual losses
        for key, value in losses.items():
            if 'loss' in key and key != 'loss':
                self.log(f'Val/{key}', value, on_epoch=True, batch_size=1, sync_dist=True)
        
        self.log('Val/Total_Loss', losses['loss'], on_epoch=True, prog_bar=True, batch_size=1, sync_dist=True)
        
        # Compute and log metrics
        self._compute_metrics(data, MRF_pred, stage='Val')

    def test_step(self,
                  data: HeteroData,
                  batch_idx: int) -> None:
        """Test step for the MRF model.

        Args:
            data (HeteroData): The input data for the model.
            batch_idx (int): The index of the current batch.
        """
        full_agent = data['agent']['category'] == 1
        full_agent_obs = data['agent']['position'][full_agent][:, :self.num_historical_steps] # [A,T_o,2]
        self.normalizer.calculate_params(full_agent_obs)
        full_agent_fut = data['agent']['position'][full_agent][:, self.num_historical_steps:] # [A,T_f,2]

        fut_mask = data['agent']['predict_mask'][:, self.num_historical_steps:][full_agent]

        # predictions = self(data, batch_idx, training=False, pairwise=True, pairwise_bp_mode='max')
        predictions = self(data, batch_idx, training=True, pairwise=True, pairwise_bp_mode='max') # keep training=True to keep track of GT anchor reconstruction for reference

        pred_prob = predictions['MRF_sample_prob'] # [N, K]
        pred_traj = predictions['MRF_sample'].permute(1,0,2,3) # [N, K, T_f, 2]

        anchor_prob = predictions['anchor_prob'] # [N, S]
        anchor_label = data['agent']['anchor_label'][full_agent] # [N, S]

        # compute the GT anchor's ADE/FDE for reference
        GT_anchor_traj = predictions['gt_anchor_pred'] # [A,T_f,2]
        self.ADE.update(pred=GT_anchor_traj, target=full_agent_fut, valid_mask=fut_mask)
        self.FDE.update(pred=GT_anchor_traj, target=full_agent_fut, valid_mask=fut_mask)


        self.minADE.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        self.minFDE.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        self.brier_score.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        self.minJADE.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        self.minJFDE.update(pred=pred_traj, target=full_agent_fut, valid_mask=fut_mask, prob=pred_prob)
        self.envCollision.update(pred=pred_traj, prob=pred_prob, valid_mask=fut_mask, scene_name=data['scenario_id'][0])
        self.socialCollision.update(pred=pred_traj, prob=pred_prob, valid_mask=fut_mask)
        self.coverage.update(pred=anchor_prob, target=anchor_label)


        self.log('Metric/Test/minADE', self.minADE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/minFDE', self.minFDE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/Brier', self.brier_score, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/minJADE', self.minJADE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/minJFDE', self.minJFDE, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/EnvCollision', self.envCollision, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/SocialCollision', self.socialCollision, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log('Metric/Test/Coverage', self.coverage, on_epoch=True, prog_bar=True, batch_size=full_agent.sum(), sync_dist=True)
        self.log(f'Metric/Test/GT_Anchor_ADE', self.ADE, on_epoch=True, prog_bar=False, batch_size=full_agent.sum(), sync_dist=True)
        self.log(f'Metric/Test/GT_Anchor_FDE', self.FDE, on_epoch=True, prog_bar=False, batch_size=full_agent.sum(), sync_dist=True)


        if self.save_output:
            # save the predictions and ground truth
            obs = full_agent_obs.cpu() # [A,T_o,2]
            gt_fut = full_agent_fut.cpu() # [A,T_f,2]
            pred = pred_traj.cpu() # [A,K,T_f,2]
            pred_save = torch.cat([obs.unsqueeze(1).repeat(1,self.num_modes,1,1), pred], dim=2) # [A,K,T_o+T_f,2]
            self.saved_outputs['pred'].append(pred_save)
            gt_save = torch.cat([obs, gt_fut], dim=1) # [A,T_o+T_f,2]
            self.saved_outputs['gt'].append(gt_save)
            batch_idx = data['agent']['batch'][full_agent]
            assert batch_idx.dim() == 1 # testing can only have one batch at a time
            for idx in batch_idx.unique():
                scene_id = data['scenario_id'][idx.item()]
                self.saved_outputs['scene_id'].append(scene_id)

    def analysis_step(self, data:HeteroData, batch_idx:int, training:bool=False, pairwise:bool=False, pairwise_bp_mode:str='sum') -> None:
        """A forward step for caching various embeddings for analysis (not to be used during training or testing in the pytorch lightning pipeline)

        Args:
            data (HeteroData): batch of data
            batch_idx (int): batch index
        """
        full_agent = data['agent']['category'] == 1
        full_agent_obs = data['agent']['position'][full_agent][:, :self.num_historical_steps].clone() # [A_f,T_o,2]
        self.normalizer.calculate_params(full_agent_obs)
        full_agent_obs_norm = self.normalizer.normalize(full_agent_obs.clone()) # [A_f,T_o,2]
        full_agent_anchor = self.prototypes.clone().unsqueeze(1).repeat(1, full_agent_obs.shape[0], 1, 1) # [S,A_f,T_f,2]

        hss_feat = self.HSSE(data, map_enc={})['x_a'][:,-1] # [A,D] take last observed step token

        headings = data['agent']['heading'][full_agent, :self.num_historical_steps].clone() # [A_f,T_o]
        range_map = data['agent']['range_map'][full_agent, :self.num_historical_steps].clone() # [A_f,T_o,360,2]
        heading_norm = data['agent']['heading'][full_agent, 1:self.num_historical_steps+1].clone() # [A_f,T_o] 
        range_map[...,0] = range_map[...,0] - heading_norm.unsqueeze(-1) # rotate to agent-centric frame
        env_view = polar_to_cartesian(range_map.reshape(-1, 2), r_first=False) # [A_f*T_o*360,2]
        env_view = env_view.reshape(-1, self.num_historical_steps, 360, 2)
        if self.apply_env_filtering:
            full_agent_anchor_env_compliant_score = data['agent']['anchor_complaint_score'][full_agent] # [A_f,S]
            env_compliant_mask = (full_agent_anchor_env_compliant_score == 1.0) # [A_f,S] 1 if compliant, 0 otherwise

        unary_pred = self.unary_selector(
                norm_obs=full_agent_obs_norm,
                hss_feat=hss_feat[full_agent],
                env_view=env_view[:, -1],
                anchors=full_agent_anchor,
                training=training,
                cache_embeddings=True,
                env_compliant_mask=None, # do not filter during analysis
            )
        # unnormalize the predicted trajectory
        fut_pred = torch.zeros_like(unary_pred['fut_pred_norm'])
        for i in range(fut_pred.shape[0]):
            fut_pred[i] = self.normalizer.denormalize(unary_pred['fut_pred_norm'][i].clone())
        unary_pred['fut_pred'] = fut_pred.permute(1,0,2,3) # [N,K,T_f,2] # different from training/testing, permute here for analysis convenience
        unary_pred['fut_pred_norm'] = unary_pred['fut_pred_norm'].permute(1,0,2,3) # [N,K,T_f,2] # different from training/testing, permute here for analysis convenience
        
        if pairwise: # during analysis, will not use pairwise and social filtering
            batch_vec = data['agent']['batch'][full_agent]
            pairwise_pred = self.pairwise_selector(data, unary_pred, batch_vec, mode=pairwise_bp_mode, training=training)
            joint_pred = self.pairwise_selector.sampling(unary_pred, pairwise_pred, sampling_mode=self.sampling_mode)
            output = {}
            output.update(unary_pred)
            output.update(joint_pred)
            output.update(pairwise_pred)
        else:
            output = {}
            output.update(unary_pred)

        output['obs'] = full_agent_obs
        output['obs_norm'] = full_agent_obs_norm
        output['obs_heading'] = headings
        output['output_env_view'] = env_view

        full_agent_fut = data['agent']['position'][full_agent][:, self.num_historical_steps:]
        full_agent_fut_norm = self.normalizer.normalize(full_agent_fut.clone())
        full_agent_anchor_label = data['agent']['anchor_label'][full_agent]

        output['gt_fut'] = full_agent_fut
        output['gt_fut_norm'] = full_agent_fut_norm
        output['gt_anchor_label'] = full_agent_anchor_label

    
        return output


    def on_test_end(self) -> None:
        if self.save_output:
            split = self.root.split('/')[-1]
            save_path = os.path.join(self.save_path, f'{split}.pkl')
            # directly dump the saved dict to pickle file
            with open(save_path, 'wb') as f:
                pickle.dump(self.saved_outputs, f)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('PrototypeMixer')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--num_layer', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2a_radius', type=float)
        parser.add_argument('--a2a_radius', type=float)
        parser.add_argument('--bp_iter', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_min', type=float, default=0.0)
        parser.add_argument('--unary_only_until', type=int, default=25)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=150)
        parser.add_argument('--num_modes', type=int, required=True, default=20)
        parser.add_argument('--pretrained_profiler_path', type=str, default=None)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--unary_loss_fn', type=str, default='focal', choices=['cross_entropy', 'focal', 'cosine'])
        parser.add_argument('--pairwise_loss_fn', type=str, default='cross_entropy', choices=['cross_entropy', 'focal',])
        parser.add_argument('--pairwise_by_row', action='store_true')
        parser.add_argument('--pairwise_potential_type', type=str, default='mlp', choices=['distance', 'mlp',])
        parser.add_argument('--distance_type', type=str, default='cosine', choices=['euclidean', 'cosine',])
        parser.add_argument('--distance_order', type=int, default=2)
        parser.add_argument('--env_hist_fuse', action='store_true', default=False)
        parser.add_argument('--unary_recon_loss', type=str, default='wta', choices=['wta', 'gt_anchor',]) # 1). winner-takes-all on all prediction 2). mse between gt anchor reconstruction and gt trajectory
        parser.add_argument('--qc_encoding_only', action='store_true', default=False)
        parser.add_argument('--apply_env_penalty', action='store_true', default=False)
        parser.add_argument('--apply_env_filtering', action='store_true', default=False)
        parser.add_argument('--no_reg_losses', action='store_true', default=False)
        parser.add_argument('--unary_lambda', type=float, default=100.0)
        parser.add_argument('--stop_grad', action='store_true', default=False)
        parser.add_argument('--apply_collision_penalty', action='store_true', default=False)
        parser.add_argument('--apply_collision_filtering', action='store_true', default=False)
        parser.add_argument('--collision_thresh', type=float, default=0.2)
        parser.add_argument('--sampling_mode', type=str, default='reranking', choices=['reranking', 'gibbs',])
        parser.add_argument('--save_output', action='store_true', default=False)
        parser.add_argument('--save_path', type=str, default=None)
        return parent_parser