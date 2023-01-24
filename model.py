from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

def concept_edge_list4GCN():
    node2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
    f=open('conceptnet_edges2nd.txt',encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).cuda()

def find_neighbour_hop(source_node, ad_matrix, already_set, node_set, kg, hop):
    if hop == 9: 
        return
    if len(node_set) == 0:
        return
    else:
        hop += 1
    next_node_set = set()
    for node in node_set:
        for relation, entity in kg[node]:
            if hop >= 5:
                ad_matrix[source_node][entity] = 1
            next_node_set.add(entity)
            already_set.add(entity)
    find_neighbour_hop(source_node, ad_matrix, already_set, next_node_set, kg, hop)

class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        # self.pad_idx = dictionary[dictionary.null_token]
        # self.start_idx = dictionary[dictionary.start_token]
        # self.end_idx = dictionary[dictionary.end_token]
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        
        self.batch_size = opt['batch_size']
        self.n_entity = opt['n_entity']
        self.max_r_length = opt['max_r_length']
        
        self.mlp = MLP(opt['dim'] * 2, opt['dim'])

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg = pkl.load(
            open("data/subkg.pkl", "rb")
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.db_kg_cross_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_2 = SelfAttentionLayer(opt['dim'], opt['dim'])
        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])

        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.user_norm2 = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm2 = nn.Linear(opt['dim'], 1)
        self.user_norm3 = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm3 = nn.Linear(opt['dim'], 1)
        self.gate_norm_a = nn.Linear(opt['dim'], opt['dim'])
        self.gate_four = nn.Linear(opt['dim']*3, opt['dim'])
        self.l_2dim = nn.Linear(opt['dim']*2, opt['dim'])
        
        self.copy_norm_1 = nn.Linear(opt['embedding_size']*2+opt['embedding_size']+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_db_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)

        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        edge_list = list(set(edge_list))
        
        self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases'])
        #self.dbpedia_RGCN=RGCNConv(self.dim, self.dim, self.n_relation, num_bases=opt['num_bases'])
        #self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.n_con_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)
        
        #self.a = nn.Parameter(torch.zeros(size=(2 * opt['dim'], opt['dim'])))
        #self.con_b = nn.Parameter(torch.zeros(size=(self.dim, 1)))
        self.cross_att_w = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        self.gate_bias = nn.Parameter(torch.FloatTensor([0]))

        print(len(edge_list) + len(self.concept_edge_sets[0]), self.n_relation)
        
        #self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        w2i=json.load(open('word2index_redial.json',encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}

        self.mask4key=torch.Tensor(np.load('mask4key.npy')).cuda()
        self.mask4movie=torch.Tensor(np.load('mask4movie.npy')).cuda()
        self.mask4=self.mask4key+self.mask4movie
        
        #self.adjacent_hop = torch.zeros(opt['n_entity'], opt['n_entity'])
        #for i in tqdm(range(opt['n_entity'])):
        #    node_set = set()
        #    already_set = set()
        #    node_set.add(i)
        #    find_neighbour_hop(i, self.adjacent_hop, already_set, node_set, self.kg, 0)
        
        if is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_db.parameters(), self.self_attn_2.parameters(), self.user_norm.parameters(),
                      self.gate_norm.parameters(), self.output_en.parameters(), self.cross_att_w, self.user_norm2.parameters(), self.gate_norm2.parameters(), self.user_norm3.parameters(), self.gate_norm3.parameters(), self.gate_bias]
            for param in params:
                try:
                    for pa in param:
                        pa.requires_grad = False
                except:
                    param.requires_grad = False

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().cpu().expand(bsz, 1)

    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz, maxlen, db_kg_cross_att):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz).cuda()
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            #batch*1*hidden
            scores = scores[:, -1:, :]
            #scores = self.output(scores)
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            
            db_attn_norm = self.db_attn_norm(attention_db)
            db_kg_cross_att_latent = self.db_kg_cross_attn_norm(db_kg_cross_att)

            copy_latent = self.copy_norm_1(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), db_kg_cross_att_latent.unsqueeze(1), scores], -1))

            # logits = self.output(latent)
            con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
            voc_logits = F.linear(scores, self.embeddings.weight)
            # print(logits.size())
            # print(mem_logits.size())
            #gate = F.sigmoid(self.gen_gate_norm(scores))

            sum_logits = voc_logits + con_logits #* (1 - gate)
            _, preds = sum_logits.max(dim=-1)
            #scores = F.linear(scores, self.embeddings.weight)


            #_, preds = sum_logits.max(dim=-1)
            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys, db_kg_cross_att):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz).cuda(), inputs.long()], 1)
        inputs = inputs.cuda()
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db) #batch*r_l*hidden

        kg_attention_latent=self.kg_attn_norm(attention_kg)

        #map=torch.bmm(latent,torch.transpose(kg_embs_norm,2,1))
        #map_mask=((1-encoder_states_kg[1].float())*(-1e30)).unsqueeze(1)
        #attention_map=F.softmax(map*map_mask,dim=-1)
        #attention_latent=torch.bmm(attention_map,encoder_states_kg[0])

        db_attention_latent=self.db_attn_norm(attention_db)
        db_kg_cross_att_latent = self.db_kg_cross_attn_norm(db_kg_cross_att)

        #db_map=torch.bmm(latent,torch.transpose(db_embs_norm,2,1))
        #db_map_mask=((1-encoder_states_db[1].float())*(-1e30)).unsqueeze(1)
        #db_attention_map=F.softmax(db_map*db_map_mask,dim=-1)
        #db_attention_latent=torch.bmm(db_attention_map,encoder_states_db[0])

        #print(kg_attention_latent.size(), db_attention_latent.size())
        copy_latent=self.copy_norm_1(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_kg_cross_att_latent.unsqueeze(1).repeat(1,seqlen,1), latent],-1))

        #logits = self.output(latent)
        con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)
        # print(logits.size())
        # print(mem_logits.size())
        #gate=F.sigmoid(self.gen_gate_norm(latent))

        sum_logits = logits+con_logits#*(1-gate)
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, con_label, db_label, mask):
        #batch*dim
        #node_count*dim
        con_emb=self.info_con_norm(con_user_emb)
        db_emb=self.info_db_norm(db_user_emb)
        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.cuda().float()),dim=-1)*mask.cuda()
        info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.cuda().float()),dim=-1)*mask.cuda()

        return torch.mean(info_db_loss), torch.mean(info_con_loss)

    def forward(self, xs, ys, mask_ys, concept_mask, db_mask, seed_sets, labels, con_label, db_label, entity_vector, rec, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None, epoch = 0):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        if test == False:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        #xxs = self.embeddings(xs)
        #mask=xs == self.pad_idx
        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs.long())

        # graph network
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        user_representation_list = []
        db_representation_list = []
        db_con_mask=[]
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                db_representation_list.append(torch.zeros(self.dim).cuda())
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  # torch can reflect
            db_representation_list.append(user_representation)
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb=torch.stack(user_representation_list)
        #db_emb = torch.stack(db_representation_list)
        db_con_mask=torch.stack(db_con_mask)

        graph_con_emb=con_nodes_features[concept_mask.long()]
        con_emb_mask=concept_mask==self.concept_padding

        con_user_emb=graph_con_emb
        cross_att_to_con_list = []
        cross_att_to_db_list = []
        cross_list = []
        C_D_list = []
        for i, db_item in enumerate(db_representation_list):
            
            cross_attention = torch.matmul(db_item, self.cross_att_w)
            cross_attention = torch.matmul(cross_attention, con_user_emb[i].permute(1,0))
            cross_attention_m = F.softmax(cross_attention, dim=-1)
            cross_attention_n = F.softmax(cross_attention, dim=0)
            cross_att_to_con = torch.matmul(cross_attention_m, con_user_emb[i])
            if len(cross_attention.size()) == 1:
                #print(db_item.size(), cross_attention.unsqueeze(-1).size())
                cross_att_to_db = torch.matmul(cross_attention_n.unsqueeze(-1), db_item.unsqueeze(0))
            #print(cross_attention.size())
            else:
                cross_att_to_db = torch.matmul(cross_attention_n.permute(1,0), db_item)
            ###if len(cross_att_to_con.size()) == 1:
            ###    cross_att_to_con = F.softmax(cross_att_to_con, dim=-1)
            ###else:
            ###    cross_att_to_con = F.softmax(torch.sum(F.softmax(cross_att_to_con, dim=-1), dim=-2), dim=-1)
            ###cross_att_to_db = F.softmax(torch.sum(F.softmax(cross_att_to_db, dim=-1), dim=-2), dim=-1)
            #C_D = torch.matmul(cross_attention_m, torch.cat([con_user_emb[i], cross_att_to_db], dim=-1)) 
            C_D = torch.matmul(cross_attention_m, cross_att_to_db)
            #print(C_D.size())
            if len(C_D.size()) == 1:
                C_D_list.append(self.self_attn_2(C_D.unsqueeze(0)).cuda())
            else:
                C_D_list.append(self.self_attn_2(C_D).cuda())
            cross_att_to_con_list.append(cross_att_to_con)
            cross_att_to_db_list.append(cross_att_to_db)
            cross_list.append(cross_attention)
        
        #cross_att_to_con_list = torch.stack(cross_att_to_con_list)
        #cross_att_to_db_list = torch.stack(cross_att_to_db_list)
        C_D_list = torch.stack(C_D_list)
        
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda())
        #C_D_list= self.l_2dim(C_D_list)
        
        
        user_emb2=self.user_norm2(torch.cat([con_user_emb,db_user_emb],dim=-1))
        uc_gate2 = F.sigmoid(self.gate_norm2(user_emb2))
        user_emb2 = uc_gate2 * db_user_emb + (1 - uc_gate2) * con_user_emb
        
        user_emb = C_D_list
        #print(C_D.size(), db_user_emb.size())
        #user_emb_four = F.softmax(self.gate_four(torch.cat([db_user_emb, con_user_emb, cross_att_to_db_list, cross_att_to_con_list], dim=-1)), dim=-1)
        #print(user_emb_four.size(), db_user_emb.size(),user_emb_four[:,0].unsqueeze(-1).size())
        #user_emb_four = user_emb_four[:,0].unsqueeze(-1) * db_user_emb +  user_emb_four[:,1].unsqueeze(-1) * con_user_emb + user_emb_four[:,2].unsqueeze(-1) * cross_att_to_db_list + user_emb_four[:,3].unsqueeze(-1) * cross_att_to_con_list
        user_emb3= self.user_norm3(torch.cat([user_emb,user_emb2],dim=-1))
        uc_gate3 = F.sigmoid(self.gate_norm3(user_emb3))
        user_emb3 = uc_gate3 * user_emb + (1 - uc_gate3) * user_emb2
        
        
        
        entity_scores = F.linear(user_emb3, db_nodes_features, self.output_en.bias)
        
        

        
        negative_sample = np.random.randint(self.n_entity, size=(labels.size()[0],10))
        #print(negative_sample[0])
        
        softmax_entity_scores = F.softmax(entity_scores, dim=-1)
        complementation_softmax_entity_scores = 1 -softmax_entity_scores
        
        log_softmax_entity_scores = torch.log(softmax_entity_scores)
        negative_loss = softmax_entity_scores * log_softmax_entity_scores
        negative_loss_sum = 0
        
        if True:
            for batch_index, item in enumerate(negative_sample):
                negative_loss_sum += 1 * torch.sum(negative_loss[batch_index][item])
                
        
        #mask loss
        #m_emb=db_nodes_features[labels.cuda()]
        #mask_mask=concept_mask!=self.concept_padding
        mask_loss=0#self.mask_predict_loss(m_emb, attention, xs, mask_mask.cuda(),rec.float())

        info_db_loss, info_con_loss=self.infomax_loss(con_nodes_features,db_nodes_features,con_user_emb,db_user_emb,con_label,db_label,db_con_mask)

        #entity_scores = F.softmax(entity_scores.cuda(), dim=-1).cuda()

        rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda())
        #rec_loss=self.klloss(entity_scores.squeeze(1).squeeze(1).float(), labels.float().cuda())
        #rec_loss_copy = rec_loss_copy
        rec_loss = torch.sum(rec_loss*rec.float().cuda()) + negative_loss_sum

        #self.user_rep=user_emb

        #generation---------------------------------------------------------------------------------------------------
        con_nodes_features4gen=con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        con_emb4gen = con_nodes_features4gen[concept_mask.long()]
        con_mask4gen = concept_mask != self.concept_padding
        #kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda())

        db_emb4gen=db_nodes_features[entity_vector.long()] #batch*50*dim
        db_mask4gen=entity_vector!=0
        #db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.cuda())

        if test == False:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, mask_ys, C_D_list)
            gen_loss = torch.mean(self.compute_loss(scores, mask_ys.long()))

        else:
            scores, preds = self.decode_greedy(
                encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,
                bsz,
                maxlen or self.longest_label,
                C_D_list
            )
            gen_loss = None

        return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss # ,rec_loss_copy #, cross_list

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        return loss

    def save_model(self, epoch = None):
        if epoch ==  None:
            torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')
        else:
            model_save_path = 'saved_model/' + str(epoch) + '_net_parameter1.pkl'
            torch.save(self.state_dict(), model_save_path)

    def load_model(self, epoch = None):
        if epoch == None:
            self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'), strict=False)
        else:
            model_save_path = 'saved_model/' + str(epoch) + '_net_parameter1.pkl'
            self.load_state_dict(torch.load(model_save_path), strict=False)
        
    def save_pretrained_model(self):
        torch.save(self.state_dict(), 'saved_model/pretrained_net_parameter1.pkl')

    def load_pretrained_model(self):
        self.load_state_dict(torch.load('saved_model/pretrained_net_parameter1.pkl'), strict=False)

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output
