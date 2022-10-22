import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from layers import MLP, FuncToNodeSum



class PredictorPlus(torch.nn.Module):
    def __init__(self, graph, mlp_rule_dim, gamma_fact, gamma_rule, hidden_dim, rule_dim, rnn_hidden_dim, num_layers,device):
        super(PredictorPlus, self).__init__()
        self.graph = graph
        # self.max_length = 3
        # self.rule_dim_plus = 16
        self.device = device
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size // 2
        self.padding_index = graph.relation_size // 2

        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim * 2 
        self.relation_dim = hidden_dim

        # decrease the dimension of rule embedding
        self.rule_dim = rule_dim

        self.mlp_rule_dim = mlp_rule_dim

        
        self.rule_to_entity = FuncToNodeSum(self.mlp_rule_dim)

        # self.relation_emb = torch.nn.Embedding(self.num_relations, self.mlp_rule_dim)
        self.score_model = MLP(self.mlp_rule_dim, [1]) # 128 for FB15k

        self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))
        self.epsilon = 2.0

        
        self.gamma_fact = nn.Parameter(
            torch.Tensor([gamma_fact]), 
            requires_grad=False
        )

        self.gamma_rule = nn.Parameter(
            torch.Tensor([gamma_rule]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma_fact.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        
        
        # self.entity_embedding = nn.Parameter(torch.zeros(self.num_entities, self.entity_dim))
        self.entity_embedding = torch.nn.Embedding(self.num_entities, self.entity_dim)
        

        nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # self.relation_embedding = nn.Parameter(torch.zeros(self.num_relations, self.relation_dim))
        
        self.relation_embedding = torch.nn.Embedding(self.num_relations + 1, self.relation_dim, padding_idx=self.padding_index)
    
        
        
        nn.init.uniform_(
            tensor=self.relation_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        # self.relation_embedding.weight.requires_grad = False
        # self.mlp =  nn.Linear(self.rule_dim, self.rule_dim_plus, bias=False)
        # nn.init.uniform_(self.mlp.weight.data)
        
        # RNN parameters
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.rnn = torch.nn.LSTM(self.relation_dim + self.rule_dim, self.rnn_hidden_dim, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.rnn_hidden_dim, self.relation_dim)
        # self.mlp_map_weight = torch.nn.Linear(self.relation_dim, self.rule_dim)
        
     

    def set_rules(self, input):
        # input: [rule_id, rule_head, rule_body]
        # self.rules = list()
        
        # if type(input) == list:
        #     for rule in input:
        #         rule_ = (rule[1], rule[2:])
        #         self.rules.append(rule_)
        #     logging.info('Predictor+: read {} rules from list.'.format(len(self.rules)))
        
        # elif type(input) == str:
        #     self.rules = list()
        #     with open(input, 'r') as fi:
        #         for line in fi:
        #             rule = line.strip().split()
        #             rule = [int(_) for _ in rule]
        #             rule_ = (rule[0], rule[1:])
        #             self.rules.append(rule_)
        #     logging.info('Predictor+: read {} rules from file.'.format(len(self.rules)))
        # else:
        #     raise ValueError

        logging.info('Predictor+: read {} rules from list.'.format(len(input)))
        self.num_rules = len(input)
        # rule_body's length
        self.max_length = max([len(rule[2:]) for rule in input])


        self.relation2rules = [[] for r in range(self.num_relations*2)]
        for rule in input:
            # print(rule)
            relation = rule[1]
            self.relation2rules[relation].append([rule[0], (rule[1], rule[2:])])
        

        self.rule_features = []
        self.rule_masks = []
        for rule in input:
            rule_ = rule + [self.padding_index for i in range(self.max_length - len(rule[2:]))]
            self.rule_features.append(rule_)
            self.rule_mask = torch.zeros_like(torch.tensor(rule_))[2:].bool()
            self.rule_mask[(len(rule[2:]))-1] = True
            self.rule_masks.append(self.rule_mask)

        self.rule_masks = torch.stack(self.rule_masks)
        self.rule_features = torch.tensor(self.rule_features, dtype=torch.long)

        self.rules_num = torch.zeros(self.num_rules, device = self.device)

        self.mlp_feature = nn.Parameter(torch.zeros(self.num_rules, self.mlp_rule_dim))
        nn.init.kaiming_uniform_(self.mlp_feature, a=math.sqrt(5), mode="fan_in")
        # self.rule_emb = nn.parameter.Parameter(torch.zeros(self.num_rules, self.rule_dim))
        self.rule_emb = torch.nn.Embedding(self.num_rules, self.rule_dim)
        nn.init.kaiming_uniform_(self.rule_emb.weight, a=math.sqrt(5), mode="fan_in")
     
   

    def compute_ruleE(self,sample,mode='single-rule'):

        if mode == 'single-rule':
            rule, mask = sample
            score = self.RNN_ruleE(rule.unsqueeze(1),mask)

        elif mode == 'rule-batch':
            pos_part, mask,  neg_idx, neg_part = sample
            batch_size, negative_sample_size = neg_idx.size(0), neg_idx.size(1)
            
            pos_part = pos_part.unsqueeze(dim=1).repeat(1,negative_sample_size,1)
            
            neg_idx = neg_idx.unsqueeze(dim=2) + 1
            neg_part = neg_part.unsqueeze(dim=2)
            rule_sample = pos_part.scatter(2, neg_idx, neg_part)
            score = self.RNN_ruleE(rule_sample, mask)
            
        return score

    def compute_ruleE_g(self,sample):

        
        rule, mask = sample
        score = self.RNN_ruleE(rule.unsqueeze(1),mask)

    
            
        return score

    def compute_KGE(self, sample, mode='single'):
        
        if mode == 'single':
            relations_flag = torch.pow(-1,sample[:,1] // (self.num_relations)).unsqueeze(-1)
            relation_com = sample[:,1] % (self.num_relations)
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = self.entity_embedding(sample[:,0]).unsqueeze(1)
            relation = (self.relation_embedding(relation_com) * relations_flag).unsqueeze(1)
            tail = self.entity_embedding(sample[:,2]).unsqueeze(1)
            

        elif mode == 'head-batch':
            
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            relations_flag = torch.pow(-1,tail_part[:,1] // (self.num_relations )).unsqueeze(-1)
            relation_com = tail_part[:1] % (self.num_relations )
            head = self.entity_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = (self.relation_embedding(relation_com) * relations_flag).unsqueeze(1)
            tail = self.entity_embedding(tail_part[:,2]).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            relations_flag = torch.pow(-1,head_part[:,1] // (self.num_relations)).unsqueeze(-1)
            relation_com = head_part[:,1] % (self.num_relations)
            head = self.entity_embedding(head_part[:,0]).unsqueeze(1)
            
            relation = (self.relation_embedding(relation_com)*relations_flag).unsqueeze(1)
            
            tail = self.entity_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            

        else:
            raise ValueError('mode %s not supported' % mode)

        return self.RotatE(head,relation,tail,mode)

    def compute_g_KGE(self,all_h,all_r):

        all_t = torch.arange(0,self.num_entities,device=all_h.device).unsqueeze(0).repeat(all_h.size(0),1)
        relations_flag = torch.pow(-1,all_r // (self.num_relations)).unsqueeze(-1)
        all_r = all_r % (self.num_relations)
        head = self.entity_embedding(all_h).unsqueeze(1)

        relation = (self.relation_embedding(all_r)*relations_flag).unsqueeze(1)
        tail = self.entity_embedding(all_t.view(-1)).view(all_h.size(0), self.num_entities, -1)
        
        return self.RotatE(head,relation,tail,'tail-batch')

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma_fact.item() - torch.norm(score, p=1, dim=2)
        return score


    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        
        score = self.gamma_rule.item() - score.sum(dim=2)
        
        return score
    


    def RNN_ruleE(self, rules, mask):
        
        inputs = rules[:,:,2:]
        
        relations_flag = torch.pow(-1,inputs // (self.num_relations)).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations*2, self.padding_index, inputs_com)
        
        rule_embedding = self.rule_emb(rules[:,:,0]).unsqueeze(2).expand(-1, -1, inputs.size(2),-1)
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1,rules[:,:,1] // (self.num_relations)).unsqueeze(-1)
        embedding_r *= relations_flag

        embedding = torch.cat([embedding, rule_embedding], dim=-1)
        embedding = embedding.view(-1,inputs.size(2),self.hidden_dim+ self.rule_dim)
        
        outputs, hidden = self.rnn(embedding)
        outputs = outputs.view(inputs.size(0),inputs.size(1),inputs.size(2),-1)
        rule_mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1,outputs.size(1),1,outputs.size(-1))
        
        last_out = outputs[rule_mask].view(outputs.size(0),outputs.size(1),-1)
        derived_embedding = self.linear(last_out)

        dist = self.gamma_rule.item() - torch.norm((derived_embedding - embedding_r), p=2, dim=-1)
        
        return dist

    def RNN_ruleE_g(self, rules, mask):
        inputs = rules[:,:,2:]
        
        relations_flag = torch.pow(-1,inputs // (self.num_relations)).unsqueeze(-1)
        inputs_com = inputs % self.num_relations

        inputs_com = torch.where(inputs==self.num_relations*2, self.padding_index, inputs_com)
        
        rule_embedding = self.rule_emb(rules[:,:,0]).unsqueeze(2).expand(-1, -1, inputs.size(2),-1)
        embedding = self.relation_embedding(inputs_com) * relations_flag
        
        # rule_head
        embedding_r = self.relation_embedding(rules[:,:,1]%self.num_relations)
        relations_flag = torch.pow(-1,rules[:,:,1] // (self.num_relations)).unsqueeze(-1)
        embedding_r *= relations_flag

        embedding = torch.cat([embedding, rule_embedding], dim=-1)
        embedding = embedding.view(-1,inputs.size(2),self.hidden_dim+ self.rule_dim)
        
        outputs, hidden = self.rnn(embedding)
        outputs = outputs.view(inputs.size(0),inputs.size(1),inputs.size(2),-1)
        rule_mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1,outputs.size(1),1,outputs.size(-1))
        
        last_out = outputs[rule_mask].view(outputs.size(0),outputs.size(1),-1)
        derived_embedding = self.linear(last_out)
        dist_emb = self.gamma_rule.item()/self.relation_dim - torch.pow((derived_embedding - embedding_r),2)
        # dist_emb = self.gamma_rule.item() - torch.pow((derived_embedding - embedding_r),2)
        
        
        # dist = self.gamma_rule.item() - torch.norm((derived_embedding - embedding_r), p=2, dim=-1)
        
        return dist_emb


    def forward(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        rule_index = list()
        rule_count = list()
        # rules_score = list()
        
        mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()
            
            # one_count = torch.where(count>0,1,count)
            # path_weight = self.soft_grounding(all_h,r_body)
        
            
            mask += count

            rule_index.append(index)
            rule_count.append(count)
            # rule_score = self.compute_rule_score(r_head,r_body,index)
            # rules_score.append(rule_score)

        # positive_rule_score = torch.stack(rules_score)
        # positive_rule_score = F.logsigmoid(positive_rule_score)
        # rule_loss = - positive_rule_score.mean()
        if mask.sum().item() == 0:
            # return mask + self.bias.unsqueeze(0), (1 - mask).bool(), torch.zeros_like(rule_loss)
            return mask + self.bias.unsqueeze(0), (1 - mask).bool()


        candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]

        # id = 0 ,则表示为第几个三元组的grounding情况
        # batch_id_of_candidate = torch.div(candidate_set, self.graph.entity_size, rounding_mode='floor')
        # kge_score = self.compute_g_KGE(all_h,all_r).view(-1)[candidate_set].unsqueeze(-1)

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)

        # smooth the number of grounding path
        # rule_count = torch.tanh(rule_count)
        # rule_count = 2 * torch.sigmoid(rule_count)
        # rule_count = torch.where(rule_count.float() > torch.tensor(0,device=rule_index.device).float(), torch.tensor(1,device=rule_index.device).float(), rule_count)
        # print("rule_count_var: ",rule_count.var(unbiased=False).item())
        
        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]
        
        # rule_emb = self.map_to_scalar(self.rule_emb(rule_index)) * self.mlp_rule_emb(rule_index)
        # rule_emb = (self.rule_emb(rule_index) * self.mlp_feature[rule_index])
        # rule_emb = torch.sum((self.rule_emb(rule_index) * self.mlp_feature[rule_index]),dim=0,keepdim=True)
        # rule_emb = torch.matmul(self.rules_weight_emb[rule_index].t_(),self.mlp_feature[rule_index])
        # rule_emb = self.rules_weight_emb[rule_index].norm(p=2,dim=1,keepdim=True)
        rule_emb = self.rules_weight_emb[rule_index]

        # rule_emb = positive_rule_score
        # rule_emb = self.mlp_map_weight(positive_rule_score)
        
        # rule_emb = self.rule_emb(rule_index)

        mlp_feature = self.mlp_feature[rule_index] * rule_emb.unsqueeze(-1)
        # mlp_feature = self.mlp_feature[rule_index]

       
        
        output = self.rule_to_entity.forward_t(rule_count, mlp_feature)
        # output = self.rule_to_entity(rule_count, rule_emb, mlp_feature)
        # feature = torch.zeros((rule_count.size(1),self.mlp_rule_dim),device=self.device)
        
        # for i in range(self.rules_weight_emb[rule_index].t_().size(0)):

        #     rule_weight = self.rules_weight_emb[rule_index].t_()[i].unsqueeze(-1)
        #     # rule_emb = torch.matmul(rule_weight.unsqueeze(0),self.mlp_feature[i][rule_index])
        #     mlp_feature = self.mlp_feature[rule_index]
        #     rule_emb = rule_weight * mlp_feature
        #     output = self.rule_to_entity.forward_t(rule_count, rule_emb)
        #     feature += output
            
        # rel = self.relation_emb(all_r[0]).unsqueeze(0).expand(output.size(0), -1)
        # feature = torch.cat([output, rel], dim=-1)
        # feature = torch.cat([output],dim=-1)
        feature = output

        output = self.score_model(feature).squeeze(-1)
        # output = output.squeeze(-1)
        score = torch.zeros(all_h.size(0) * self.graph.entity_size, device=device)
        score.scatter_(0, candidate_set, output)
        score = score.view(all_h.size(0), self.graph.entity_size)
        score = score + self.bias.unsqueeze(0)
        mask = torch.ones_like(mask).bool()
        
        # kge_score = self.compute_g_KGE(all_h,all_r)
        # score += 0.5 * kge_score

        return score, mask

    # during grounding process, we one time compute the rule score on rule embedding and relation embedding
    def eval_compute_rule_weight(self,prior):
        # self.rules_weight_emb = torch.tensor(prior,device=self.device)
        batch = 128
        self.rule_masks = self.rule_masks.to(self.device)
        self.rule_features = self.rule_features.to(self.device)
        split_num = self.rule_features.size(0) // batch 
        rule_batches = torch.split(self.rule_features, split_num, 0)
        rule_mask_batches = torch.split(self.rule_masks, split_num, 0)
        rules_weight_emb = list()

        for rules, rules_mask in zip(rule_batches, rule_mask_batches):

            rule_weight_emb = self.RNN_ruleE_g(rules.unsqueeze(1),rules_mask).squeeze(1)
            # rule_score = self.compute_ruleE_g((rules, rules_mask)).squeeze(-1)
            
            rules_weight_emb.append(rule_weight_emb)

        rules_weight_emb = torch.cat(rules_weight_emb).sum(-1)
        # self.rules_weight_emb = torch.ones_like(rules_weight_emb)

    def soft_grounding(self, all_h, r_body):

        r_body = torch.tensor(r_body,device=self.entity_embedding.weight.device)
        all_t = torch.arange(0,self.num_entities,device=all_h.device).unsqueeze(0).repeat(all_h.size(0),1)
        head = self.entity_embedding(all_h).unsqueeze(1)
        tail = self.entity_embedding(all_t)


        rule_body = torch.zeros_like(self.relation_embedding(r_body[0]))
        for relation in r_body:
            rule_body += self.relation_embedding(relation)

        rule_body = rule_body.unsqueeze(0).repeat(head.size(0),1).unsqueeze(1)
        # self.gamma_fact.data = torch.tensor(8.0)

        path_weight = self.RotatE(head,rule_body,tail,'tail-batch')

        return torch.sigmoid(path_weight)
