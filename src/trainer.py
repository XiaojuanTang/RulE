from turtle import pos
import comm
from utils import *
import torch
from torch import device
from torch import nn
from torch.utils import data as torch_data
from itertools import islice
from data import Iterator, RulERuleDataset
import torch.nn.functional as F
import wandb

class TrainerPredictor(object):
    
    def __init__(self, model, train_set, valid_set, test_set, optimizer, gpus=None, num_worker=0):
        self.device = device
        self.gpus = gpus
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(gpus)

        
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        

    def train(self, batch_per_epoch, smoothing, print_every):
        
        logging.info('>>>>> Predictor: Training')
        
        self.train_set.make_batches()
        
        dataloader = torch_data.DataLoader(self.train_set, 1, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        
            
        model.train()

        total_loss = 0.0
        total_size = 0.0

        

        for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
                target_t = target_t.cuda(device=self.device)
               
            target = target * smoothing + target_t * (1 - smoothing)
            
            rule_score, mask = model(all_h, all_r, edges_to_remove)

           

            if mask.sum().item() != 0:
                rule_logits = (torch.softmax(rule_score, dim=1) + 1e-8).log()
                # kge_logits = (torch.softmax(kge_score, dim=1) + 1e-8).log()

                rule_loss = -(rule_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                # kge_loss = -(kge_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)

                # loss = rule_loss + kge_loss
                loss = rule_loss
                # wandb.log({'train/rule_loss':rule_loss, 'train/kge_loss':kge_loss})
                wandb.log({'train/rule_loss':rule_loss})
                
                # loss = kg_loss
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
            
            if (batch_id + 1) % print_every == 0:
                
                if comm.get_rank() == 0:
                    
                    # logging.info('kg_loss:{} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), total_loss / print_every, total_size / print_every))
                    # logging.info('rule_loss:{} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), total_loss / print_every, total_size / print_every))
                    logging.info('loss:    {} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), loss, total_size / print_every))
                    # logging.info('rule_loss:  {} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), rule_loss, total_size / print_every))

                total_loss = 0.0
                total_size = 0.0

    @torch.no_grad()
    def evaluate(self, split, alpha=3.0, expectation=True):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        
        dataloader = torch_data.DataLoader(test_set, 1,num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        
        for batch in dataloader:

            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            logits, mask = model(all_h, all_r, None)
            # kge_score = model.compute_g_KGE(all_h,all_r)
            # logits += alpha * kge_score

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
     
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        if comm.get_rank() == 0:
            logging.info('Data : {}'.format(len(query2LH)))
            logging.info('Hit1 : {:.6f}'.format(hit1))
            logging.info('Hit3 : {:.6f}'.format(hit3))
            logging.info('Hit10: {:.6f}'.format(hit10))
            logging.info('MR   : {:.6f}'.format(mr))
            logging.info('MRR  : {:.6f}'.format(mrr))
        
        if split == 'valid':
        
            wandb.log({'valid/mrr': mrr, 'valid/hit@1':hit1, 'valid/hit3':hit3, 'valid/hit10':hit10})
        
        return mrr

 
    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, args, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        # if comm.get_rank() == 0:
        logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
       
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
       
        torch.save(state, checkpoint)

        g_rule_embedding = self.model.mlp_feature.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'g_rule_embedding'), 
            g_rule_embedding
        )


class TrainerRuleE(object):

    def __init__(self, graph, model, train_set_rule, train_set, valid_set, test_set, test_set_data, train_iter, rule_set, expectation, device, num_worker):
        
        self.expectation = expectation

        self.train_set_rule = train_set_rule
        self.num_worker = num_worker
        self.device = device
        logging.info("Preprocess training set")
       
       
        self.graph = graph
        self.model = model
        self.train_iter = train_iter
        self.rule_set = rule_set
        
        
        
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.test_set_data = test_set_data
    
    def train(self, max_steps, args):
        
        # Set training configuration
        dataloader = torch_data.DataLoader(self.rule_set, args.rule_batch_size, shuffle=True, collate_fn=RulERuleDataset.collate_fn)
        iterator = Iterator(dataloader)
        current_learning_rate = float(args.learning_rate)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=current_learning_rate
        )

        
        logging.info('>>>>> ruleE: Training')
        training_logs = []
        best_mrr = 0.0

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        for step in range(0, max_steps+1):

            log = self.train_step( optimizer, self.train_iter, iterator, args)
            
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                mrr = self.evaluate("valid",self.expectation )
                if mrr > best_mrr:
                    save_model(self.model,optimizer, args)
                    best_mrr = mrr

    def train_step(self, optimizer, train_iterator, train_iterator_rule, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model = self.model
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        positive_rule, negative_idx, negative_rule, mode_rule, rule_mask = next(train_iterator_rule)

        if args.cuda:
            positive_sample = positive_sample.cuda(self.device)
            negative_sample = negative_sample.cuda(self.device)
            positive_rule = positive_rule.cuda(self.device)
            negative_idx = negative_idx.cuda(self.device)
            negative_rule = negative_rule.cuda(self.device)
            rule_mask = rule_mask.cuda(self.device)
            subsampling_weight = subsampling_weight.cuda(self.device)

        negative_fact_score = model.compute_KGE((positive_sample, negative_sample), mode=mode) 
        negative_rule_score = model.compute_ruleE((positive_rule,  rule_mask, negative_idx, negative_rule),mode=mode_rule) 

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_fact_score = (F.softmax(negative_fact_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_fact_score)).sum(dim = 1)
            negative_rule_score = (F.softmax(negative_rule_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_rule_score)).sum(dim = 1)
        else:
            negative_fact_score = F.logsigmoid(-negative_fact_score).mean(dim = 1)
            negative_rule_score = F.logsigmoid(-negative_rule_score).mean(dim = 1)


        positive_fact_score = model.compute_KGE(positive_sample)
        positive_rule_score = model.compute_ruleE((positive_rule,rule_mask), mode='single-rule')


        positive_fact_score = F.logsigmoid(positive_fact_score).squeeze(dim = 1)
        positive_rule_score = F.logsigmoid(positive_rule_score)

        negative_rule_score_weight = negative_rule_score 
        positive_rule_score_weight = positive_rule_score 

        if args.uni_weight:
            positive_fact_loss = - positive_fact_score.mean() 
            negative_fact_loss = - negative_fact_score.mean()
        else:
            positive_fact_loss = - (subsampling_weight * positive_fact_score).sum()/subsampling_weight.sum()            
            negative_fact_loss = - (subsampling_weight * negative_fact_score).sum()/subsampling_weight.sum() 

        
        positive_rule_loss = - positive_rule_score_weight.mean() * args.weight_rule
        negative_rule_loss = - negative_rule_score_weight.mean() * args.weight_rule


        loss_fact = (positive_fact_loss + negative_fact_loss)/2
        loss_rule = (positive_rule_loss + negative_rule_loss)/2
       
        loss = loss_rule + loss_fact
     
       
        # wandb.log({'train/loss_fact':loss_fact, 'train/loss_rule':loss_rule})
        
        
       
        loss.backward()

        optimizer.step()

        
        log = {
            
            'positive_fact_loss': positive_fact_loss.item(),
            'negative_fact_loss': negative_fact_loss.item(),
            'positive_rule_loss': positive_rule_loss.item(),
            'negative_rule_loss': negative_rule_loss.item(),
                       
            'loss': loss.item()
        }

        return log

    @torch.no_grad()
    def evaluate(self, split, expectation=True):
        if comm.get_rank() == 0:
            logging.info('>>>>> RuleE Predictor: Evaluating on {}'.format(split))
        
        # test_set = getattr(self, "%s_set" % split)
        test_set = self.test_set_data
        dataloader = torch_data.DataLoader(test_set, 1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        # concat_mask = []
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)

            KGE_score = model.compute_g_KGE(all_h,all_r)
            # logits, mask = model(all_h, all_r, None)
            logits = KGE_score
            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            # concat_mask.append(mask)
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        # concat_mask = torch.cat(concat_mask, dim=0)
        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            val = concat_logits[k, t]

            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            # if concat_mask[k, t].item() == True:
            #     val = concat_logits[k, t]
            #     L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            #     H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            # else:
            #     L = 1
            #     H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
      
        query2LH = dict()
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        if comm.get_rank() == 0:
            logging.info('Data : {}'.format(len(query2LH)))
            logging.info('Hit1 : {:.6f}'.format(hit1))
            logging.info('Hit3 : {:.6f}'.format(hit3))
            logging.info('Hit10: {:.6f}'.format(hit10))
            logging.info('MR   : {:.6f}'.format(mr))
            logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
    