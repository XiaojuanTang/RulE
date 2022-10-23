
import imp
import comm
from utils import *
import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import DataLoader
from itertools import islice
from data import Iterator, RuleDataset, KGETrainDataset
import torch.nn.functional as F
import wandb
from tqdm import tqdm

class GroundTrainer(object):
    
    def __init__(self, model, train_set, valid_set, test_set, scheduler=None, gpus=None, num_worker=0):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                logging.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.scheduler = scheduler
        
    def train(self, args):
        
        # fix the parameters of pre-training

        self.model.entity_embedding.weight.requires_grad = False
        self.model.relation_embedding.weight.requires_grad = False
        self.model.rule_emb.weight.requires_grad = False

        for param in self.model.rnn.parameters():
            param.requires_grad = False
        for param in self.model.linear.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=float(args.g_lr), 
            weight_decay=float(args.weight_decay))


        self.train_set.make_batches()
        
        train_dataloader = DataLoader(self.train_set, 1, num_workers=self.num_worker)
        
        self.model.eval_compute_rule_weight(self.device)


        if comm.get_rank() == 0:
            logging.info('>>>>> RulE: Grounding-Training')
        

        best_valid_mrr = 0.0 
        test_mrr = 0.0

        # warm_up_steps = args.num_iters // 2
        # current_learning_rate = float(args.g_lr)

        for k in range(args.num_iters):

            if comm.get_rank() == 0:
                logging.info('-------------------------')
                logging.info('| Iteration: {}/{}'.format(k + 1, args.num_iters))
                logging.info('-------------------------')
            
            # if k >= warm_up_steps:

            #     current_learning_rate = current_learning_rate / 10
            #     logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, k))
            #     optim = torch.optim.Adam(
            #         filter(lambda p: p.requires_grad, predictor.parameters()), 
            #         lr=current_learning_rate
            #     )
            #     warm_up_steps = warm_up_steps * 3

            self.train_step( optimizer, train_dataloader, args.batch_per_epoch, args.smoothing, args.print_every)
            valid_mrr_iter = self.evaluate('valid', args.alpha, expectation=True)
            test_mrr_iter = self.evaluate('test', args.alpha, expectation=True)
            # test_mrr_iter = solver_p.evaluate_t('test', args.alpha, expectation=True)
            

            if valid_mrr_iter > best_valid_mrr:
                best_valid_mrr = valid_mrr_iter
                test_mrr = test_mrr_iter
                self.save(args, os.path.join(args.save_path, 'grounding.pt'))
        
        if comm.get_rank() == 0:
            logging.info('-------------------------')
            logging.info('| Final Test MRR: {:.6f}'.format(test_mrr))
            logging.info('-------------------------')


    def train_step(self, optimizer, train_dataloader, batch_per_epoch, smoothing, print_every):
        
        batch_per_epoch = batch_per_epoch or len(train_dataloader)
        model = self.model
        
        model.train()

        total_loss = 0.0
        total_size = 0.0

        for batch_id, batch in enumerate(islice(train_dataloader, batch_per_epoch)):
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
            
            grounding_rule_score, mask = model(all_h, all_r, edges_to_remove)

            if mask.sum().item() != 0:
                rule_logits = (torch.softmax(grounding_rule_score, dim=1) + 1e-8).log()
                
                loss = -(rule_logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
            
            if (batch_id + 1) % print_every == 0:
                
                if comm.get_rank() == 0:
                    logging.info('loss:    {} {} {:.6f} {:.1f}'.format(batch_id + 1, len(train_dataloader), loss, total_size / print_every))
                
                total_loss = 0.0
                total_size = 0.0

    @torch.no_grad()
    def evaluate(self, split, alpha=3.0, expectation=True):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        
        dataloader = DataLoader(test_set, 1, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        
        for batch in tqdm(dataloader):

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
            kge_score = model.compute_g_KGE(all_h,all_r)
            logits += alpha * kge_score

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
        }
       
        torch.save(state, checkpoint)

        g_rule_embedding = self.model.mlp_feature.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'g_rule_embedding'), 
            g_rule_embedding
        )


class PreTrainer(object):

    def __init__(self, graph, model, valid_set, test_set, tripletset, ruleset, expectation, scheduler=None, gpus=None, num_worker=0):
        
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                logging.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.graph = graph
        self.model = model
        self.valid_set = valid_set
        self.test_set = test_set
        self.TripletSet = tripletset
        self.RuleSet = ruleset
        self.expectation = expectation


    
    def train(self, args):
        
        # Set training configuration
        
        triplets_dataloader = DataLoader(
            self.TripletSet,
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=KGETrainDataset.collate_fn)
        
        rules_dataloader = DataLoader(
            self.RuleSet, 
            batch_size=args.rule_batch_size, 
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=RuleDataset.collate_fn)

        self.triplets_iterator = Iterator(triplets_dataloader)
        self.rules_iterator = Iterator(rules_dataloader)
    
        current_learning_rate = float(args.learning_rate)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=current_learning_rate
        )
        if comm.get_rank == 0:
            logging.info('>>>>> ruleE: Pre-training')
        training_logs = []
        best_mrr = 0.0

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        for step in range(0, args.max_steps + 1):

            log = self.train_step( optimizer, self.triplets_iterator, self.rules_iterator, args)
            
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
                mrr = self.evaluate("valid", self.expectation )
                if mrr > best_mrr:
                    save_model(self.model,optimizer, args)
                    best_mrr = mrr

    def train_step(self, optimizer, triplets_iterator, rules_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model = self.model
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode= next(triplets_iterator)
        positive_rule, negative_idx, negative_rule, mode_rule, rule_mask = next(rules_iterator)

        if self.device.type == "cuda":
            positive_sample = positive_sample.cuda(self.device)
            negative_sample = negative_sample.cuda(self.device)
            positive_rule = positive_rule.cuda(self.device)
            negative_idx = negative_idx.cuda(self.device)
            negative_rule = negative_rule.cuda(self.device)
            rule_mask = rule_mask.cuda(self.device)
            subsampling_weight = subsampling_weight.cuda(self.device)

        negative_fact_score = model.compute_KGE((positive_sample, negative_sample), mode) 
        negative_rule_score = model.compute_ruleE((positive_rule,  rule_mask, negative_idx, negative_rule), mode=mode_rule) 

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
        positive_rule_score = model.compute_ruleE((positive_rule,rule_mask))


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
            logging.info('>>>>> RuleE emb: Evaluating on {}'.format(split))
        
        test_set = getattr(self, "%s_set" % split)
        # test_set = self.test_set_data
        dataloader = DataLoader(test_set, batch_size=1, num_workers=self.num_worker)
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
            
            logits = KGE_score
            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            
        
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)

        
        ranks = []
        for k in range(concat_all_t.size(0)):
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            val = concat_logits[k, t]

            L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
            H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
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