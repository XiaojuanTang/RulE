
import logging, os, datetime
import argparse
import torch
from torch.utils.data import DataLoader
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RulERuleDataset, RuleETrainDataset, Iterator
from predictors import PredictorPlus
from utils import load_config, save_config, set_logger, set_seed
from trainer import TrainerPredictor, TrainerRuleE
import comm

def save_files(rules):
    with open('mined_rules.txt','w') as fw:
        for rule in rules:
            for relation in rule[0:-1]:
                fw.writelines(str(relation) + ' ')

            fw.writelines(str(rule[-1])+'\n')

def formatted_rules(_rules):
    rules = []
    
    for i, _rule in enumerate(_rules):
        rule = [i,len(_rule)]
        rule += _rule
        rules.append(rule)
    return rules

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
   
    # data path
    parser.add_argument('--data_path', default="../data/wn18rr", type=str)
    parser.add_argument('--rule_file', default="../data/wn18rr/mined_rules.txt", type=str)
   
    # device 
    parser.add_argument('--cuda', action='store_true',default=True, help='use GPU')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)

    parser.add_argument('--seed',default=800, type=int, help='seed')
    
    # pre train process (KGE + rulE)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=256 , type=int)
    parser.add_argument('--rule_batch_size',default=128,type=int, help='rule batch size')
    parser.add_argument('--rule_negative_size',default=64,type=int)

    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g_f', '--gamma_fact', default=6, type=float)
    parser.add_argument('-g_r', '--gamma_rule', default=5, type=float)
    
    parser.add_argument('-adv', '--negative_adversarial_sampling', default=True, action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=0.5, type=float)
                            
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--g_warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--weight_rule',type=float,default=1)

    parser.add_argument('--max_steps', default=15000, type=int)

    # save path
    parser.add_argument('-init', '--init_checkpoint_config', default="../config/fb15k237_config.json", type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)

    # RNN parameters
    parser.add_argument('--rnn_hidden_dim', default=512, type=int)
    parser.add_argument('--num_layers', default=2, type=int)

    # grounding training process
    parser.add_argument('--rule_dim', default=100, type=int)
    parser.add_argument('--mlp_rule_dim', default=100, type=int)
    parser.add_argument('--alpha', default=5.0, type=int, help='weight the KGE score')
    parser.add_argument('--smoothing', default=0.5, type=float)
    parser.add_argument('--batch_per_epoch', default=1000000, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--g_batch_size', default=16, type=int)
    parser.add_argument('--g_lr', default=0.00005, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_iters', default=20, type=int)
  
   
    return parser.parse_args(args)

def main(args):
    
    # read the given config
    if args.init_checkpoint_config:
        args = load_config(args.init_checkpoint_config)
        args = args[0]
       

    # wandb.init(project='RulE',group='RotatE', name = args.save_path, config=args)
   
    if args.save_path is None:
        args.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))
    else:
        args.save_path = '../outputs/'+ args.save_path
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    save_config(args)

    set_logger(args.save_path)
    set_seed(args.seed)

    graph = KnowledgeGraph(args.data_path)
    train_set = TrainDataset(graph, args.g_batch_size)
    valid_set = ValidDataset(graph, args.g_batch_size)
    test_set = TestDataset(graph, args.g_batch_size)

    # dataset = RuleDataset(graph.relation_size,args.rule_file)
    dataset = RulERuleDataset(graph.relation_size, args.rule_file,args.rule_negative_size)
    # # Set training dataloader iterator

   
    train_dataloader_tail = DataLoader(
        RuleETrainDataset(graph.train_facts, graph.entity_size, graph.relation_size, args.negative_sample_size, 'tail-batch'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=RuleETrainDataset.collate_fn
    )
    
    train_iterator = Iterator(train_dataloader_tail)

    rules = [rule[0] for rule in dataset.rules]

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Train Final Predictor+')
        logging.info('-------------------------')

    predictor = PredictorPlus(graph, args.mlp_rule_dim, args.gamma_fact, args.gamma_rule, args.hidden_dim, args.rule_dim, args.rnn_hidden_dim, args.num_layers, device)
    predictor.set_rules(rules)
    
    if args.cuda:
        predictor = predictor.cuda()

  
    solver_ruleE = TrainerRuleE(graph, predictor, train_set ,graph.train_facts, graph.valid_facts, graph.test_facts, test_set, train_iterator, dataset, True, device, args.cpu_num)
   
    solver_ruleE.train(args.max_steps, args)


    print("loading RulE trainer......")

    # load rule embedding and KGE embedding

    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    predictor.load_state_dict(checkpoint['model'])

    test_mrr = solver_ruleE.evaluate('test', expectation=True)

    predictor.entity_embedding.weight.requires_grad = False
    predictor.relation_embedding.weight.requires_grad = False
    predictor.rule_emb.weight.requires_grad = False

    for param in predictor.rnn.parameters():
        param.requires_grad = False
    for param in predictor.linear.parameters():
        param.requires_grad = False


    optim = torch.optim.Adam([ param for param in predictor.parameters() if param.requires_grad == True], lr=float(args.g_lr), weight_decay=float(args.weight_decay))
    
    solver_p = TrainerPredictor(predictor,  train_set, valid_set, test_set, optim, device, args.cpu_num)
   
    predictor.eval_compute_rule_weight()
    # output_weight(predictor.rule_features, predictor.rules_weight_emb)

    test_mrr_iter = solver_p.evaluate('test', args.alpha, expectation=True)
    # test_mrr = solver_ruleE.evaluate('test', expectation=cfg.predictor.eval.expectation)

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

        solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, device, args.cpu_num)

        solver_p.train(args.batch_per_epoch, args.smoothing, args.print_every)
        valid_mrr_iter = solver_p.evaluate('valid', args.alpha, expectation=True)
        test_mrr_iter = solver_p.evaluate('test', args.alpha, expectation=True)
        # test_mrr_iter = solver_p.evaluate_t('test', args.alpha, expectation=True)
        

        if valid_mrr_iter > best_valid_mrr:
            best_valid_mrr = valid_mrr_iter
            test_mrr = test_mrr_iter
            solver_p.save(args, os.path.join(args.save_path, 'predictor.pt'))
    
    if comm.get_rank() == 0:
        logging.info('-------------------------')
        logging.info('| Final Test MRR: {:.6f}'.format(test_mrr))
        logging.info('-------------------------')

if __name__ == '__main__':
    main(parse_args())
    