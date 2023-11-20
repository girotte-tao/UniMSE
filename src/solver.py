import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from utils import contains_fine
from transformers import T5Tokenizer
from model import Model
from config import DEVICE, get_args, get_config

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.info_nce = hp.info_nce   #False
        self.mosi_test_loader, self.mosei_test_loader = None, None
        if len(test_loader) == 2:
            self.mosi_test_loader, self.mosei_test_loader = test_loader
        elif len(test_loader) == 3:
            self.mosi_test_loader, self.mosei_test_loader, self.meld_test_loader = test_loader
        elif len(test_loader) == 4:
            self.mosi_test_loader, self.mosei_test_loader, self.meld_test_loader, self.iemocap_test_loader = test_loader
        else:
            self.test_loader = test_loader

        self.is_train = is_train
        self.model = model
        self.use_adapter = hp.use_adapter  #True

        # Training hyperarams
        model_path = './t5-base'
        # model_path = '../t5-large'
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = Model(hp)
            
        if torch.cuda.is_available():

            model = self.model.to(DEVICE)
           
        else:
            self.device = torch.device("cpu")


        # optimizer
        self.optimizer={}

        if self.is_train:
            adapter_param = []
            main_param = []
            info_param = []
            T5_param = []

            if hp.fine_T5:       #fine-tune T5
                fine_T5_layers = hp.fine_T5_layers  # ['block.10', 'block.11']
                for name, p in model.named_parameters():
                    # print(name)
                    if p.requires_grad:
                        if 'adapter' in name:
                            adapter_param.append(p)
                        elif 'info_loss' in name:
                            info_param.append(p)
                        elif 'T5' in name:
                            if contains_fine(name, fine_T5_layers):
                                # print(name)
                                p.requires_grad = True
                                T5_param.append((name, p))
                            else:
                                p.requires_grad = False
                        else:
                            p.requires_grad = True
                            main_param.append(p)

                no_decay = ['bias', 'LayerNorm.weight']
                if self.use_adapter:
                    print('--------------use adapter------------------------')
                    print('--------------finetune T5------------------------')
                    if self.info_nce:
                        print('--------------use info_nce------------------------')
                        self.optimizer_main_group = [
                            {'params': [p for n, p in T5_param if not any(nd in n for nd in no_decay)],
                             'weight_decay': hp.weight_decay_T5, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': [p for n, p in T5_param if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': info_param, 'weight_decay': hp.weight_decay_info, 'lr': hp.lr_info},
                            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
                            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                        ]
                    
                    else:

                        self.optimizer_main_group = [
                            {'params': [p for n, p in T5_param if not any(nd in n for nd in no_decay)],
                             'weight_decay': hp.weight_decay_T5, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': [p for n, p in T5_param if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
                            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                        ]
                else:
                    print('--------------------without adapter-------------------')
                    print('--------------finetune T5------------------------')

                    self.optimizer_main_group = [
                        {'params': [p for n, p in T5_param if not any(nd in n for nd in no_decay)],
                         'weight_decay': hp.weight_decay_T5, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                        {'params': [p for n, p in T5_param if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                    ]
            else:
                for name, p in model.named_parameters():
                    # print(name)
                    if p.requires_grad:
                        if 'adapter' in name:
                            adapter_param.append(p)
                        elif 'T5' in name:
                            p.requires_grad = False
                            T5_param.append(p)
                        # elif 'adapter' in name:
                        #     adapter_param.append(p)
                        else:
                            p.requires_grad = True
                            main_param.append(p)

                if self.use_adapter:
                    print('--------------use adapter------------------------')
                    self.optimizer_main_group = [
                        {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
                        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                    ]
                else:
                    print('--------------------without adapter-------------------')
                    self.optimizer_main_group = [
                        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                    ]



            # for name, p in model.named_parameters():
            #     if p.requires_grad:
            #        print(name)

        # if self.use_adapter:
        #     print('--------------use adapter------------------------')
        #     self.optimizer_main_group = [
        #         {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
        #         {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        #     ]
        # else:
        #     print('--------------------without adapter-------------------')
        #     self.optimizer_main_group = [
        #         {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        #     ]
        # self.optimizer_adapter = getattr(torch.optim, self.hp.optim)(
        #     adapter_param, lr=self.hp.lr_adapter, weight_decay=self.hp.weight_decay_adapter)

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
           self.optimizer_main_group
        )
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main

        scheduler_main = self.scheduler_main


        def train(model, optimizer):
            epoch_loss = 0.0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size

            for i_batch, batch_data in enumerate(tqdm(self.train_loader)):
                sentences, ids = None, None
                t5_input_id, t5_att_mask, t5_labels, acoustic, alens, visual, vlens, y= \
                    batch_data['t5_input_id'], batch_data['t5_att_mask'], batch_data['t5_labels'], batch_data['audio'], batch_data['audio_lengths'], batch_data['behavior'], batch_data['behavior_lengths'], batch_data['labels']

                model.zero_grad()

                with torch.cuda.device(0):
                    if visual != None and acoustic != None:
                        visual, acoustic, t5_input_id, t5_att_mask, t5_labels = \
                        visual.to(DEVICE), acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                        t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                    elif acoustic != None:
                        acoustic, t5_input_id, t5_att_mask, t5_labels = \
                            acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                            t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                    elif visual != None:
                        visual, t5_input_id, t5_att_mask, t5_labels = \
                        visual.to(DEVICE), t5_input_id.to(DEVICE), \
                        t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                    else:
                        t5_input_id, t5_att_mask, t5_labels = \
                            t5_input_id.to(DEVICE), \
                            t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)

                logits, total_loss = self.model(sentences, t5_input_id, t5_att_mask,
                                          t5_labels, ids, visual, acoustic, vlens, alens)   #sentence 和 ids 没啥用

                # for mosei we only use 50% dataset in stage 1
                # print('batch: {}, train loss:{}'.format(i_batch, loss))
                # print('total_loss:{}'.format(total_loss))
                loss, tv_loss, ta_loss = total_loss
                ### 如果不采用对比学习，tv_loss=0, ta_loss=0
                # print('Training: main_loss:{}, tv_loss:{}, ta_loss:{}'.format(loss, tv_loss, ta_loss))
                loss = loss + 0.5 * tv_loss + 0.5 * ta_loss
                # epoch_loss += loss
                epoch_loss += loss

                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                loss = loss.requires_grad_(True)
                loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)                    
                self.optimizer_main.step()
                ### 设置下Training step
#                 loss = loss / self.hp.gradient_accumulation_step
#                 if i_batch % self.hp.gradient_accumulation_step == 0:
#                     loss = loss.requires_grad_(True)
#                     loss.backward()
#                     # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
#                     self.optimizer_main.step()
                    

            return epoch_loss / self.hp.n_train
        
        def pre_gen(results):
            #保障生成格式
            new_results = []
            for ele in results:
                if len(str(ele).split(','))==1:
                    if is_number(str(ele)):
                        new_results.append(str(ele)+','+'neutral')
                    else:
                        new_results.append('0.0'+','+str(ele))
                else:
                    new_results.append(ele)
                    
            return new_results

        def evaluate(model, loader, n_loader=None, test=False):
            model.eval()
            # loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0

            results = []
            truths = []
            ids_list = []
            tmp = []

            with torch.no_grad():
                for i, batch_data in enumerate(tqdm(loader)):
                    sentences, ids = None, None
                    t5_input_id, t5_att_mask, t5_labels, acoustic, alens, visual, vlens, y= \
                        batch_data['t5_input_id'], batch_data['t5_att_mask'], batch_data['t5_labels'], batch_data['audio'], batch_data['audio_lengths'], batch_data['behavior'], batch_data['behavior_lengths'], batch_data['labels']


                    with torch.cuda.device(0):
                        if visual != None and acoustic != None:
                            visual, acoustic, t5_input_id, t5_att_mask, t5_labels = \
                                visual.to(DEVICE), acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                        elif acoustic != None:
                            acoustic, t5_input_id, t5_att_mask, t5_labels = \
                                acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                        elif visual != None:
                            visual, t5_input_id, t5_att_mask, t5_labels = \
                                visual.to(DEVICE), t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                        else:
                            t5_input_id, t5_att_mask, t5_labels = \
                                t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)


                    # we don't need lld and bound anymore
                    logits, loss = self.model(sentences, t5_input_id, t5_att_mask,
                                          t5_labels, ids, visual, acoustic, vlens, alens)
                    output_ids = self.model.generate(t5_input_id, t5_att_mask, visual, acoustic, vlens, alens)
                    # print(output_ids)
                    main_loss, tv_loss, ta_loss  = loss
                # print('Training: tv_loss:{}, ta_loss:{}, main_loss:{}'.format(tv_loss, ta_loss, main_loss))
                    loss = main_loss + 0.5 * tv_loss + 0.5 * ta_loss
                    pred_token = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#                     if test:
#                         print('---------------------------------------------------------------------')
#                         # print('pred_token:{}'.format(pred_token))
#                         por_pred_token = [ele.split(',')[0] for ele in pred_token]
#                         score_pred_token = [ele.split(',')[1] for ele in pred_token]
#                         meld_pred_token = [ele.split(',')[2] for ele in pred_token]
#                         iemocap_pred_token = [ele.split(',')[3] for ele in pred_token]
#                         print('score pred token:{}'.format(score_pred_token))
#                         print('meld pred token:{}'.format(meld_pred_token))
#                         print('iemocap pred token:{}'.format(iemocap_pred_token))

#                         print('truth token:{}'.format(y))
                    # # print('truth token:{}'.format(list(y.numpy())))
                    for token in pred_token:
                        if is_number(token):
                            tmp.append(float(token))
                        else:
                            tmp.append(token)

                        # print('batch:{}, pred_tokens:{}'.format(i, tmp))
                        # print('batch:{}, pred_tokens:{}'.format(i, list(y)))

                        # Collect the results into ntest if test else self.hp.n_valid)
                    if len(tmp) != len(list(y)):
                        print('error')
                    total_loss += loss
                    results.extend(tmp)
                    truths.extend(y)
                    # ids_list.extend(ids)

                    tmp = []

            if self.hp.n_valid == 0:
                self.hp.n_valid = 1
            if n_loader is None:
                avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
            else:
                avg_loss = total_loss / n_loader

            # results = torch.cat(results)
            if self.hp.pred_type == 'regression':
                truths = torch.cat(truths)
            WF1 = f1_score(truths, results, average='weighted')
            return WF1, avg_loss, results, truths, ids_list

        best_WF1 = 0.0
        test_loss = 0.0
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
           
            self.epoch = epoch

            # minimize all losses left
            train_loss = train(model, optimizer_main)
            WF1,val_loss, _, _, _ = evaluate(model, self.dev_loader, test=False)
            
            
            end = time.time()
            duration = end - start
            
            if WF1 > best_WF1:
                best_WF1 = WF1
                print(WF1)
                _, test_loss, results, truths, ids_list = evaluate(model, self.test_loader, test=True)
                report = classification_report(truths, results, digits=4)
                print(report)
            print_info = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
                epoch, duration, train_loss, val_loss, test_loss)
            print("-"*50)
            # print('Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, train_loss, val_loss, test_loss))
            print(print_info)
            print("-"*50)
        sys.stdout.flush()