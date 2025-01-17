# tstMat.pkl กับ valMat.pkl ข้อมูลเหมือนกัน

import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph, GTLayer
from DataHandler import DataHandler
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import pandas as pd

open('log.txt', 'w').close() # ล้างข้อมูลมุกอย่างใน log.txt ก่อน

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')

class Coach:
    def __init__(self, handler):
        self.handler = handler

        #print('USER', args.user, 'ITEM', args.item)
        logging.info(f'USER: {args.user}, ITEM: {args.item}')

        #print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        logging.info(f'NUM OF INTERACTIONS: {self.handler.trnLoader.dataset.__len__()}')

        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        self.train_losses = []
        self.test_recalls = []
        self.test_ndcgs = []

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        os.makedirs(f'./Models/{type(self.opt).__name__}_{args.lr}', exist_ok=True)
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        bestRes = None
        result = []
        recall_ndcg_epochs = []
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            
            if tstFlag:
                recall_ndcg_epochs.append(ep) 
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory(ep)
                result.append(reses)
                bestRes = reses if bestRes is None or reses['Recall'] > bestRes['Recall'] else bestRes
            print()
            
        reses = self.testEpoch()
        recall_ndcg_epochs.append(args.epoch)
        result.append(reses)
        torch.save(result, "Saeg_result.pkl")
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('Best Result', args.epoch, bestRes, True))
        self.saveHistory(ep)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, marker='o')
        plt.title(f'Training Loss Curve\nLearning Rate: {args.lr}, Optimizer: {type(self.opt).__name__}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'loss_{type(self.opt).__name__}_{args.lr}.png')


        print(f"recall ndcg epoch: {recall_ndcg_epochs}")
        plt.figure(figsize=(10, 6))
        plt.plot(recall_ndcg_epochs, self.test_recalls, marker='o', label='Recall')
        plt.plot(recall_ndcg_epochs, self.test_ndcgs, marker='s', label='NDCG')
        plt.title(f'Recall and NDCG Curve\nLearning Rate: {args.lr}, Optimizer: {type(self.opt).__name__}')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'recall_ndcg_{type(self.opt).__name__}_{args.lr}.png')

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.gtLayer).cuda()
        #self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.opt = t.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        self.handler.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(),
                                                 self.handler)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            usrEmbeds, itmEmbeds, cList, subLst = self.model(self.handler, False, sub, cmp,  encoderAdj,
                                                                           decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            usrEmbeds2 = subLst[:args.user]
            itmEmbeds2 = subLst[args.user:]
            ancEmbeds2 = usrEmbeds2[ancs]
            posEmbeds2 = itmEmbeds2[poss]

            bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            
            scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
            bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

            regLoss = calcRegLoss(self.model) * args.reg

            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(
                ancs,
                usrEmbeds,
                itmEmbeds) + args.ctra*contrastNCE(ancs, subLst, cList)
            loss = bprLoss + regLoss + contrastLoss + args.b2*bprLoss2

            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
                i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
        
        avgLoss = epLoss / steps
        self.train_losses.append(avgLoss) 
        
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        test_results = []

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds, _, _ = self.model(self.handler, True, self.handler.torchBiAdj, self.handler.torchBiAdj,
                                                          self.handler.torchBiAdj)

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            
            # Collect test data
            for u, top in zip(usr.cpu().numpy(), topLocs.cpu().numpy()):
                neg_items = list(set(top))  # get the top items for this user

                test_results.append({
                    'user_id': u,
                    'item_id': top[0],  # You can choose any other logic for item_id
                    'neg_items': neg_items
                })
            
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        
        avgRecall = epRecall / num
        avgNdcg = epNdcg / num
        self.test_recalls.append(avgRecall)
        self.test_ndcgs.append(avgNdcg)
        
        predict_path = "log_movie_lens/predict.csv"
        # Save to CSV
        df = pd.DataFrame(test_results)
        df.to_csv(predict_path, index=False, sep='\t')
        
        ret = dict()
        ret['Recall'] = avgRecall
        ret['NDCG'] = avgNdcg 
                
        return ret

    # code คำนวณ recall แล้วได้มากกว่า 1, ไม่ต้องไปสนใจ
    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            #temTopLocs = list(topLocs[i])
            #temTstLocs = tstLocs[batIds[i]]
            temTopLocs = list(set(topLocs[i]))  # Remove duplicates
            temTstLocs = list(set(tstLocs[batIds[i]]))  # Ensure no duplicates

            tstNum = len(temTstLocs)

            if tstNum ==0:
                continue

            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            count = 0
            corr_count = 0
            for val in temTstLocs:
                count +=1
                if val in temTopLocs:
                    corr_count+=1
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            #logging.info(f"count: {count}, corr_count: {corr_count}, tstNum: {tstNum}, maxDcg: {maxDcg}")
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg
        #return allRecall / len(batIds), allNdcg / len(batIds)

    def saveHistory(self, ep):
        self.epoch = ep
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, f'./Models/{type(self.opt).__name__}_{args.lr}/' + args.save_path + f'_ep{ep}.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load(f'./Models/{type(self.opt).__name__}_{args.lr}/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    logger.saveDefault = True

    log('Start')
    if t.cuda.is_available():
        logging.info("using cuda")
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
