import traceback
import numpy as np
import json
from tqdm import tqdm
import os
import os.path as osp

def segment_labels(Yi):
	idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
	Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
	return Yi_split

def segment_intervals(Yi):
	idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
	intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
	return intervals

def overlap_f1(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)

def evaluate(preds,epoch,complete=True):
    main_path = osp.join('/','media','SSD0','nayobi','All_datasets','SAR-RARP50','videos','fold1')
    videos = os.listdir(main_path)
    module = 1 
    # module = 10
    video_dict = {}
    # preds = json.load(open('{}/{}_preds_gestures.json'.format(experiment,'best_predictions/best' if args.epoch <1 else 'epoch_{}'.format(args.epoch))))

    for video in tqdm(videos,desc='videos'):
        with open(osp.join(main_path,video,'action_discrete.txt'),'r') as f:
            lines = f.readlines()
        
        labels = []
        cat_preds = []
        scores = []
        topks = []

        for lid,line in enumerate(lines):
            if lid % module == 0:
                frame,label = line.split(',')
                pred_label = np.argmax(preds['{}/{}.png'.format(video,frame)])
                labels.append(int(label))
                cat_preds.append(pred_label)
                scores.append(preds['{}/{}.png'.format(video,frame)][pred_label])
        
        labels = np.array(labels)
        cat_preds = np.array(cat_preds)
        present = set(np.unique(labels)) | set(np.unique(cat_preds))

        assert len(labels)==len(cat_preds), 'Diferente longitud {} & {}'.format(len(labels),len(cat_preds))
        assert np.min(labels)>=0 and np.min(cat_preds) >= 0, 'Fuera de rango inferior {} & {}'.format(np.min(labels),np.min(cat_preds))
        assert np.max(labels)<8 and np.max(cat_preds) < 8, 'Fuera de rango superior {} & {}'.format(np.max(labels),np.max(cat_preds))
        
        # scores = np.array(scores)
        # for _ in range(100):
        #     new_scores = copy(scores)
        #     new_preds = copy(cat_preds)
        #     for idx in range(1,len(labels)-1):
        #         ant = max(0,idx-1)
        #         nex = min(len(cat_preds)-1,idx+1)
        #         if (cat_preds[ant] != cat_preds[idx]) or (cat_preds[idx] != cat_preds[nex]):
        #             if cat_preds[ant] == cat_preds[nex] and np.mean([scores[ant],scores[nex]])>scores[idx]:
        #                 new_preds[idx] = cat_preds[ant] if ant>0 else cat_preds[nex]
        #                 new_scores[idx] = scores[ant] if ant>0 else scores[nex]
        #             # elif cat_preds[ant] != cat_preds[nex] and (scores[ant]>scores[idx] or scores[nex]>scores[idx]):
        #             #     if scores[ant] > scores[nex]:
        #             #         new_preds[idx] = cat_preds[ant]
        #             #         arr = [scores[ant],scores[idx]]
        #             #         new_scores[idx] = float(np.mean(arr))
        #             #     elif scores[ant] < scores[nex]:
        #             #         new_preds[idx] = cat_preds[nex]
        #             #         arr = [scores[nex],scores[idx]]
        #             #         new_scores[idx] = float(np.mean(arr))
        #     scores = new_scores
        #     cat_preds = new_preds

        if complete:
            with open(osp.join(main_path,video,'action_continues.txt'),'r') as f:
                cont_lines = f.readlines()

            vid_len = int(cont_lines[-1].split()[1])+1
            long_labels = np.zeros(vid_len,dtype='uint8')
            long_preds = np.zeros(vid_len,dtype='uint8')

            for lid,line in enumerate(cont_lines):
                inf,sup,label_id = map(int,line.split())
                long_labels[inf:sup+1] = label_id

            for lid,pred in enumerate(cat_preds):
                if lid>0:
                    sup_idx = lid*module*6
                    inf_idx = (lid-1)*module*6
                    sup_score = scores[lid]
                    sup_pred = cat_preds[lid]
                    inf_score = scores[lid-1]
                    inf_pred = cat_preds[lid-1]

                    long_preds[sup_idx] = sup_pred

                    if inf_pred==sup_pred:
                        long_preds[inf_idx:sup_idx+1] = inf_pred
                    else:# args.inter:
                        inf_sup_idx = int(round((6*module*inf_score)/(inf_score+sup_score)))
                        sup_inf_idx = int(round((6*module*sup_score)/(inf_score+sup_score)))

                        try:
                            long_preds[inf_idx+1 : inf_idx+inf_sup_idx+1] = inf_pred
                            long_preds[sup_idx-sup_inf_idx+1 : sup_idx] = sup_pred
                        except:
                            traceback.print_exc()
                            breakpoint()

                else:
                    long_preds[lid] = cat_preds[lid]


        TPs = np.sum(labels==cat_preds)
        All = len(labels)
        Acc = TPs/All
        try:
            F1 = overlap_f1(cat_preds,labels,8,None)
            if complete:
                c_F1 = overlap_f1(long_preds,long_labels,8,None)
        except:
            traceback.print_exc()
            breakpoint()
        assert video not in video_dict, 'Ya estaba {}'.format(video)

        video_dict[video]={'Accuracy': Acc, 'F1': F1}
        if complete:
            video_dict[video]['cF1'] = c_F1
            
    mAcc = np.mean([video_dict[video]['Accuracy'] for video in video_dict])
    mF1 = np.mean([video_dict[video]['F1'] for video in video_dict])
    if complete:
        mcF1 = np.mean([video_dict[video]['cF1'] for video in video_dict])
    else:
        mcF1 = 0

    print('- mAcc: {}, mF1: {}, mcF1: {}, & mSR50: {}\n'.format(mAcc,mF1,mcF1,float(np.sqrt(mAcc*mF1))))
    video_dict['mAcc'] = mAcc
    video_dict['mF1'] = mF1
    video_dict['mSR50'] = float(np.sqrt(mAcc*mF1))

    json.dump(video_dict,open('output/GESTURES/post_processing/gesture_scores_epoch{}.json'.format(epoch),'w'))
    return float(np.sqrt(mAcc*mF1))