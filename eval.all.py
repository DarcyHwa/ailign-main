# This script run bertalign and ailign on the evaluation corpora and compute the scores
# the evaluation corpora are : BAF, MD.fr-ar, text+berg
# To run bertalign, download it from https://github.com/bfsujason/bertalign
# then uncomment the following ligne
#~ from bertalign import Bertalign

import sys
import numpy as np
import os
import re
import time
import ailign

from ast import literal_eval
from collections import defaultdict

# the list of evaluation corpora
corpora=['BAF','MD.fr-ar','text+berg']

ailign_params= "Ailign with : --verbose --savePlot --minDensityRatio 0.3 --cosThreshold 0.4 --margin 0.05 --distNull 1 --penalty_n_n 0.06 --runDTW --outputFormat bertalign --useMargin --detectIntervals"


#***************************************************************************
# these functions come from https://github.com/bfsujason/bertalign
def score_multiple(gold_list, test_list, value_for_div_by_0=0.0):
    # accumulate counts for all gold/test files
    pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    for goldalign, testalign in zip(gold_list, test_list):
        pcounts += _precision(goldalign=goldalign, testalign=testalign)
        # recall is precision with no insertion/deletion and swap args
        test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
        gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
        rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)

    # Compute results
    # pcounts: tpstrict,fnstrict,tplax,fnlax
    # rcounts: tpstrict,fpstrict,tplax,fplax

    if pcounts[0] + pcounts[1] == 0:
        pstrict = value_for_div_by_0
    else:
        pstrict = pcounts[0] / float(pcounts[0] + pcounts[1])

    if pcounts[2] + pcounts[3] == 0:
        plax = value_for_div_by_0
    else:
        plax = pcounts[2] / float(pcounts[2] + pcounts[3])

    if rcounts[0] + rcounts[1] == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = rcounts[0] / float(rcounts[0] + rcounts[1])

    if rcounts[2] + rcounts[3] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = rcounts[2] / float(rcounts[2] + rcounts[3])

    if (pstrict + rstrict) == 0:
        fstrict = value_for_div_by_0
    else:
        fstrict = 2 * (pstrict * rstrict) / (pstrict + rstrict)

    if (plax + rlax) == 0:
        flax = value_for_div_by_0
    else:
        flax = 2 * (plax * rlax) / (plax + rlax)

    result = dict(recall_strict=rstrict,
                  recall_lax=rlax,
                  precision_strict=pstrict,
                  precision_lax=plax,
                  f1_strict=fstrict,
                  f1_lax=flax)

    return result
    
def _precision(goldalign, testalign):
    """
    Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter

    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    # mappings from source test sentence idxs to
    #    target gold sentence idxs for which the source test sentence 
    #    was found in corresponding source gold alignment
    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)

    for (test_src, test_target) in testalign:
        if (test_src, test_target) == ((), ()):
            continue
        if (test_src, test_target) in goldalign:
            # strict match
            tpstrict += 1
            tplax += 1
        else:
            # For anything with partial gold/test overlap on the source,
            #   see if there is also partial overlap on the gold/test target
            # If so, its a lax match
            target_ids = set()
            for src_test_id in test_src:
                for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                    target_ids.add(tgt_id)
            if set(test_target).intersection(target_ids):
                fpstrict += 1
                tplax += 1
            else:
                fpstrict += 1
                fplax += 1

    return np.array([tpstrict, fpstrict, tplax, fplax], dtype=np.int32)

def log_final_scores(corpus_name,params,res,f):
    print(' ---------------------------------'+ corpus_name, file=f)
    print('| Params = '+params, file=f)
    print('|             |  Strict |    Lax  |', file=f)
    print('| Precision   |   {precision_strict:.3f} |   {precision_lax:.3f} |'.format(**res), file=f)
    print('| Recall      |   {recall_strict:.3f} |   {recall_lax:.3f} |'.format(**res), file=f)
    print('| F1          |   {f1_strict:.3f} |   {f1_lax:.3f} |'.format(**res), file=f)
    print(' ---------------------------------', file=f)
    
def read_alignments(file):
    alignments = []
    with open(file, 'rt', encoding="utf-8") as f:
        for line in f:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))
    return alignments

#*****************************************************************************

# main 
if __name__=="__main__":
        
    for corpus_name in corpora:
        eval_dir = f'eval/{corpus_name}'
        gold_dir = f'eval/{corpus_name}/gold'
        bertalign_dir = f'eval/{corpus_name}/bertalign'
        ailign_dir = f'eval/{corpus_name}/ailign'
        if not os.path.isdir(bertalign_dir):
            os.mkdir(bertalign_dir)
        if not os.path.isdir(ailign_dir):
            os.mkdir(ailign_dir)

        bertalign_alignments = []
        ailign_alignments = []
        gold_alignments = []
        time_bertalign=0
        time_ailign=0
        file_pattern=r".*" # if you want to work on a file subset change this regex



        for file_name in sorted(os.listdir(gold_dir)):
            
            if not re.search(file_pattern,file_name):
                continue
            
            print("======> processing",file_name)

            # processing bertalign alignment
            bertalign_file = os.path.join(bertalign_dir, file_name).replace(".ref",".bertalign")
            if os.path.isfile(os.path.join(bertalign_dir, file_name).replace(".ref",".bertalign")):
                bertalign_alignments.append(read_alignments(bertalign_file))
            else:
                m= re.search(r"(.*)[.](\w\w)-(\w\w)[.]ref",file_name)
                name= m.group(1)
                l1= m.group(2)
                l2= m.group(3)
                
                src_file = os.path.join(eval_dir, name+"."+l1+".txt")
                tgt_file = os.path.join(eval_dir, name+"."+l2+".txt")
                src = open(src_file, 'rt', encoding='utf-8').read()
                tgt = open(tgt_file, 'rt', encoding='utf-8').read()

                print("Start aligning {} to {}".format(src_file, tgt_file))
                t0=time.time()
                aligner = Bertalign(src, tgt, is_split=True)
                aligner.align_sents()
                aligner.save_result(bertalign_file)
                time_bertalign+=time.time()-t0
                bertalign_alignments.append(aligner.result)

            # processing ailign alignment
            ailign_file = os.path.join(ailign_dir, file_name).replace(".ref",".bertalign")
            if not os.path.isfile(os.path.join(ailign_dir, file_name).replace(".ref",".bertalign")):
                m= re.search(r"(.*)[.](\w\w)-(\w\w)[.]ref",file_name)
                name= m.group(1)
                l1= m.group(2)
                l2= m.group(3)
                
                src_file = os.path.join(eval_dir, name+"."+l1+".txt")
                tgt_file = os.path.join(eval_dir, name+"."+l2+".txt")
                
                print("Run Ailign on files {} and {}".format(src_file, tgt_file))

                t0=time.time()
                ailign.align(l1,l2,"",src_file,tgt_file,"txt",ailign_dir,["bertalign"],name+"."+l1+"-"+l2)
                time_ailign+=time.time()-t0
            
            ailign_alignments.append(read_alignments(ailign_file))
            
            gold_file = os.path.join(gold_dir, file_name)
            gold_alignments.append(read_alignments(gold_file))
         


        log=open("eval.log",mode="a",encoding="utf8")

        scores = score_multiple(gold_list=gold_alignments, test_list=bertalign_alignments)
        log_final_scores(corpus_name,"Bertalign",scores,log)
        log.write(f"Elapsed : {time_bertalign} s.\n")

        scores = score_multiple(gold_list=gold_alignments, test_list=ailign_alignments)
        log_final_scores(corpus_name,ailign_params,scores,log)
        log.write(f"Elapsed : {time_ailign} s.\n\n")


        log.close()
