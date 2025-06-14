# -*- coding:utf8 -*-
"""
This module defines various functions for sentence alignment using Dynamic Time Warping (DTW)
a kind of Viterbi algorithm. The distance of a given path is the sum of the cos distance that corresponds
to the embeddings of each group.
Performs lexical or chunk alignment if necessary.

Main function : 
    
    align(params,preprocessor,encoder)
    
"""

import os
import sys
import re
import math
import time
import shelve
import math
from BTrees.OOBTree import OOBTree
import numpy as np

import matplotlib.pyplot as plt

# local modules
from anchor_points import extract_anchor_points, compute_points_from_ngrams, compute_points_with_encoder
from read_write_files import read_input_file, write_alignable, write_aligned_points, add_anchor_in_output, read_alignment_file
from lexical_alignment import chunk_alignment, word_alignment

# Various low level parameters
coeff_sent_len = 0.33  # balance between sentence based length
coeff_neighbour_sim = 0.6  # strength of the margin penalty
max_group_size = 4 # the max size of a group (for allowed groups generation)
only_one_2_one_pairing = False # if True, only [(0, 1), (1, 0), (1, 1)] are considered

# Initializing global variable
infinite = float('inf')

################################################################## definitions    
def load_sentence_encoder(params):
    """
    Conditionnaly import alternative models (main model is labse)
    arg1 : global params
    
    returns (preprocessor, encoder)
    """
    preprocessor = False
    encoder = False

    # open various pretrained models (https://www.sbert.net/docs/pretrained_models.html) including labse
    # n.b.: some model are more adapted to translation comparison, other to paraphrasing
    if params['embedModel'] == "laser":
        # import modules for laser
        from laserembeddings import Laser

        encoder = Laser()
    elif params['embedModel'] == "sbert":
        # import modules for sbert
        from sentence_transformers import SentenceTransformer

        print("*** Loading sbert model", params['modelName'])
        encoder = SentenceTransformer(params['modelName'])
        if not params['useGPU']:
            encoder.to("cpu")
            print("Using device: CPU")
        
        
    elif params['embedModel'] == "labse-keras":
        import tensorflow_hub as hub
        import tensorflow as tf
        import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess

        print("*** LABSE : Loading preprocessor")
        preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
        print("*** LABSE : Loading model")
        encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
    
    return (preprocessor,encoder)        

########################################################################## align function
def align(params,preprocessor,encoder):
    """
    Sentence alignment function
    
    param1 (dict) : the global parameters. Main parameters are :
        params['l1'] (str) : the iso code for language 1 (eg. "en","fr","it",...)
        params['l2'] (str) : the iso code for language 2 (eg. "en","fr","it",...)
        params['inputDir'] (str): the directory that contains input files
        params['inputFile1'] (str): the filename 1
        params['inputFile2'] (str): the filename 2
        params['inputFormat'] (str) : the directory where to write output files
        params['outputFormats'] (list[str]) : the list of output formats (eg. ['tmx','txt','tsv','ces']
        params['outputFileName'] (str, optionnal) : the prefix of the output filename (without extension ".tmx", ".txt", etc.)
        params['col1'] (int, optionnal) : for tsv input format, the number of column that contains language 1 text
        params['col2'] (int, optionnal) : for tsv input format, the number of column that contains language 2 text
        params['addAnchor'] (bool) : indicates whether anchors should be inserted in xml format
        params['fileId1'] (str): the file id prefix in the xml anchors
        params['fileId2'] (str): the file id prefix in the xml anchors
    param2 (preprocessor): the preprocesser of embedding encoder
    param3 (encoder): the embedding encoder
    
    """
    
    l1=params['l1']
    l2=params['l2']
    input_dir=params['inputDir']
    file1=params['inputFile1']
    file2=params['inputFile2']
    input_format=params['inputFormat']
    output_dir=params['outputDir']
    output_formats=params['outputFormats']
    output_file_name=params['outputFileName']
    col1=params['col1']
    col2=params['col2']
    print_ids=params['printIds']
    file_id1=params['fileId1']
    file_id2=params['fileId2']
    add_anchor=params['addAnchor']
    print_log=params['printLog']
    # the log handle is saved in global params
    if print_log:
        log=params['logHandle']
    

    embed_shelve={}
    if params['useShelve']:
        embed_shelve = shelve.open("embeds")

    # processing of an aligned file pair
    if params['verbose']:
        print("Processing", file1, "and", file2)
        
    # Generating allowed groups and saving it into params
    allowed_groups = [(0, 1), (1, 0), (1, 1)]
    if not only_one_2_one_pairing:
        for i in range(2, max_group_size + 1):
            allowed_groups.append((1, i))
            allowed_groups.append((i, 1))
        if params['noEmptyPair']:
            allowed_groups.remove((1, 0))
            allowed_groups.remove((0, 1))
        if not params['no2_2Group']:
            allowed_groups.append((2, 2))
    params['allowedGroups']=allowed_groups
    if params['verbose']:
        print(f"Allowed groups : {allowed_groups}")

    # reading files
    (sents1, id_sents1, len_sents1, seg2sents1, nb_chars1,pre_anchors_x,xml_root1) = read_input_file(params, file1, params['splitSent1'],col1, l1)
    (sents2, id_sents2, len_sents2, seg2sents2, nb_chars2,pre_anchors_y,xml_root2) = read_input_file(params, file2, params['splitSent2'],col2, l2)
  

    params['verbose'] and print(f"len(pre_anchors_x)={len(pre_anchors_x)}, len(pre_anchors_x)={len(pre_anchors_y)}")

    # dans le cas où les fichiers sont alignés, on saute toute l'étape d'alignement phrastique
    if not params['alreadyAligned'] and not params['alignedFileName']:
        # checking if anchors are coherent
        if len(pre_anchors_x) != len(pre_anchors_y) :
            if params['matchFirstPreAnchors']:
                if len(pre_anchors_x) > len(pre_anchors_y):
                    pre_anchors_x=pre_anchors_x[:len(pre_anchors_y)]
                else:
                    pre_anchors_y=pre_anchors_y[:len(pre_anchors_x)]
                print("*************** Prealignment anchor mismatch ! only first ",len(pre_anchors_x)," anchors are kept !")
            else:
                print("*************** Prealignment anchor mismatch ! anchors will be ignored !")
                pre_anchors_x.clear()
                pre_anchors_y.clear()
            
        if len_sents1 * len_sents2 == 0:
            print(f"File is empty ! No sentence read : len_sents1={len_sents1} len_sents2={len_sents2}")
            return
        # computing output file names
        if output_file_name == "":
            m = re.search(params['filePattern'], file1)

            if m:
                name1 = m.group(1)
                m = re.search(params['filePattern'], file2)
                name2 = m.group(1)
                if name1 != name2:
                    name = name1 + "-" + name2
                else:
                    name = name1
                output_file_name = name + "." + l1 + "-" + l2
                output_anchor_filename = name + ".anchor." + l1 + "-" + l2
            else:
                output_file_name = os.path.basename(file1) + "-" + os.path.basename(file2)
                output_anchor_filename = file1 + "-" + file2 + ".anchor"
        else:
            output_anchor_filename = output_file_name + ".anchor"

        ####################################################### extract candidate anchor points here !

        # =====> STEP 1-5 : extracting anchor points from similarity matrix

        if params['useNgrams']:
            (points, x, y, sim_mat) = compute_points_from_ngrams(params,sents1, sents2)  
        else:
            (points, x, y, sim_mat, embeds1, embeds2) = compute_points_with_encoder(params,preprocessor, encoder, sents1, sents2, embed_shelve)

        # prints the similarity matrix
        if params['showSimMat']:
            print(sim_mat)
            plt.imshow(sim_mat, vmin=0.2,  cmap='hot',origin='lower')
            plt.show()

        # adding anchor points and deleting mismatching coordinates
        if len(pre_anchors_x) > 0 :
            for x_anchor,y_anchor in zip(pre_anchors_x,pre_anchors_y):
                params['verbose'] and print("Anchor :",x_anchor,y_anchor)
                insertPoint=len(x)
                for i in range(len(x)):
                    if x[i] == x_anchor:
                        # deleting old x_anchor,y point 
                        if y[i]!=y_anchor and (x[i],y[i]) in points:
                            params['verbose'] and print(f"Conflict with pre anchor Deleting point [x[i],y[i]]={[x[i],y[i]]}")
                            del(points[(x[i],y[i])])
                        break
                
                for j in range(len(y)):
                    if y[j] == y_anchor:
                        # deleting old x,y_anchor
                        if x[j]!=x_anchor: 
                            params['verbose'] and print(f"Conflict with pre anchor Deleting point [x[i],y[i]]={[x[j],y[j]]}")
                            if (x[j],y[j]) in points :
                                del(points[(x[j],y[j])])
                        break
                 
                # [x_anchor,y_anchor] point has no conflicts
                points[(x_anchor,y_anchor)]=1
            
            # sorting points according to first coordinate
            points={ point:1 for point in sorted(list(points.keys()),key=lambda point:point[0])}

        #######################################################  extract filtered anchor points here !

        # =====> STEP 6-8 : filtering anchor points and extracting alignable intervals

        (filtered_x, filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1,
         interval_length_char2) = extract_anchor_points(params,pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat)

        # In this mode, the char and sent ratios are recomputed according to the aligned intervals
        # Then the anchor points are reextracted more finely using these values
        if params['adaptativeMode']:
            params['sentRatio'] = interval_length_sent2 / interval_length_sent1
            params['charRatio'] = interval_length_char2 / interval_length_char1
            print(f"Adapted ratios : sentRatio={sentRatio} charRatio={charRatio}")
            (filtered_x, filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1,
             interval_length_char2) = extract_anchor_points(params, pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat)

        # write the intervals output file if necessary
        if params['writeIntervals'] and len(intervals) > 0:
            output_interval_filename = output_anchor_filename.replace(".anchor", ".intervals") + ".txt"
            f_int = open(output_interval_filename, mode="w", encoding="utf8")
            for interval in intervals:
                (x1, y1) = interval[0]
                (x2, y2) = interval[1]
                # here sentence num starts from 1
                f_int.write(f"{x1 + 1}-{x2 + 1}\t{y1 + 1}-{y2 + 1}\n")
            f_int.close()

        # write anchor point output
        if (len(filtered_x) > 0):
            if params['writeAnchorPoints']:
            
                x_final = []
                y_final = []
                score = 0
                nbScore = 0
                for (x2, y2) in zip(filtered_x, filtered_y):
                    if sim_mat[x2, y2] >= params['cosThresholdInOutputAnchors']:
                        x_final.append(x2)
                        y_final.append(y2)
                        score += sim_mat[x2, y2]
                        nbScore += 2
                if nbScore > 0:
                    mean_score = score / nbScore
                    for output_format in output_formats:
                        write_aligned_points(params, sents1, id_sents1, sents2, id_sents2, x_final, y_final, output_dir,
                                             output_anchor_filename + "." + output_format, output_format, True, print_ids,
                                             mean_score)
                else:
                    print("No anchor points over the cos Threshold")


            # display of the points : eliminated points are red
            if params['savePlot'] or params['showPlot']:

                plt.axis([1, len_sents1, 1, len_sents2])
                plt.autoscale()
                plt.title(output_file_name + '.txt - filtered')
                plt.scatter(x, y, c="red", s=1)
                plt.scatter(filtered_x, filtered_y, c="black", s=1)
                for interval in intervals:
                    (i1, j1) = interval[0]
                    (i2, j2) = interval[1]
                    X = [i1, i1, i2, i2, i1]
                    Y = [j1, j2, j2, j1, j1]
                    plt.plot(X, Y, c="grey")
                if params['savePlot']:
                    plt.savefig(os.path.join(output_dir, output_file_name) + '.png')
                if params['showPlot']:
                    plt.show()
                plt.close()

            # writing intervals
            if len(intervals) > 0 and params['writeAlignableArea']:
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                write_alignable(sents1, id_sents1, intervals, 0, output_dir, file1 + "." + output_format, output_format)
                write_alignable(sents2, id_sents2, intervals, 1, output_dir, file2 + "." + output_format, output_format)

        # If no interval is alignable
        if interval_length_sent1 == 0 or interval_length_sent2 == 0:
            if print_log:
                log.write(
                    f"{output_file_name} not alignable\t{l1}={len(sents1)}\t{l2}={len(sents2)}\tmean_score={0}\tsilence1={1:.3f}\tsilence2={1:.3f}\tcommandLine=" + " ".join(
                        sys.argv) + "\n")
            if params['verbose']:
                print(f"{output_file_name} not alignable")
            return

        # =====> STEP 9 : extracting complete alignment using DTW

        if not params['doNotRunDTW']:
            char_ratio = nb_chars2 / nb_chars1 if params['charRatio'] == 0 else params['charRatio']
            params['verbose'] and print("Chararacter ratio=", char_ratio)

            (dtw_path, score) = run_dtw(params, encoder, sents1, sents2, intervals, filtered_x, filtered_y, pre_anchors_x, sim_mat, embeds1,
                                        embeds2, char_ratio, embed_shelve)
            # x_dtw and y_dtw contains a list of list of corresponding coordinates
            # eg. x_dtw=[[0],[1,2],[]]
            # eg. y_dtw=[[0],[1],[2]]

            x_dtw = []
            y_dtw = []
            nb_x = 0
            nb_y = 0

            if params['useShelve']:
                encode_hash = embed_shelve
            else:
                encode_hash = {}

            # Chaining the points

            # adding empty pairs at the end
            (last_x, last_y) = dtw_path[-1]
            x_l = list(range(last_x + 1, len_sents1 - 1))
            y_l = list(range(last_y + 1, len_sents2 - 1))

            if len(x_l) > 0:
                x_dtw.append(x_l)
                y_dtw.append([])
                if params['veryVerbose']:
                    print(f"Empty pair=([{x_l}],[])")
            if len(y_l) > 0:
                x_dtw.append([])
                y_dtw.append(y_l)
                if params['veryVerbose']:
                    print(f"Empty pair=([],[{y_l}])")
            # constitution des groupes en fonctions des bornes
            for i in range(len(dtw_path) - 1, -1, -1):
                if dtw_path[i] != ():
                    (x, y) = dtw_path[i]
                    # if the point is not empty (interval border)
                    if i >= 1 and dtw_path[i - 1] != ():
                        (prev_x, prev_y) = dtw_path[i - 1]
                        x_l = list(range(prev_x + 1, x + 1))
                        y_l = list(range(prev_y + 1, y + 1))

                        if len(x_l) > 0 or len(y_l) > 0:
                            x_dtw.append(x_l)
                            y_dtw.append(y_l)
                        nb_x += len(x_l)
                        nb_y += len(y_l)
                    # if the point is the first of the interval, then use (x,y) as a simple point
                    else:
                        # ~ x_dtw.append([x])
                        # ~ y_dtw.append([y])
                        # ~ nb_x+=1
                        # ~ nb_y+=1
                        # creating empty pairs for each gap
                        if i - 2 >= 0 and dtw_path[i - 2] != () and params['printGap']:
                            (prev_x, prev_y) = dtw_path[i - 2]
                            x_l = list(range(prev_x + 1, x + 1))
                            y_l = list(range(prev_y + 1, y + 1))

                            if len(x_l) > 0:
                                x_dtw.append(x_l)
                                y_dtw.append([])
                                if params['veryVerbose']:
                                    print(f"Empty pair=([{x_l}],[])")
                            if len(y_l) > 0:
                                x_dtw.append([])
                                y_dtw.append(y_l)
                                if params['veryVerbose']:
                                    print(f"Empty pair=([],[{y_l}])")

            # ~ print(f"first x={x},first y={y}")
            # adding first empty pair
            if params['printGap']:
                x_l = list(range(0, x))
                y_l = list(range(0, y))
                if len(x_l) > 0:
                    x_dtw.append(x_l)
                    y_dtw.append([])
                    if params['veryVerbose']:
                        print(f"Empty pair=([{x_l}],[])")
                if len(y_l) > 0:
                    x_dtw.append([])
                    y_dtw.append(y_l)
                    if params['veryVerbose']:
                        print(f"Empty pair=([],[{y_l}])")

            x_dtw.reverse()
            y_dtw.reverse()

            # grouping points may occur here
            if params['lateGrouping']:
                (x_dtw, y_dtw) = late_grouping(x_dtw, y_dtw, encoder, sents1, sents2, encode_hash, sim_mat, embeds1,
                                               embeds2, char_ratio)

            # writing output files
            mean_score = len(sents1) + len(sents2) - score
            if params['verbose'] and len(sents1) > 0:
                mean_score = 1 - (score / (len(sents1) + len(sents2)))
                print(f"Average similarity={mean_score:.4f}")
            silence1 = (len(sents1) - nb_x) / len(sents1)
            silence2 = (len(sents2) - nb_y) / len(sents2)

            if print_log:
                log.write(
                    f"{output_file_name}\t{l1}={len(sents1)}\t{l2}={len(sents2)}\tmean_score={mean_score}\tignored1={(len(sents1) - nb_x)}\tsilence1={silence1:.3f}\tignored2={(len(sents2) - nb_y)}\tsilence2={silence2:.3f}\tcommandLine=" + " ".join(
                        sys.argv) + "\n")
            # print(f"{input_format=} {add_anchor=}")
            
    elif params['alignedFileName']:
        (x_dtw,y_dtw)=read_alignment_file(params)
        mean_score=0
    else:
        # todo : attention, corriger si split_sent, ou si on a des alignements vides
        x_dtw=[[x] for x in pre_anchors_x]
        y_dtw=[[y] for y in pre_anchors_y]
        
    # write alignment files
    for output_format in output_formats:
        if output_format == "xml" and input_format == "xml" and add_anchor:
            if not file_id1:
                file_id1 = l1
            if not file_id2:
                file_id2 = l2
            params['verbose'] and print("Add anchors in XML", file1, file2)
            add_anchor_in_output(params, file1, file2, xml_root1, xml_root2, file_id1, file_id2, x_dtw, y_dtw)
        else:
            write_aligned_points(params, sents1, id_sents1, sents2, id_sents2, x_dtw, y_dtw, output_dir,
                                 output_file_name + "." + output_format, output_format, False, print_ids,
                                 mean_score, file1, file2)

   
    if params['useShelve']:
        embed_shelve.close()
                
    # =====> STEP 10 : parse aligned sentence, extract chunks and align chunk to get word 2 word alignment
   
    output_formats = params.get("outputFormats")
    if params.get('chunkAlignment', True):
        params['verbose'] and  print("Starting Chunk alignment....")
        chunk_alignment(l1, l2, x_dtw, y_dtw, encoder, sents1, sents2, output_file_name, output_dir, output_formats)

    if params.get('wordAlignment', True):
        params['verbose'] and print("Starting Word alignment....")
        word_alignment(l1, l2, x_dtw, y_dtw, encoder, sents1, sents2, output_file_name, output_dir, output_formats)
    return mean_score




    # for group [x_inf,..,x_sup], return the interval [x_inf-1,x_sup] (to use in distance_DTW)


# for group [y_inf,..,y_sup], return the interval [y_inf-1,y_sup] (to use in distance_DTW)
def calc_int(group_x, group_y):
    if len(group_x) == 0:
        x_inf = 0
        x_sup = 0
    else:
        x_inf = group_x[0] - 1
        x_sup = group_x[-1]
    if len(group_y) == 0:
        y_inf = 0
        y_sup = 0
    else:
        y_inf = group_y[0] - 1
        y_sup = group_y[-1]
    return (x_inf, x_sup, y_inf, y_sup)


# apply a greedy algorithme to perform the best grouping (which increase sim between source and target)
def late_grouping(x_dtw, y_dtw, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio):
    # this btree records the index of each group ordered by their gain
    gains = OOBTree()
    groups = []
    # initialisation of the groups data structure : foreach group, record x,y, and the corresponding dist
    for (group_x, group_y) in zip(x_dtw, y_dtw):
        (inf_x, sup_x, inf_y, sup_y) = calc_int(group_x, group_y)
        dist = distance_dtw(encoder, sents1, sents2, encode_hash, {}, sim_mat, embeds1, embeds2, inf_x, sup_x, inf_y,
                            sup_y, char_ratio, False)
        groups.append({'x': group_x, 'y': group_y, "dist": dist})

    # first iteration : for each group, the gain of similarity is computed whether grouping
    # on the left or on the right (direction indicates which direction has the best gain)
    # all the strictly positive gains are recorded in the gains btree
    for i in range(len(groups)):
        compute_gain(gains, groups, i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio)

    if len(gains) > 0:
        best_gain = gains.maxKey()
    else:
        best_gain = 0
    # while best grouping produce a positive gain in similarity
    while best_gain > 0:
        i = gains[best_gain][-1]

        group_x = groups[i]['x']
        group_y = groups[i]['y']

        # group i with next group
        if groups[i]['direction'] == 1:
            next_i = next(groups, i)
            params['verbose'] and print(f"group {i} with next {next_i} :", groups[i]['newX'], groups[i]['newY'])
            if next_i != -1:
                # the next group is first "deleted" : dist is set to -1, and x and y are set to []
                groups[next_i]['dist'] = -1
                groups[next_i]['x'] = []
                groups[next_i]['y'] = []
            else:
                print("Wrong direction")
                break
        elif groups[i]['direction'] == -1:
            # group i with previous group
            prev_i = prev(groups, i)
            params['verbose'] and print(f"group {i} with prev {prev_i} :", groups[i]['newX'], groups[i]['newY'])
            if prev_i != -1:
                # the prev group is first "deleted" : dist is set to -1, and x and y are set to []
                groups[prev_i]['dist'] = -1
                groups[prev_i]['x'] = []
                groups[prev_i]['y'] = []
            else:
                print("Wrong direction")
                break
        else:
            print("No direction", i, groups[i])
            break

        # setting the new group with the recorded merging
        groups[i]['x'] = groups[i]['newX']
        groups[i]['y'] = groups[i]['newY']
        groups[i]['dist'] = groups[i]['newDist']

        # update of the gain, on the left and on the right (after the prev group or the next_group which are "deleted")
        compute_gain(gains, groups, i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio)

        # update gain on the left and right side
        prev_i = prev(groups, i)
        if prev_i != -1:
            compute_gain(gains, groups, prev_i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2,
                         char_ratio)

        next_i = next(groups, i)
        if next_i != -1:
            compute_gain(gains, groups, next_i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2,
                         char_ratio)

        # computing best gain for next iteration
        if len(gains) > 0:
            best_gain = gains.maxKey()
        else:
            best_gain = 0

    # returning final groups
    x_dtw = []
    y_dtw = []
    for i, group in enumerate(groups):
        if groups[i]['dist'] != -1:
            x_dtw.append(groups[i]['x'])
            y_dtw.append(groups[i]['y'])
    return (x_dtw, y_dtw)


# compute the gain when grouping on the left (direction=-1) or on the right (direction=1) side
# and record the corresponding merged groups and distance
def compute_gain(gains, groups, i, encoder, sents1, sents2, encode_hash, sim_mat, embeds1, embeds2, char_ratio):
    group_x = groups[i]['x']
    group_y = groups[i]['y']
    dist = groups[i]['dist']

    # removing i for old_gain if any
    if 'gain' in groups[i]:
        old_gain = groups[i]['gain']
        if old_gain > 0 and i in gains[old_gain]:
            gains[old_gain].remove(i)
            # removing the old gain key if necessary
            if len(gains[old_gain]) == 0:
                del (gains[old_gain])

    # no gain is computed for empty groups
    if len(group_x) == 0 or len(group_y) == 0:
        groups[i]['gain'] = 0
        return

    prev_i = prev(groups, i)
    prev_gain = 0
    if prev_i != -1:
        prev_group_x = groups[prev_i]['x']
        prev_group_y = groups[prev_i]['y']
        no_empty = len(prev_group_x) > 0 and len(prev_group_y) > 0
        new_group_x1 = prev_group_x + group_x
        new_group_y1 = prev_group_y + group_y
        (inf_x, sup_x, inf_y, sup_y) = calc_int(new_group_x1, new_group_y1)
        prev_dist = distance_dtw(encoder, sents1, sents2, encode_hash, {}, sim_mat, embeds1, embeds2, inf_x, sup_x,
                                 inf_y, sup_y, char_ratio, False)
        prev_gain = dist - prev_dist
        if no_empty:
            prev_gain -= params['penalty_n_n']
        else:
            prev_gain += params['penalty_0_n']
        # ~ print(i,"prev",no_empty,prev_gain,new_group_x1,new_group_y1,dist,prev_dist,prev_gain)

    next_i = next(groups, i)
    next_gain = 0
    if next_i != -1:
        next_group_x = groups[next_i]['x']
        next_group_y = groups[next_i]['y']
        no_empty = len(next_group_x) > 0 and len(next_group_y) > 0
        new_group_x2 = group_x + next_group_x
        new_group_y2 = group_y + next_group_y
        (inf_x, sup_x, inf_y, sup_y) = calc_int(new_group_x2, new_group_y2)
        next_dist = distance_dtw(params,encoder, sents1, sents2, encode_hash, {}, sim_mat, embeds1, embeds2, inf_x, sup_x,
                                 inf_y, sup_y, char_ratio, False)
        next_gain = dist - next_dist
        if no_empty:
            next_gain -= params['penalty_n_n']
        else:
            next_gain += params['penalty_0_n']
        # ~ print(i,"next",no_empty,next_gain,new_group_x2,new_group_y2,dist,next_dist,next_gain)

    if next_gain > prev_gain and next_gain > 0:
        groups[i]['gain'] = next_gain
        groups[i]['direction'] = 1
        groups[i]['newX'] = new_group_x2
        groups[i]['newY'] = new_group_y2
        groups[i]['newDist'] = next_dist
        gain = next_gain
    elif prev_gain > 0:
        groups[i]['gain'] = prev_gain
        groups[i]['direction'] = -1
        groups[i]['newX'] = new_group_x1
        groups[i]['newY'] = new_group_y1
        groups[i]['newDist'] = prev_dist
        gain = prev_gain
    else:
        groups[i]['gain'] = 0
        groups[i]['direction'] = 0
        gain = 0

    # updating gains btree
    if gain > 0:
        if not gains.has_key(gain):
            gains[gain] = []
        if i not in gains[gain]:
            gains[gain].append(i)

    return gain


# search for the previous non deleted group before group i
def prev(groups, i):
    if i == 0:
        return -1
    i -= 1
    while i > 0 and groups[i]["dist"] == -1:
        i -= 1
    if groups[i]["dist"] == -1:
        return -1
    else:
        return i


# search for the next non deleted group after group i
def next(groups, i):
    if i == len(groups) - 1:
        return -1
    i += 1
    while i < len(groups) - 1 and groups[i]["dist"] == -1:
        i += 1
    if groups[i]["dist"] == -1:
        return -1
    else:
        return i



def run_dtw(params, encoder, sents1, sents2, intervals, filtered_x, filtered_y, pre_anchors_x, sim_mat, embeds1, embeds2, char_ratio, embed_shelve):
    """
    Run the Dynamic time warping algorithm (Viterbi) by computing all the paths
    from each anchor points (the paths must not deviate from these anchors points
    at a distance lower than dtwBeam)
        
    arg1 (dict): the global parameters
    arg2 (funct): the encoding function
    arg4 (list[str]): the sentence list in lang1
    arg5 (list[str]): the sentence list in lang2
    arg6 (list[([x1,x2],[y1,y2])]) : a list of aligned intervals
    arg7 (list[int]) : the filtered list of  x coordinates for points
    arg8 (list[int]) : the filtered list of  y coordinates for points
    arg9 (list[int]) : pre_anchors_x - the x coordinates of pre anchors (pre anchors are give in the input files)
    arg10 (list[list]) : the similarity matrix
    arg11 (list[tensor]): the embeddings that correspond to sents1
    arg12 (list[tensor]): the embeddings that correspond to sents2
    arg13 (float) : the char ratio nb_chars2 / nb_chars1
    arg14 (shelve) : the embedding shelve to record embedding persistently (to save time when running again)
    
    Returns a tuple with the following data :
        best_path (list[pairs]): a list of pairs (coordinates) that represent the best path
            Each coordinate represents the upper bound of the point (the lower bound is computed
            as the next coordinate of the previous point).
            The first point that representents the beginning of the path is [-1,-1].
            For instance [[-1,-1],[0,1],[3,2]] represents the following alignments :
            (0:0,1), (1,2,3:2)
        score (float): the cumulative score of the best_bath (the sum of the distances)
    """

    global infinite
    path_hash = {}
    dist_hash = {"-2--1;-2--1": 0}  # for the point (-1,-1), the lower bound

    if params['useShelve']:
        encode_hash = embed_shelve
    else:
        encode_hash = {}

    # initialization for the NULL path
    x_first = intervals[0][0][0]
    y_first = intervals[0][0][1]
    
    path_hash[(x_first,y_first)] = [[[-1, -1]], 0]
    params['verbose'] and print(f"First point : {x_first}-{y_first}")
    print(f"Init : dtw from ", intervals[0][0], " to ", intervals[-1][1])

    lastBestPath = [[intervals[0][0][0] - 1, intervals[0][0][1] - 1]]
    lastBestScore = 0

    t8 = time.time()
    # process each alignable intervals
    for interval in intervals:
        (x_begin, y_begin) = interval[0]
        (x_end, y_end) = interval[1]
        print (f"Current interval {interval}")
        key_xy = (x_begin,y_begin)
        coeff_y_per_x = (y_end - y_begin) / (x_end - x_begin)

        # these dict allow to drive the paths near the anchor points that are located IN the interval
        x_2_y = {}
        y_2_x = {}
        for i in range(len(filtered_x)):
            x = filtered_x[i]
            y = filtered_y[i]
            if x < x_begin:
                continue
            if x > x_end:
                break
            if y < y_begin or y > y_end:
                continue
            x_2_y[x] = y
            y_2_x[y] = x

        # Bridging the gap between alignable intervals
        # if there is a gap between the last point in path and the first point in current interval, add an empty point () in the path
        if key_xy not in path_hash:
            (lastI, lastJ) = lastBestPath[-1]
            if params['verbose']:
                print(f"Inserting gap between ({lastI},{lastJ}) and ({x_begin},{y_begin})")
            lastBestPath.append(())  # an empty point indicate a break in the path
            lastBestPath.append((x_begin - 1, y_begin - 1))
            path_hash[key_xy] = [lastBestPath, lastBestScore]

        # now run the DTW search between each anchor point in the interval
        # the path are computed recursively, but in order to minimize the recursive depth, the
        # dtw hash is progressively filled by calling the function point by point
        previous_x = x_begin
        previous_y = y_begin
        for x in range(x_begin, x_end + 1):
            localBeam = params['dtwBeam']

            # if (x,y) is an anchor point, run dtw from x !
            if x in x_2_y:
                y = x_2_y[x]
                if params['verbose']:
                    print(f"Anchor point {x},{y}")

                # if it is a preanchor the point cannot be discarded and the local beam is null
                if x in pre_anchors_x:
                    localBeam=0
                else :
                    # computing deviation and beam
                    # if (x,y) is too far from the interval diagonal, it is discarded
                    deviation = 0
                    if y >= y_begin and (x_end-x_begin)*(y_end-y_begin)!=0:
                        deviation = abs((y - y_begin) / (y_end - y_begin) - (x - x_begin) / (x_end - x_begin))
                    else:
                        continue

                    # First condition : 1/ deviation > localDiagBeam
                    if (deviation > params['localDiagBeam'] and deviation * (y_end - y_begin) > params['dtwBeam']):
                        del x_2_y[x]
                        if y in y_2_x:
                            del y_2_x[y]
                        if params['verbose']:
                            print(
                                f"deviation*(y_end-y_begin)= {deviation * (y_end - y_begin)} - Anchor point ({x},{y}) is too far from the interval diagonal - point has been discarded!")
                        continue
                    # Second condition : 2/ the ratio between deltaX and deltaY exceeds 4 (1-4 or 4-1 grouping is the max allowed)
                    if (params['noEmptyPair'] and (
                            min(y - previous_y, x - previous_x) == 0 or max(y - previous_y, x - previous_x) / min(
                            y - previous_y, x - previous_x) > 4)):
                        del x_2_y[x]
                        if y in y_2_x:
                            del y_2_x[y]
                        if params['verbose']:
                            print(
                                f"Deviating anchor point ({x},{y}) is too close from the preceding - point has been discarded!")
                        continue

                    # Processing of the gaps (taking into account non monotony) :
                    # if y < previous_y, the area is enlarged : y will be set equal to previous_y and previous_x is decreased, to correspond 
                    # to the last point with x_2_y[prev_x] < y
                    
                    if y < previous_y:
                        print(f"Monotonic discrepancy : y={y} < previous_y={previous_y}. Recomputing previous_x.")
                        prev_x=previous_x
                        # looking for previous point according to y
                        found=False
                        while prev_x > x_begin:
                            prev_x -= 1
                            if prev_x in x_2_y:
                                prev_y = x_2_y[prev_x]
                                if prev_y < y:
                                    y=previous_y
                                    previous_x=prev_x
                                    previous_y=prev_y
                                    found=True
                                    break
                        if not found:
                             y=previous_y
                             previous_x=x_begin
                             previous_y=y_begin     
                
                if params['veryVerbose']:
                    print(f"Running DTW for the point : ({x},{y}) - elapsed from (1,1) =", time.time() - t8, "s.")

   
                # compute the inferior values to give an interval to cut recursion : points that are before
                # x_inf,y_inf should not be considered
                x_inf = previous_x - localBeam
                y_inf = previous_y - localBeam
                
                print( f"Lancement de DTW entre ({max(x_begin, x_inf)},{max(y_begin, y_inf)}) et ({x},{y})")
                (path, dist) = dtw(params, encoder, sents1, sents2, encode_hash, path_hash, dist_hash, x_2_y, y_2_x,
                                   sim_mat, embeds1, embeds2,max(x_begin, x_inf),max(y_begin, y_inf),x, y, char_ratio)

                if dist == infinite and params['verbose']:
                    print(f"Infinite distance from : ({x},{y})")
                    # initiating a new interval starting from x,y
                    x_begin = x
                    y_begin = y
                    key_xy = (x_begin,y_begin)
                    # here creation of a copy of lastBestPath, and addition of the breakpoint
                    lastBestPath = lastBestPath[:]
                    lastBestPath.append(())  # an empty point indicate a break in the path
                    lastBestPath.append((x_begin - 1, y_begin - 1))
                    path_hash[key_xy] = [lastBestPath, lastBestScore]
                    # ~ sys.exit()
                else:
                    lastBestPath = path
                    lastBestScore = dist

                if params['veryVerbose']:
                    print(f"Distance->{dist}")
                previous_x = x
                previous_y = y

        (lastBestPath, lastBestScore) = path_hash[(previous_x,previous_y)]

    # chaining with the end of the text
    last_x = len(sents1) - 1
    last_y = len(sents2) - 1
    if (last_x - x) + (last_y - y) < 200:
        if params['verbose']:
            print(f"Last point ({last_x},{last_y})")
        dtw(params, encoder, sents1, sents2, encode_hash, path_hash, dist_hash, x_2_y, y_2_x, sim_mat, embeds1, embeds2, x_end, y_end, last_x,
            last_y, char_ratio)
    # if last point has not been discarded
    score = infinite
    if (last_x,last_y) in path_hash:
        (best_path, score) = path_hash[(last_x,last_y)]
    # the last interval is used instead
    if score == infinite:
        (best_path, score) = path_hash[(previous_x,previous_y)]

    t9 = time.time()
    if params['verbose']:
        print(f"\n9. Elapsed time for complete DTW-->", t9 - t8, "s.\n")

    return (best_path, score)


# 
# The current point correspond to the interval between (infI,inJ) excluded
def dtw(params, 
        encoder, 
        sents1, 
        sents2, 
        encode_hash, 
        path_hash, 
        dist_hash, 
        x_2_y, 
        y_2_x, 
        sim_mat, 
        embeds1, 
        embeds2, 
        x_begin, 
        y_begin,         
        x_end, 
        y_end,
        char_ratio):
    """
    Compute the bestpath in the interval (x_begin,y_begin)-(x_end,y_end).
        
    arg1 (dict): the global parameters
    arg2 (funct): the encoding function
    arg3 (list[str]): the sentence list in lang1
    arg4 (list[str]): the sentence list in lang2
    arg5 (dict) : a hash that records the encoding for groups of sentences
    arg6 (dict[(x,y):(path,score)]) : a hash that records the (bestPath,score) that leads to a given point.
    arg7 (dict) : a hash that records the distance between a pair of groups of sentences
    arg8 (dict) : a hash that gives the corresponding y coordinate for the x coord of an anchor point
    arg9 (dict) : a hash that gives the corresponding x coordinate for the y coord of an anchor point
    arg10 (list[list]) : the similarity matrix   
    arg11 (list[tensor]) : the embeddings that correspond to sents1
    arg12 (list[tensor]) : the embeddings that correspond to sents2
    arg13 (int) : the x coordinate of the beginning of the interval
    arg14 (int) : the y coordinate of the beginning of the interval
    arg15 (int) : the x coordinate of the end of the interval
    arg16 (int) : the y coordinate of the end of the interval
    arg16 (float) : the char ratio nb_chars2 / nb_chars1
    
    Returns (list[pairs]): the best path to (x_end,y_end)

    """
            
    global infinite
    
            
    for i in range(x_begin,x_end+1):
        for j in range(y_begin,y_end+1):
            # The hash path_hash records the result for already computed path, in order to reduce recursivity
            dtw_key = (i,j)

            # skip if already computed
            if dtw_key in path_hash:
                continue

            path_by_group = {}
            dist_by_group = {}
            # on examine chaque groupe
            for group in params['allowedGroups']:
                previous_i=i - group[0]
                previous_j=j - group[1]
                previous_key= (previous_i,previous_j)
                
                # en principe, previous_key doit être trouvée
                if previous_key in path_hash:
                    (path_by_group[group], dist_by_group[group])=path_hash[previous_key]
                    # ~ print (f"path_hash[{previous_key}]={path_hash[previous_key]}")
                else:
                    # ~ print (f"{previous_key=} pas trouvée")
                    (path_by_group[group], dist_by_group[group])=([], infinite)
                
                # on incrémente la distance pour le groupe courant
                dist_by_group[group] += distance_dtw(params,encoder, sents1, sents2, encode_hash, dist_hash, sim_mat, embeds1, embeds2,
                                                     previous_i, i, previous_j, j,
                                                     char_ratio) 

            best_group = None
            min_dist = infinite
            for group in params['allowedGroups']:
                if dist_by_group[group] < min_dist:
                    min_dist = dist_by_group[group]
                    best_group = group
            if best_group != None:
                path = path_by_group[best_group][:]  # warning here, create a copy !
                path.append([i, j])
                path_hash[dtw_key] = [path, min_dist]
            else:
                path_hash[dtw_key] = [[], infinite]
   
    return path_hash[(x_end,y_end)]


def distance_dtw(
    params, 
    encoder, 
    sents1, 
    sents2, 
    encode_hash, 
    dist_hash, 
    sim_mat, 
    embeds1, 
    embeds2, 
    inf_i, 
    i, 
    inf_j, 
    j,
    char_ratio, 
    use_coeff=True):
    
    """
    Compute the distance (1-cosinus) for the point defined by (inf_i,i-1),(inf_j,i-1)
    For empty aligning, dist is equal to distNull which should be near to 1
    When the similarity is below a given threshold (sim_threshold), the dist is fixed to 1 (in order to force using 1-0 or 0-1 pairing)
        
    arg1 (dict): the global parameters
    arg2 (funct): the encoding function
    arg3 (list[str]): the sentence list in lang1
    arg4 (list[str]): the sentence list in lang2
    arg5 (dict): a hash that records the encoding for groups of sentences
    arg7 (dict): a hash that records the distance between a pair of groups of sentences
    arg8 (dict): the similarity matrix
    arg9 (list[tensor]): the embeddings that correspond to sents1
    arg10 (list[tensor]): the embeddings that correspond to sents2
    arg11 (int): the x coordinate of the beginning of the interval
    arg12 (int): the y coordinate of the beginning of the interval
    arg13 (int): the x coordinate of the end of the interval (excluded)
    arg14 (int): the y coordinate of the end of the interval (excluded)
    arg15 (float): the char ratio nb_chars2 / nb_chars1
    arg16 (bool, optional): indicates whether the cos should be multiplieds by the coeff, which is
                            the size of the grouping (the number of sentences in the group)
                            If no coeff is used, 2-2, or 3-3 pairing will results in shorter path than 1-1
                            which is not the targeted behaviour
    """
    
    global infinite, coeff_neighbour_sim, coeff_sent_len
        
    # if the distance has already been stored in dist_hash
    key = str(inf_i) + "-" + str(i) + ";" + str(inf_j) + "-" + str(j)
    if key in dist_hash:
        return dist_hash[key]

    # coeff indicates the total number of segments (for both language) involved in the alignment
    coeff = 1
    penalty = params['penalty_n_n']

    # case of relations 1-0 et 0-1
    if inf_i == i or inf_j == j:
        return params['distNull'] * coeff

    if i < 0 or j < 0 or inf_i < -2 or inf_j < -2:
        return infinite

    coeff = 2
    if params['useEncoder']:
        # similarity are computed for sentence group
        # case of relations 1-1
        if inf_i == i - 1 and inf_j == j - 1:
            sim = sim_mat[i, j]
            if use_coeff:
                penalty = 0
        # case of relations n-n
        else:
            # calculate embed_i
            if inf_i == i - 1:
                embed_i = embeds1[i][:]
                len_i = len(sents1[inf_i + 1])
            else:
                sent_i = sents1[inf_i + 1]
                for coord_i in range(inf_i + 2, i + 1):
                    sent_i += " " + sents1[coord_i]
                    if use_coeff:
                        coeff += 1
                len_i = len(sent_i)
                if sent_i not in encode_hash:
                    embed_i = encoder.encode([sent_i])
                    embed_i = embed_i / np.linalg.norm(embed_i)  # normalize
                    encode_hash[sent_i] = embed_i
                else:
                    embed_i = encode_hash[sent_i]
            # calculate embed_j
            if inf_j == j - 1:
                embed_j = embeds2[j][:]
                len_j = len(sents2[inf_j + 1])
            else:
                sent_j = sents2[inf_j + 1]
                for coord_j in range(inf_j + 2, j + 1):
                    sent_j += " " + sents2[coord_j]
                    if use_coeff:
                        coeff += 1
                len_j = len(sent_j)
                if sent_j not in encode_hash:
                    embed_j = encoder.encode([sent_j])
                    embed_j = embed_j / np.linalg.norm(embed_j)  # normalize
                    encode_hash[sent_j] = embed_j
                else:
                    embed_j = encode_hash[sent_j]
            sim = float(np.matmul(embed_i, np.transpose(embed_j)))
    else:
        # similarity are computed with vector addition
        # case of relations 1-1 : no penalty
        if inf_i == i - 1 and inf_j == j - 1:
            penalty = 0

        embed_i = embeds1[inf_i + 1][:]
        embed_j = embeds2[inf_j + 1][:]

        len_i = len(sents1[inf_i + 1])
        for coord_i in range(inf_i + 2, i + 1):
            len_i += len(sents1[coord_i])
            embed_i = np.add(embed_i, embeds1[coord_i])
            if use_coeff:
                coeff += 1

        len_j = len(sents2[inf_j + 1])
        for coord_j in range(inf_j + 2, j + 1):
            len_j += len(sents2[coord_j])
            embed_j = np.add(embed_j, embeds2[coord_j])
            if use_coeff:
                coeff += 1
        try:
            norm_i = np.linalg.norm(embed_i)  # normalize
        except:
            norm_i = 0
            for k in range(len(embed_i)):
                norm_i += embed_i[k] ** 2
            norm_i = math.sqrt(norm_i)
            print(f"Plantage de linalg.norm, norme calculée directement norm_i={norm_i}")

        try:
            norm_j = np.linalg.norm(embed_j)  # normalize
        except:
            norm_j = 0
            for k in range(len(embed_j)):
                norm_j += embed_j[k] ** 2
            norm_j = math.sqrt(norm_j)
            print(f"Plantage de linalg.norm, norme calculée directement norm_j={norm_j}")

        embed_i = embed_i / norm_i  # normalize
        embed_j = embed_j / norm_j  # normalize
        sim = np.matmul(embed_i, np.transpose(embed_j))

    # compute the similarity with neighbouring sentences and substract it to the global sim
    if not params['noMarginPenalty']:
        nb = 0
        nn = 0
        if inf_j >= 0:
            left_embed_j = embeds2[inf_j][:]
            left_sim_j = np.matmul(embed_i, np.transpose(left_embed_j))
            nb += 1
        else:
            left_sim_j = 0
        if j + 1 < len(embeds2):
            right_embed_j = embeds2[j + 1][:]
            right_sim_j = np.matmul(embed_i, np.transpose(right_embed_j))
            nb += 1
        else:
            right_sim_j = 0
        neighbour_sim_j = 0
        if nb > 0:
            neighbour_sim_j = (left_sim_j + right_sim_j) / nb
            nn += 1

        nb = 0
        if inf_i >= 0:
            left_embed_i = embeds1[inf_i][:]
            left_sim_i = np.matmul(left_embed_i, np.transpose(embed_j))
            nb += 1
        else:
            left_sim_i = 0
        if i + 1 < len(embeds1):
            right_embed_i = embeds1[i + 1][:]
            right_sim_i = np.matmul(right_embed_i, np.transpose(embed_j))
            nb += 1
        else:
            right_sim_i = 0
        neighbour_sim_i = 0
        if nb > 0:
            neighbour_sim_i = (left_sim_i + right_sim_i) / nb
            nn += 1

        average_neighbour_sim = 0
        if nn > 0:
            average_neighbour_sim = (neighbour_sim_i + neighbour_sim_j) / nn
        sim -= coeff_neighbour_sim * average_neighbour_sim

    # for empty sentences
    if len_i * len_j == 0:
        return params['distNull'] * coeff

    dist = 1 - sim
    if use_coeff:
        dist += penalty * coeff

    dist = (1 - coeff_sent_len) * dist + coeff_sent_len * lenPenalty(len_i * char_ratio, len_j)

    dist *= coeff
    dist_hash[key] = dist
    return dist


# cf Bertalign
def lenPenalty(len1, len2):
    min_len = min(len1, len2)
    max_len = max(len1, len2)
    return 1 - np.log2(1 + min_len / max_len)
