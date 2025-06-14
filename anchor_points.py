# -*- coding:utf8 -*-
"""

This module defines various functions to extract anchor points using sentence embeddings (or ngram)

Main functions :

    extract_anchor_points(params,pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat)
    
    compute_points_from_ngrams(params,sents1, sents2) 
    
    compute_points_with_encoder(params,preprocessor, encoder, sents1, sents2, embed_shelve)

"""

import os
import sys
import re
import math
import time

import shelve
import math

import numpy as np

import matplotlib.pyplot as plt


# Various low level global parameters
show_plot_4_new_interval = False # for debugging purposes
min_sent_length_ratio = 0.2  # the minimal ratio between the shorter and the longer sentence to yield a candidate point
min_sent_length = 1  # the minimal sentence size to look for ngram


########################################################################### filtering points functions
# computation of local density
# local space may be centered, or before (for a point wich ends an interval)
# or after (for a point that begins an interval).
# max density is taken
def compute_local_density(params,i, j, points, max_i, max_j, sim_mat):
    """
    Compute the local density along the diagonal
    arg1 (dict) - params : the global parameters
    arg2 (int) - i : the x coordinate of the point
    arg3 (int) - j : the y coordinate of the point
    arg4 (dict[(x,y):1]) : points - a dictionnary that contains all the corresponding points, sorted upon x coordinate
    arg5 (int) : the max value of x coordinate
    arg6 (int) : the max value of y coordinate
    arg7 (list[list]) : the similarity matrix
    
    return (float) : the local anchor point density
   
    """
    
    delta_x=params['deltaX']
    delta_y=params['deltaY']
        
    coeff = max_j / max_i if params['sentRatio'] == 0 else params['sentRatio']
    local_space_size_before = 0
    nb_points_in_local_space_size_before = 0

    local_space_size_centered = 0
    nb_points_in_local_space_size_centered = 0

    local_space_size_after = 0
    nb_points_in_local_space_size_after = 0

    for X in range(max(0, i - 2 * delta_x), min(i + 2 * delta_x + 1, max_i)):
        for Y in range(int(max(0, j - (i - X) * coeff - delta_y)), int(min(j - (i - X) * coeff + delta_y + 1, max_j))):
            if X <= i:
                local_space_size_before += 1
                if (X, Y) in points.keys():
                    nb_points_in_local_space_size_before += sim_mat[X, Y]
            if X >= i:
                local_space_size_after += 1
                if (X, Y) in points.keys():
                    nb_points_in_local_space_size_after += sim_mat[X, Y]
            if max(0, i - delta_x) <= X < min(i + delta_x + 1, max_i):
                local_space_size_centered += 1
                if (X, Y) in points.keys():
                    nb_points_in_local_space_size_centered += sim_mat[X, Y]

    (densityBefore, densityAfter, densityCentered) = (0, 0, 0)
    if local_space_size_before:
        densityBefore = nb_points_in_local_space_size_before / local_space_size_before
    if local_space_size_after:
        densityAfter = nb_points_in_local_space_size_after / local_space_size_after
    if local_space_size_centered:
        densityCentered = nb_points_in_local_space_size_centered / local_space_size_centered

    return max(densityBefore, densityAfter, densityCentered)


def filter_points(params,pre_anchors_x, points, max_i, max_j, average_density, sim_mat):
    """
    Filtering points by eliminating every point in the center of a low density local area
    
    arg1 (dict) : params
    arg2 (list[int]) : the x coordinates of anchor points
    arg3 (dict[(x,y):1]) : points - a dictionnary that contains all the corresponding points, sorted upon x coordinate
    arg4 (int) : the x coordinate of the candidate point
    arg5 (int) : the y coordinate of the candidate point
    arg6 (float) : the global average_density
    arg7 (list[list]) : the similarity matrix
    
    return a tuple with the following data :
    
    points (dict[(x,y):1]): the filtered points
    filtered_x (list[int]): the x coordinate of the filtered points
    filtered_y (list[int]): the y coordinate of the filtered points
    
    """
    
    # initialisation of filtered points
    filtered_x = []
    filtered_y = []
    nbDeleted = 0

    if params['veryVerbose']:
        print("Filtering ", len(points), "...")

    # computation of local density for each point
    points_key = sorted(list(points.keys()), key=lambda point: point[0])

    for point in points_key:
        (i, j) = point

        localDensity = compute_local_density(params,i, j, points, max_i, max_j, sim_mat)

        if params['veryVerbose']:
            print("i=", i, "j=", j, "Local density=", localDensity, "Average density=", average_density, "Ratio=",
                  round(localDensity / average_density, 2))

        # point is removed if density is not high enough
        if i not in pre_anchors_x and average_density > 0 and localDensity / average_density < params['minDensityRatio']:
            del (points[(i,j)])
            nbDeleted += 1
        else:
            filtered_x.append(i)
            filtered_y.append(j)

    if params['verbose']:
        print(nbDeleted, "points have been removed!")

    return (points, filtered_x, filtered_y)

def resolving_conflicts(params,points, max_i, max_j, sim_mat):
    """
    Removing points that are conflicting on the same column : only the point with the higher local density is kept
    
    arg1 (dict) : params
    arg2 (dict[(x,y):1]) : points - a dictionnary that contains all the corresponding points, sorted upon x coordinate
    arg3 (int) : the max value of x coordinate
    arg4 (int) : the max value of y coordinate
    arg5 (float) : the global average_density
    
    returns a tuple with the following data :
    
    points (dict[(x,y):1]): the filtered points
    filtered_x (list[int]): the x coordinate of the filtered points
    filtered_y (list[int]): the y coordinate of the filtered points
    
    """    
    x2y = {}
    y2x = {}
    filtered_x = []
    filtered_y = []
    nbDeleted = 0
    points_key = list(points.keys())
    for point in points_key:
        (i, j) = point
        # conflict on x coordinate
        if i in x2y.keys():
            if x2y[i] != j:
                # for x coordinate, conflict between (i,j) and (i,x2y[i])
                # only the best point is kept
                density1 = compute_local_density(params,i, j, points, max_i, max_j, sim_mat)
                density2 = compute_local_density(params,i, x2y[i], points, max_i, max_j, sim_mat)
                nbDeleted += 1
                if density1 > density2:
                    if (i, x2y[i]) in points:
                        del (points[(i, x2y[i])])
                    x2y[i] = j
                else:
                    del (points[(i, j)])
                    continue
        else:
            x2y[i] = j

        if j in y2x.keys():
            if y2x[j] != i:
                # for x coordinate, conflict between (i,j) and (i,x2y[i])
                # only the best point is kept
                density1 = compute_local_density(params,i, j, points, max_i, max_j, sim_mat)
                density2 = compute_local_density(params,y2x[j], j, points, max_i, max_j, sim_mat)
                nbDeleted += 1
                if density1 < density2:
                    if (y2x[j], j) in points:
                        del (points[(y2x[j], j)])
                    y2x[j] = i
                else:
                    del (points[(i, j)])
        else:
            y2x[j] = i

    if params['verbose']:
        print(nbDeleted, "conflicting points have been removed!")

    points_key = list(points.keys())
    for point in points_key:
        (i, j) = point
        filtered_x.append(i)
        filtered_y.append(j)
    return (points, filtered_x, filtered_y)


########################################################################### ngram points functions

# ngram that contain only the same repeated character are not valid (e.g. blank spaces...)
def valid(ngram):
    return not re.match(r'(.)\1+', ngram)


# extract candidates points using ngram search
def compute_points_from_ngrams(params,sents1, sents2):
    """
    Extract anchor points using ngram density
    
    arg1 (dict): the global parameters
    arg2 (list[str]): the sentence list in lang1
    arg3 (list[str]): the sentence list in lang2
    """
    
    global min_sent_length_ratio, min_sent_length
    
    # extracting hash table that records all the ngrams for sents1
    len_sents1 = len(sents1)
    len_sents2 = len(sents2)
    sim_mat=np.array([0]*len_sents1*len_sents2)
    sim_mat.shape=len_sents1,len_sents2
    ngrams1 = []
    for i in range(len_sents1):
        ngrams1.append({})
        sent1 = sents1[i]
        for k in range(0, len(sent1) - params['ngram']):
            ngram = sent1[k:k + params['ngram']]
            if valid(ngram):
                if ngram not in ngrams1[i].keys():
                    ngrams1[i][ngram] = 0
                ngrams1[i][ngram] += 1

    # extracting hash table that records all the ngrams for sents2
    ngrams2 = []
    for j in range(len_sents2):
        sent2 = sents2[j]
        ngrams2.append({})
        for k in range(0, len(sent2) - params['ngram']):
            ngram = sent2[k:k + params['ngram']]
            if valid(ngram):
                if ngram not in ngrams2[j].keys():
                    ngrams2[j][ngram] = 0
                ngrams2[j][ngram] += 1

    # record the corresponding coordinate, sorted according to dice
    bestJ = {}
    bestI = {}

    # Using diagBeam param
    if params['diagBeam']:
        range2 = len_sents2 * params['diagBeam']
    else:
        range2 = len_sents2
    # dice computation for each point (i,j)
    for i in range(len_sents1):
        nb1 = max(1, len(sents1[i]) - params['ngram'] + 1)
        if params['verbose'] and i % 100 == 0:
            print("x =", i, "/", len_sents1)
        for J in range(range2):
            if params['diagBeam']:
                # when using fixed vertical width around diag, j must be computed as: int(i*len_sents2/len_sents1-range2/2)
                j = int(i * len_sents2 / len_sents1 - range2 / 2)
            else:
                j = J
            if j < 0:
                continue
            nb2 = max(1, len(sents2[j]) - params['ngram'] + 1)
            # length of sent1 and sent2 must be comparable
            if nb1 > min_sent_length and nb2 > min_sent_length and nb1 / nb2 >= min_sent_length_ratio and nb2 / nb1 >= min_sent_length_ratio:
                # computing the number of common ngrams (based on occurrences and not on type)
                nbCommon = 0
                for ngram in ngrams1[i].keys():
                    if ngram in ngrams2[j].keys():
                        nbCommon += min(ngrams1[i][ngram], ngrams2[j][ngram])
                dice = 2 * nbCommon / (nb1 + nb2)
                sim_mat[i,j]=dice
                # if dice is greater than the threshold, candidate point (i,j) is recorded
                if dice > params['diceThreshold']:
                    if not j in bestI.keys():
                        bestI[j] = []
                    if not i in bestJ.keys():
                        bestJ[i] = []
                    bestI[j].append((dice, i))
                    bestJ[i].append((dice, j))
    
    (points, x, y)=k_best_points(params,bestI, bestJ)
    return (points, x, y, sim_mat)

def k_best_points(params,bestI, bestJ):
    """
    Building the point list taking, for each coordinate, the k best corresponding points
    
    arg1 (dict) - params: the global params
    arg2 (dict) - bestI: a dict that associate to Y coordinate the list of (x,Y) points which get similarity greater than a threshold
    arg3 (dict) - bestJ: a dict that associate to X coordinate the list of (X,y) points which get similarity greater than a threshold
    
    Returns a tuple with the following data :
    
    points (dict[(x,y):1]): the best points
    filtered_x (list[int]): the x coordinate of the best points
    filtered_y (list[int]): the y coordinate of the best points

    """
    
    x = []
    y = []
    points = {}  # points are recorded here as keys
    for i in bestJ.keys():
        # sorting the candidate according to sim
        bestJ[i] = sorted(bestJ[i], key=lambda x: x[0], reverse=True)
        if len(bestJ[i]) > 1:
            if (bestJ[i][0][0] - bestJ[i][1][0]) < params['margin']:
                if params['verbose']:
                    print("Filtering using margin criterion : ", bestJ[i][0][0], "-", bestJ[i][1][0], "<",
                          params['margin'])
                bestJ[i] = ()
            else:
                # only the k best are recorded
                bestJ[i] = [bestJ[i][l][1] for l in range(0, min(params['kBest'], len(bestJ[i])))]

    for j in bestI.keys():
        # sorting the candidate according to dice
        bestI[j] = sorted(bestI[j], key=lambda x: x[0], reverse=True)
        if len(bestI[j]) > 1:
            if (bestI[j][0][0] - bestI[j][1][0]) < params['margin']:
                if params['verbose']:
                    print("Filtering using margin criterion : ", bestI[j][0][0], "-", bestI[j][1][0], "<",
                          params['margin'])
                bestI[j] = ()
            else:
                # only the k best are recorded
                bestI[j] = [bestI[j][l][1] for l in range(0, min(params['kBest'], len(bestI[j])))]

    for i in bestJ.keys():
        for j in bestJ[i]:
            if j in bestI and i in bestI[j]:
                x.append(i)
                y.append(j)
                points[(i, j)] = 1
    return (points, x, y)


############################################################# LABSE points functions
# Function to normalize the embeddings by dividing them with their L2-norm
def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms


def compute_points_with_encoder(params, preprocessor, encoder, sents1, sents2, embed_shelve):
    """
    Compute all the candidate points that correspond to a given similarity (cos)
    The k best points for each x and y coordinate are kept.
    If params['diagBeam'] < 1, only points not far from the diagonal are kept
    
    arg1 (dict): the global parameters
    arg2 (funct): the preprocessor function
    arg3 (funct): the encoding function
    arg4 (list[str]): the sentence list in lang1
    arg5 (list[str]): the sentence list in lang2
    
    Returns a tuple with the following data :

    points (dict[(x,y):1]): the candidate points
    x (list[int]): the x coordinates of the candidate points
    y (list[int]): the y coordinates of the candidate points
    mat (list[list]): the similarity matrix
    embeds1 (list[tensor]): the embeddings that correspond to sents1
    embeds2 (list[tensor]): the embeddings that correspond to sents2

    """

    points = {}  # points are recorded here as keys

    t0 = time.time()

    runEncoder = True
    # load from shelve in test mode (embeds are already computed)
    if params['useShelve']:
        embeds1 = []
        embeds2 = []
        runEncoder = False
        for sent in sents1:
            if sent in embed_shelve:
                embeds1.append(embed_shelve[sent])
            else:
                runEncoder = True
                break
        for sent in sents2:
            if sent in embed_shelve:
                embeds2.append(embed_shelve[sent])
            else:
                runEncoder = True
                break
    if runEncoder:
        if params['verbose']:
            print("Running Encoder...\n")

        embeds1 = compute_embeds(preprocessor, encoder, params['embedModel'], sents1, params['l1'])
        embeds2 = compute_embeds(preprocessor, encoder, params['embedModel'], sents2, params['l2'])

        t1 = time.time()
        if params['verbose']:
            print("\n1. Encoding -->", t1 - t0, "s.\n")

        # For semantic similarity tasks, apply l2 normalization to embeddings
        embeds1 = normalization(embeds1)
        embeds2 = normalization(embeds2)
        t2 = time.time()
        if params['verbose']:
            print("\n2. Normalization -->", t2 - t1, "s.\n")

    # saving normalized embeddings to shelve
    if params['useShelve'] and runEncoder:
        for i, sent in enumerate(sents1):
            embed_shelve[sent] = embeds1[i]
        for i, sent in enumerate(sents2):
            embed_shelve[sent] = embeds2[i]
        t2 = time.time()
        if params['verbose']:
            print("1-2. Loading embeddings from shelve -->", t2 - t0, "s.\n"),
    t3 = time.time()
    # similarity
    mat = np.matmul(embeds1, np.transpose(embeds2))

    t4 = time.time()
    if params['verbose']:
        print("\n3. Similarity matrix -->", t4 - t3, "s.\n"),

    # building the point list taking, for each coordinate, the k best corresponding point
    x = []
    y = []
    points = {}  # points are recorded here as keys

    # if the searchspace is reduced around the diagonal, compute the kBest point manually
    if params['diagBeam'] < 1:
        # record the corresponding coordinate
        bestJ = {}
        bestI = {}
        maxVertDistToTheDiagonal = int(len(sents2) * params['diagBeam'])
        for i in range(len(mat)):
            diagJ = int(i / len(mat) * len(mat[i]))
            infJ = max(0, diagJ - maxVertDistToTheDiagonal)
            supJ = min(diagJ + maxVertDistToTheDiagonal, len(mat[i]))
            for j in range(infJ, supJ):
                if mat[i][j] > params['cosThreshold']:
                    if i not in bestJ:
                        bestJ[i] = []
                    if j not in bestI:
                        bestI[j] = []
                    bestJ[i].append((mat[i][j], j))
                    bestI[j].append((mat[i][j], i))
        t5 = time.time()
        if params['verbose']:
            print("\n4. Extracting points -->", t5 - t4, "s.\n"),

        (points, x, y) = k_best_points(params,bestI, bestJ)
        t6 = time.time()
        if params['verbose']:
            print("\n5. Filtering k best vertically and horizontally -->", t6 - t5, "s.\n"),

    # use numpy argpartition for kBest extraction
    else:
        k = params['kBest']
        # for k=1 we extract the 2 best, in order to apply the margin criterion
        if k == 1:
            k = 2
        k = min(k, len(mat[0]))
        # using argpartition allow to extract quickly the k-best col for each line
        ind_by_line = np.argpartition(mat, -k, axis=1)[:, -k:]
        sim_by_line = np.take_along_axis(mat, ind_by_line, axis=1)
        bestJ = [list(zip(sim_by_line[i], ind_by_line[i])) for i in range(len(sim_by_line))]

        for i in range(len(bestJ)):
            bestJ[i].sort(key=lambda x: x[0], reverse=True)
            if (bestJ[i][0][0] - bestJ[i][1][0]) < params['margin']:
                if params['veryVerbose']:
                    print("Filtering using margin criterion : ", bestJ[i][0][0], "-", bestJ[i][1][0], "<",
                          params['margin'])
                bestJ[i] = []
            # once margin criterion has been applied, apply the threshold
            bestJ[i] = [pair for pair in bestJ[i] if pair[0] > params['cosThreshold']]
            # if kBest==1, crop the candidate list
            if params['kBest'] == 1 and len(bestJ[i]) > 1:
                bestJ[i] = bestJ[i][0:1]

        ind_by_col = np.argpartition(mat, -k, axis=0)[-k:, :]
        sim_by_col = np.take_along_axis(mat, ind_by_col, axis=0)
        ind_by_col = ind_by_col.swapaxes(1, 0)
        sim_by_col = sim_by_col.swapaxes(1, 0)

        bestI = [list(zip(sim_by_col[i], ind_by_col[i])) for i in range(len(ind_by_col))]
        for j in range(len(bestI)):
            bestI[j].sort(key=lambda x: x[0], reverse=True)
            if (bestI[j][0][0] - bestI[j][1][0]) < params['margin']:
                if params['veryVerbose']:
                    print("Filtering using margin criterion : ", bestI[j][0][0], "-", bestI[j][1][0], "<",
                          params['margin'])
                bestI[j] = []
            # once margin criterion has been applied, apply the threshold
            bestI[j] = [pair for pair in bestI[j] if pair[0] > params['cosThreshold']]
            # if kBest==1, crop the candidate list
            if params['kBest'] == 1 and len(bestI[j]) > 1:
                bestI[j] = bestI[j][0:1]

        # adding points
        for i in range(len(bestJ)):
            for n in range(len(bestJ[i])):
                j = bestJ[i][n][1]
                for m in range(len(bestI[j])):
                    if i == bestI[j][m][1]:
                        x.append(i)
                        y.append(j)
                        points[(i, j)] = 1
                        break

        t5 = time.time()
        if params['verbose']:
            print("\n4-5. Extracting and filtering k best vertically and horizontally -->", t5 - t4, "s.\n"),

    return (points, x, y, mat, embeds1, embeds2)


############################################################# LASER and BERT points functions

# return the normalized embeddings for a given encoder and a sentence list
def compute_embeds(preprocessor, encoder, embed_model, sents, language=""):
    if embed_model == "laser":
        # Use the Laser model to embed the sentences in different languages
        embeds = encoder.embed_sentences(sents, language)
    else:
        if preprocessor:
            embeds = encoder(preprocessor(sents))["default"]
        else:
            embeds = encoder.encode(sents)

    # Normalize the embeddings using the normalization function
    embeds = normalization(embeds)
    return embeds

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


############################################################ Main function

def extract_anchor_points(params,pre_anchors_x, pre_anchors_y, points, x, y, sents1, sents2, len_sents1, len_sents2, sim_mat):
    """
    From the complete cloud of points, extract and filter anchor points that occur in high density area
    
    arg1 (list[int]) : pre_anchors_x - the x coordinates of pre anchors (pre anchors are give in the input files)
    arg2 (list[int]) : pre_anchors_y - the y coordinates of pre anchors (pre anchors are give in the input files)
    arg3 (dict[(x,y):1] : points - a dictionnary that contains all the corresponding points, sorted upon x coordinate
    arg4 (list[int]) : x - the list of x coordinates for points
    arg5 (list[int]) : y - the list of y coordinates for points
    arg6 (list[str]) : sents1 - the list of segments for lang1
    arg7 (list[str]) : sents2 - the list of segments for lang2
    arg8 (list[list]) : the similarity matrix of dim (len_sents1,len_sents2)
    
    Return value : a tuple with the following data
    filtered_x (list[int]) : the filtered list of  x coordinates for points
    filtered_y (list[int]) : the filtered list of  y coordinates for points
    intervals (list[([x1,x2],[y1,y2])]) : a list of aligned intervals.
    interval_length_sent1 : the complete length of x intervals in number of sentences
    interval_length_sent2 : the complete length of y intervals in number of sentences
    interval_length_char1 : the complete length of x intervals in number of chars
    interval_length_char2 : the complete length of y intervals in number of chars
   
    """
    
    global show_plot_4_new_interval # for debugging purpose
    
    anchor_points = dict.copy(points)

    # =====> STEP 6 : compute average local density around selected points
    t5 = time.time()

    points_key = list(anchor_points.keys())

    if len(points_key) == 0:
        print("No anchor points !!!")
        beginInt = (-1, -1)
        lastI = len_sents1 - 1
        lastJ = len_sents2 - 1
        interval_length_sent1 = lastI - beginInt[0] + 1
        interval_length_sent2 = lastJ - beginInt[1] + 1
        for n in range(0, lastI + 1):
            interval_length_char1 += len(sents1[n])
        for n in range(0, lastJ + 1):
            interval_length_char2 += len(sents2[n])

    else:
        tot_density = 0
        for point in points_key:
            (x2, y2) = point
            tot_density += compute_local_density(params,x2, y2, anchor_points, len_sents1, len_sents2, sim_mat)

        average_density = tot_density / float(len(points_key))

        t6 = time.time()
        if params['verbose']:
            print("\n6. Computing average density-->", t6 - t5, "s.\n"),

        # =====> STEP 7 : filtering out low density points

        (anchor_points, filtered_x, filtered_y) = filter_points(params,pre_anchors_x, anchor_points, len_sents1, len_sents2, 
                                                                average_density, sim_mat)
        (anchor_points, filtered_x, filtered_y) = resolving_conflicts(params,anchor_points, len_sents1, len_sents2, sim_mat)

        if params['reiterateFiltering']:
            (anchor_points, filtered_x, filtered_y) = filter_points(params,pre_anchors_x, anchor_points, 
                                                                    len_sents1, len_sents2,
                                                                    average_density * 2, sim_mat)

        t7 = time.time()
        if params['verbose']:
            print("\n7. Removing low density points-->", t7 - t6, "s.\n"),

        # ~ x=[point[0] for point in points]
        # ~ y=[point[1] for point in points]
        # ~ plt.axis([1,len_sents1,1,len_sents2])
        # ~ plt.title(output_file_name+'.txt - filtered')
        # ~ plt.scatter(x,y,c="red",s=1)
        # ~ plt.show()

        # =====> STEP 8 : finding aligned intervals

        beginInt = (-1, -1)
        # adding last point as an anchor
        filtered_x.append(len_sents1 - 1)
        filtered_y.append(len_sents2 - 1)
        lastI = 0
        lastJ = 0
        intervals = []  # the array of pairs (beginInt,endInt) where beginInt and endInd are two points that define the interval
        nb_in_interval = 0

        (interval_length_sent1, interval_length_sent2, interval_length_char1, interval_length_char2) = (0, 0, 0, 0)

        # adding pre_anchors as intervals to drive alignment
        if len(pre_anchors_x)>0:
            for x,y in zip(pre_anchors_x,pre_anchors_y):
                endInt=(x,y)
                if params['verbose']:
                    print ("Adding interval according to pre anchors ",(beginInt,endInt))
                intervals.append((beginInt,endInt))
                interval_length_sent1 += endInt[0] - beginInt[0] + 1
                interval_length_sent2 += endInt[1] - beginInt[1] + 1
                beginInt=endInt
            lastI = len_sents1 - 1
            lastJ = len_sents2 - 1
        elif params['detectIntervals']:
            coeff = 1 if params['sentRatio'] == 0 else params['sentRatio']
            for num in range(0, len(filtered_x)):
                (i, j) = (filtered_x[num], filtered_y[num])
                
                localDensity = compute_local_density(params,i, j, anchor_points, len_sents1, len_sents2, sim_mat)
                density_ratio = 0
                if average_density > 0:
                    density_ratio = localDensity / average_density
                # computation of the distance between (i,j) and (i,expected(j))
                expectedJ = lastJ + (i - lastI) * coeff
                vertical_deviation = abs(j - expectedJ)
                new_interval = False
                print(f"vertical_deviation={vertical_deviation}")

                # monotony constraint : if the two previous and the two next anchors are monotonic but not the current
                # the current anchor is discarded
                if num > 1 and num < len(filtered_x) - 2:
                    if (filtered_x[num - 2] <= filtered_x[num - 1] <= filtered_x[num + 1] <= filtered_x[num + 2]) and \
                            (filtered_y[num - 2] <= filtered_y[num - 1] <= filtered_y[num + 1] <= filtered_y[
                                num + 2]) and \
                            (not (filtered_x[num - 1] <= i <= filtered_x[num + 1]) or \
                             not (filtered_y[num - 1] <= j <= filtered_y[num + 1])):
                        print(f"({i},{j}) is ignored (non monotonic)")
                        # the current point is skipped
                        filtered_x[num] = lastI
                        filtered_y[num] = lastJ
                        continue

                # deviated and low density point
                if (vertical_deviation > params[
                    'maxDistToTheDiagonal'] / 2 or i < lastI or j < lastJ) and density_ratio < params[
                    'minDensityRatio']:
                    
                    # deviated point is removed if density is not high enough
                    print(f"({i},{j}) is ignored. Low density : density_ratio={density_ratio}")
                    # the current point is skipped
                    filtered_x[num] = lastI
                    filtered_y[num] = lastJ
                    continue

                # only the points that are near the diagonal are taken into account
                if vertical_deviation <= params['maxDistToTheDiagonal']:
                    nb_in_interval += 1
                    lastI = i
                    lastJ = j
                    print(f"({i},{j}) is valid\n")
                else:
                    params['verbose'] and print(
                        f"({i},{j}) is a deviating point lastI={lastI}, lastJ={lastJ}, density_ratio={density_ratio}, vertical_deviation={vertical_deviation}")

                    # considering next points to compute next deviation
                    preview_scope = 2
                    if num + preview_scope < len(filtered_x):
                        (next_i, next_j) = (filtered_x[num + preview_scope], filtered_y[num + preview_scope])
                        next_expectedJ = lastJ + (next_i - lastI) * params['sentRatio']
                        next_vertical_deviation = abs(next_j - next_expectedJ)
                        # the next point is aligned with previous point
                        if next_vertical_deviation <= params['maxDistToTheDiagonal']:
                            params['verbose'] and print(
                                f"({i},{j}) is ignored (next point is aligned with the previous). vertical_deviation={vertical_deviation}")
                            # the current point is skipped
                            filtered_x[num] = lastI
                            filtered_y[num] = lastJ
                            continue
                        else:
                            next_expectedJ = j + (next_i - i) * params['sentRatio']
                            next_vertical_deviation = abs(next_j - next_expectedJ)
                            # if the next point is aligned with the current point, then a new interval should be created
                            if next_vertical_deviation <= params['maxDistToTheDiagonal'] and density_ratio > params[
                                'minDensityRatio']:
                                params['verbose'] and print(
                                    f"({i},{j}) is kept for a new interval because aligned with next points")
                                new_interval = True
                            else:
                                params['verbose'] and print(
                                    f"({i},{j}) is ignored (next point is not aligned) next_vertical_deviation={next_vertical_deviation} density_ratio={density_ratio}")
                                # the current point is skipped
                                filtered_x[num] = lastI
                                filtered_y[num] = lastJ
                                continue
                    # if the deviating point has a high density then create a new interval
                    # ~ # a new interval must be created from the deviating point
                    # ~ if density_ratio > 1.5:
                    # ~ params['verbose'] and print(f"({i},{j}) is kept for a new interval because of high density",density_ratio)
                    # ~ new_interval=True
                    # ~ else:
                    # ~ params['verbose'] and print(f"({i},{j}) is ignored. {density_ratio=}")
                    # ~ # the current point is skipped
                    # ~ filtered_x[num]=lastI
                    # ~ filtered_y[num]=lastJ
                    # ~ continue

                # ~ # computing distance
                d = math.sqrt((i - lastI) ** 2 + (j - lastJ) ** 2)
                # if a there is a gap the previous interval is closed and a new interval will begin
                if d > params['maxGapSize'] and density_ratio > 1.5:
                    params['verbose'] and print(f"{d} > maxGapSize, density_ratio={density_ratio}")
                    new_interval = True

                # Creating a new interval if necessary
                if new_interval:
                    endInt = (lastI, lastJ)
                    params['verbose'] and print(d, f"Closing interval ({beginInt},{endInt}) for point ({i},{j})")
                    if beginInt[0] < lastI and beginInt[1] < lastJ:
                        # to save the interval, we compute the density of selected points according to the horizontal width
                        if nb_in_interval / (lastI - beginInt[0]) >= params[
                            'minHorizontalDensity'] and nb_in_interval > 1:
                            intervals.append((beginInt, endInt))
                            interval_length_sent1 += lastI - beginInt[0] + 1
                            interval_length_sent2 += lastJ - beginInt[1] + 1
                            for n in range(max(0, beginInt[0]), lastI + 1):
                                interval_length_char1 += len(sents1[n])
                            for n in range(max(0, beginInt[1]), lastJ + 1):
                                interval_length_char2 += len(sents2[n])
                        else:
                            if params['verbose']:
                                print("Interval", beginInt, endInt, "has been discarded (density too low)")
                    beginInt = (i, j)
                    nb_in_interval = 0

                    if show_plot_4_new_interval:
                        min_x = max(0, i - 100)
                        max_x = min(len(sents1) - 1, i + 100)
                        min_y = max(0, j - 100)
                        max_y = min(len(sents2) - 1, j + 100)

                        x = [point[0] for point in anchor_points if
                             min_x <= point[0] <= max_x and min_y <= point[1] <= max_y]
                        y = [point[1] for point in anchor_points if
                             min_x <= point[0] <= max_x and min_y <= point[1] <= max_y]
                        plt.axis([min_x, max_x, min_y, max_y])
                        plt.title(str(i) + "," + str(j) + '=> new interval')
                        plt.scatter(x, y, c="black", s=1)
                        (i1, j1) = (i - params['deltaX'] / 2, j - params['deltaX'] / 2 - params['deltaY'] / 2)
                        (i1, j2) = (i - params['deltaX'] / 2, j - params['deltaX'] / 2 + params['deltaY'] / 2)
                        (i2, j3) = (i + params['deltaX'] / 2, j + params['deltaX'] / 2 + params['deltaY'] / 2)
                        (i2, j4) = (i + params['deltaX'] / 2, j + params['deltaX'] / 2 - params['deltaY'] / 2)
                        X = [i1, i1, i2, i2, i1]
                        Y = [j1, j2, j3, j4, j1]
                        plt.plot(X, Y, c="grey")
                        plt.show()

                    # la mise à jour de lastI et lastJ ne se fait pas pour
                    # un point déviant n'ayant pas ouvert un intervalle
                    lastI = i
                    lastJ = j
        else:
            lastI = len_sents1 - 1
            lastJ = len_sents2 - 1

        t8 = time.time()
        if params['verbose']:
            print("\n8. Extracting alignable intervals-->", t8 - t7, "s.\n"),

    if lastI != beginInt[0]:
        # closing last interval
        interval_length_sent1 += lastI - beginInt[0] + 1
        interval_length_sent2 += lastJ - beginInt[1] + 1
        for n in range(max(0, beginInt[0]), lastI + 1):
            interval_length_char1 += len(sents1[n])
        for n in range(max(0, beginInt[1]), lastJ + 1):
            interval_length_char2 += len(sents2[n])
        intervals.append((beginInt, (lastI, lastJ)))
        params['verbose'] and print( f"Closing last interval ({beginInt},({lastI},{lastJ}))")

    if params['verbose']:
        print("Total interval length=", interval_length_sent1, "+", interval_length_sent2)
        
    # last filtering step : for each interval, points that are two far from the diagonal are discarded  
    i=0
    for (begin,end) in intervals:
        (x_begin,y_begin)=begin
        (x_end,y_end)=end
        if (x_end-x_begin)*(y_end-y_begin)==0:
            continue
        # looking for anchor points in interval begin, end
        while i<len(filtered_x) and filtered_x[i] < x_begin:
            i+=1
        # if point i falls in x interval
        while i<len(filtered_x) and filtered_x[i]>=x_begin and filtered_x[i]<=x_end:
            delete=False
            #  if point i does not fall in y interval, delete point
            if filtered_y[i]<y_begin or filtered_y[i]>y_end:
                delete=True
            expected_y=y_begin+(filtered_x[i]-x_begin)/(x_end-x_begin)*(y_end-y_begin) 
            # if point i is two far from diag, delete point
            if abs((filtered_y[i]-expected_y)/(y_end-y_begin)) > params['localDiagBeam'] or abs(filtered_y[i]-expected_y) > params['maxDistToTheDiagonal']:
                delete=True
                
            if delete :
                if params['veryVerbose']:
                    print(f"point {i} ({filtered_x[i]},{filtered_y[i]}) too far from diagonal")
                del(filtered_x[i])
                del(filtered_y[i])
                if i>=len(filtered_x):
                    break
            else:
                i+=1
        
    return (filtered_x, filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1,
            interval_length_char2)


