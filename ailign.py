# -*- coding:utf8 -*-
"""

USAGE :

1/ aligning 2 files FILE1 and FILE2 :

python3 ailign.py [--inputFormat INPUTFORMAT] --inputFile1 FILE1 --inputFile2 FILE2 --outputFileName outputFileName --outputFormats FORMATS

Examples :
python3 ailign.py --inputFormat json --inputFile1 4.\ stanza/KHM53.1819.grimm.de.json --inputFile2 4.\ stanza/KHM53.1869.alsleben.fr.json --outputFileName KHM53.alsleben.de-fr.txt --outputFormats txt ces
python3 ailign.py --inputFile1 2.\ txt/KHM53.1846.martin.fr.txt --inputFile2 2.\ txt/KHM53.1869.alsleben.fr.txt --outputFileName 5.\ aligned/KHM.1846-1869.fr-fr --outputFormats tmx txt  --savePlot --verbose
python3 ailign.py --inputFile1 corpus_aristophane/Plutus.Fleury.fr.txt --inputFile2 corpus_aristophane/Plutus.Fallex.fr.txt --outputFileName corpus_aristophane_aligné/Plutus.Fallex-Fleury.fr-fr --outputFormats tmx txt  --savePlot --verbose --margin 0.01 --cosThreshold 0.5 --k 2 --deltaX 20 --minDensityRatio 1.1


NB :
- outputFileName is the file name without the extension. The extension will be added according to the format
- FORMATS may contain more than one format ex. "ces txt"


2/ aligning the files that matches PATTERN in INPUTDIR and write the output in OUTPUTDIR in the formats FORMATS :

python3 ailign.py --filePattern PATTERN --inputDir INPUTDIR --outputDir OUTPUTDIR --l1 LANG1 --l2 LANG2 --outputFormat FORMATS

Options :
  --writeAnchorPoints : to write the anchor points (incomplete but very reliable 1-1 alignement)
  --runDTW : to extract the complete alignment with 0-1, 1-0, 1-1, 1-2, 2-1, 1-3, 3-1, 2-2 grouping
                this extraction may be far longer !

How anchor points are filtered :

First of all, candidate points are created when a similarity threshold greater than cosThreshold (typically 0.4 or 0.5) is obtained between a sentence x and a sentence y. Then, for each column or row, only the k points with the highest scores are retained (the kBest parameter is set to a default value of 4).
Then, for each column or row, only the k points with the highest scores are retained (kBest parameter set to 4 by default).
At this stage, filtering is performed using the margin parameter, which allows us to retain only those points with a score greater than margin compared with their best competitor (margin=0.05 by default). If we apply this criterion, it may be consistent to set kBest to 1.

We then apply a two-stage high-pass filter.
The first filtering corresponds to the filter_points() function. The first filter is based on a calculation of the density of candidates around each candidate point. This density is not calculated in a square centered around the point, but rather in a corridor centered on the diagonal passing through the point (the alignment path generally follows this diagonal). The width of this corridor corresponds to the deltaY parameter. The length of this corridor corresponds to the deltaY parameter. The number of candidate points divided by the size of this space gives a density value. If this density, divided by the average density of all candidate points, is greater than a certain ratio (minDensityRatio, typically around 0.5) then the point is retained.
The second filter, which corresponds to the resolving_conflicts() function, focuses on resolving conflicts on the vertical and horizontal axes respectively - when for the same x-coordinate there are several points with different y-coordinates, and conversely, when for the same y-coordinate there are several points with different x-coordinates - these cases only arise if KBest is greater than 1. Competitors are eliminated on the basis of density: only the point with the best density along its diagonal is retained.
This density filtering can be repeated once if the --reiterateFiltering parameter is given.

"""

import os
import re
import sys
import argparse
import time

from datetime import datetime

# local module
from align import align, load_sentence_encoder
from default_params import default_params


# reading the command line arguments
parser = argparse.ArgumentParser(
    prog='ailign',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''\
A program that align sentences for parallel text pairs. 

Input : 
* --inputFile1 and --inputFile2 indicate a parallel text pair
* for multiple files aligning, when using --inputDir param, the corresponding files 
must be named following this pattern : NAME..*.L1.txt NAME..*.L2.txt
* alternatively, text pairs may be listed in a file with --inputFileList param
* Recognized format are TXT, ARC, JSON, TSV, XML-CONLL, XML
* In TXT format, files should be formatted as raw utf8 text with one sentence per line.

Output :
The output will yield a new file (or a new pair of files) that contain a selection of 
sentences in TXT, CES, TSV, BERTALIGN or ARC format, in order to be easily aligned
in a next step (the non parallel text is supposed to be removed).
'''
)

# main arguments for input / output
parser.add_argument('--l1', type=str, help='The source language (ISO : ex. "en" for English)')
parser.add_argument('--l2', type=str, help='The target language (ISO : ex. "fr" for French, "*" for any)')
parser.add_argument('-i', '--inputFormat', help='Format of the input (txt, arc, ces, json, tsv, xml-conll, xml)')
parser.add_argument('--xmlGuide', nargs='+', type=str, help='List of markups that should be read in the XML input')
parser.add_argument('--anchorTag',type=str, help='Tag that defines prealigned anchors in XML input (eg. "anchor" or "p")')
           
parser.add_argument('--col1', help='For TSV format, indicate the column of l1', type=int)
parser.add_argument('--col2', help='For TSV format, indicate the column of l2', type=int)
parser.add_argument('-o', '--outputFormats', nargs='+', type=str,
                    help='Formats of the output (TXT, TXT2, CES, ARC, XML, TSV, TSV2, BERTALIGN)')
parser.add_argument('--collectionName', help='for TSV2 format (Lexicoscope) name of the collection')
parser.add_argument('--alreadyAligned', help='for TXT or TSV format with two files aligned line by line', action="store_true")
parser.add_argument('--alignedFileName', type=str, help='to import sentence alignments from a TSV file')
parser.add_argument('--addAnchor', help='Add anchor in xml files', action="store_true")
parser.add_argument('--direction', type=str, help='The aligning direction for anchors: "1<->2","1->2","2->1"')
parser.add_argument('--inputFile1', type=str, help='The l1 input file to process')
parser.add_argument('--inputFile2', type=str, help='The l2 input file to process')
parser.add_argument('--fileId1', type=str, help='The id prefix of file1 in xml anchors')
parser.add_argument('--fileId2', type=str, help='The id prefix of file2 in xml anchors')
parser.add_argument('--inputFileList', type=str, help='A tsv file with corresponding filenames separated by tab')
parser.add_argument('--inputDir', type=str, help='The directory to process')
parser.add_argument('--outputDir', type=str, help='The directory to save output files')
parser.add_argument('--outputFileName', type=str, help='The output filename (optional), without format extension')
parser.add_argument('-f', '--filePattern', type=str,
                    help='The pattern of the files that should be processed. A capturing group such as (.*) should capture the common prefix between aligned files.')
parser.add_argument('--writeAnchorPoints', help='Write anchor points', action="store_true")
parser.add_argument('--writeSegmentedInput', help='Write sentence segmented input files in txt format',
                    action="store_true")
parser.add_argument('--writeIntervals', help='Write aligned intervals (as corresponding sentence numbers)',
                    action="store_true")
parser.add_argument('--printIds', help='Print IDs in txt output', action="store_true")
parser.add_argument('--splitSent1', help='Split the txt segments into sentences for l1', action="store_true")
parser.add_argument('--splitSent2', help='Split the txt segments into sentences for l2', action="store_true")
parser.add_argument('--splitSentRegex', type=str, help='Regex to split sentences')

parser.add_argument('--useSentenceSegmenter',
                    help='Use the trankit sentence segmenter for txt input (instead of regex segmenter)',
                    action="store_true")
parser.add_argument('--mergeLines', help='Merge lines until a line ends with a separator for txt input',
                    action="store_true")
parser.add_argument('--adaptativeMode',
                    help='Using interval detection, compute estimated sentRatio and charRatio, and reiterate filtering.',
                    action="store_true")

# special arguments for output control
parser.add_argument('-v', '--verbose', help='Verbose messages', action="store_true")
parser.add_argument('-w', '--writeAlignableArea', help='Write alignable area files', action="store_true")
parser.add_argument('-V', '--veryVerbose', help='Very verbose messages', action="store_true")
parser.add_argument('--savePlot', help='Save scatter plot in a png file', action="store_true")
parser.add_argument('--showPlot', help='Show scatter plot (with a pause during execution)', action="store_true")
parser.add_argument('--showSimMat', help='Show heat map for similarity matrix', action="store_true")

# controlling stage 1 and 2
parser.add_argument('--detectIntervals', help='Detect alignable interval using anchor points.', action="store_true")
parser.add_argument('-u', '--useNgrams', help='Use ngrams to extract points', action="store_true")
parser.add_argument('-r', '--doNotRunDTW', help='Perform only first step without DTW algorithm)', action="store_true")
parser.add_argument('--lateGrouping',
                    help='Run DTW algorithm with only 1-1 pairing, then, group the contiguous points with lateGrouping method (greedy algorithm)',
                    action="store_true")
parser.add_argument('--noMarginPenalty',
                    help='Do not compute the similarity with neighbouring sentences, and substract the neighbouring similarity to the bead similarity)',
                    action="store_true")

# controlling anchor points building and filtering
# (important parameters are : cosThreshold, kBest, deltaX, minDensityRatio)
parser.add_argument('--embedModel', type=str, help='Choose embedding model : sbert or laser or labse-keras or stsb-xlm-r-multilingual')
parser.add_argument('--modelName', type=str, help='Choose sbert model name (default=sentence-transformers/LaBSE)')
parser.add_argument('-l', '--cosThreshold', type=float,
                    help='The minimum similarity for labse vectors to yield one point')
parser.add_argument('--cosThresholdInOutputAnchors', type=float, help='The minimum similarity for final anchor points')
parser.add_argument('--ngram', type=int, help='The ngram size')
parser.add_argument('-d', '--diceThreshold', type=float, help='The minimum dice score to yield a candidate point')
parser.add_argument('--margin', type=float,
                    help='Margin used to eliminate sentences that have too close neighbours on the vertical or horizontal axis')
parser.add_argument('-k', '--kBest', type=int,
                    help='Number of the best coordinates for each line ore column to keep when creating points')
parser.add_argument('-x', '--deltaX', type=int, help='Local space definition : +/-delta X on horizontal axis')
parser.add_argument('-y', '--deltaY', type=int, help='Local space definition : +/-delta Y on vertical axis')
parser.add_argument('-H', '--minHorizontalDensity', type=float,
                    help='The minimal horizontal density in a interval to be kept in the final result')
parser.add_argument('-m', '--maxDistToTheDiagonal', type=int,
                    help='The maximal distance to the diagonal (inside a given interval) for a point to be taken into account in the horizontal density')
parser.add_argument('-D', '--minDensityRatio', type=float,
                    help='The minimal local density ratio (reported to the average local density) to keep a candidate point')
parser.add_argument('-g', '--maxGapSize', type=int,
                    help='The maximal distance between to consecutive points in the same interval')
parser.add_argument('--diagBeam', type=float,
                    help='A real number in the range 0-1 which indicate the max distance of anchor points to the diagonal (vertically), in proportion (1 indicates that the whole search space is used')
parser.add_argument('--localDiagBeam', type=float,
                    help='A real number in the range 0-1 which indicate the max distance of anchor points to the diagonal of each alignable interval (vertically), in proportion (1 indicates that the whole search space is used')
parser.add_argument('--sentRatio', type=float,
                    help='The sentence ratio is used during anchor point filtering. Normally computed automatically, may be forced when texts have very different length.')
parser.add_argument('--charRatio', type=float,
                    help='The character ratio is used during final aligning when groups of sentences are paired. Normally computed automatically, may be forced when texts have very different length.')
parser.add_argument('--reiterateFiltering', help='Filter the anchor points according to density twice',
                    action="store_true")

# controlling DTW algorithm
parser.add_argument('--dtwBeam', help='Max dist to the anchor point in DTW algorithm', type=int)
parser.add_argument('--localBeamDecay', help='Decreasing value of localBeam at each recursion step', type=float)
parser.add_argument('--distNull', help='Default distance for null correspondance', type=float)
parser.add_argument('--noEmptyPair', help='No 1-0 or 0-1 pairing', action="store_true")
parser.add_argument('--no2_2Group', help='No 2-2 pairing', action="store_true")
parser.add_argument('--penalty_n_n', help='Penalty score given for each n-n grouping', type=float)
parser.add_argument('--penalty_0_n',
                    help='Penalty score given for each 0-n (or n-0) grouping (only used in lateGrouping)', type=float)

parser.add_argument('--wordAlignment', help='Run the word alignment script', action="store_true")
parser.add_argument('--chunkAlignment', help='Run the chunk alignment script', action="store_true")

# other : persistance of embeddings
parser.add_argument('--useShelve', help='Save the embeddings in shelve (in order to quick up the next run)',
                    action="store_true")
parser.add_argument('--useGPU', help='Use GPU (otherwise CPU)',
                    action="store_true")



args = vars(parser.parse_args())

# global parameters
# reading default parameters in the params dict
params = default_params()

# add command line argument in params
for param,value in params.items():
    if args.get(param):
        params[param]=args[param]
        print("Reading parameter ",param,"=>",args[param])

# compile the regex
params['filePattern']=re.compile(params['filePattern'])

# if no output dir, we take the path of outputFileName
if params['outputDir'] == "":
    params['outputDir'] = os.path.split(params['outputFileName'])[0]
    print("Setting outputDir to parameter ",params['outputDir'])


# Fixed params
params['printLog'] = True # Print execution log in log file
params['useEncoder'] = False # Use encoder for concatened sentences in groups 
                              # (if false, single sentence embeddings are summed)
params['printGap'] = False # Print empty points between intervals
params['matchFirstPreAnchors'] = True # when True, if the numbers of preanchors are not equal
                                         # only the first corresponding preanchors will be used
                                         # When false, no preanchor will be used

# ************************************************************************* MAIN
if __name__ == "__main__":
    t0 = time.monotonic()
    
    (preprocessor,encoder)=load_sentence_encoder(params)
    
    # opening log file if required
    if params['printLog']:
        log = open(os.path.join(params['outputDir'], "ailign.log"), mode="a", encoding="utf8")
        now = datetime.now()
        # Formater la date et l'heure
        formatted_date = now.strftime("%d-%m-%Y, %H:%M:%S")
        log.write("\n"+formatted_date+"\nExecution of : "+" ".join(sys.argv)+"\n")
        params['logHandle']=log

    # processing a simple pair of files
    if params['inputFile1'] and params['inputFile2']:
        align(params,preprocessor,encoder)

    # processing a full directory
    else:
        if params['verbose']:
            print("Processing directory", params['input_dir'])
        # reading a tsv file with pairs fileName1 tab fileName2
        
        # processing files according to inputFileList
        if params['inputFileList']:
            f = open(params['inputFileList'], encoding="utf8")
            files1 = []
            files2 = []
            for line in f:
                # skipping comments
                if line[0] != "#":
                    values = line.strip().split("\t")

                    if len(values) == 2:
                        f1 = values[0]
                        f2 = values[1]
                        files1.append(f1)
                        files2.append(f2)
            f.close()
            if params['verbose']:
                print("Files to process", list(zip(files1, files2)))
            for file1, file2 in zip(files1, files2):
                params['inputFile1']=file1
                params['inputFile2']=file2
                output_file_name = ""
                if params['filePattern'].match(file1):
                    output_file_name = params['filePattern'].match(file1).group(1)
                    params['l1'] = params['filePattern'].match(file1).group(2)
                    params['l2'] = params['filePattern'].match(file2).group(2)
                align(params,preprocessor,encoder)
        else:
            # processing files according to filePattern, l1 and l2, in input_dir
            files = [f for f in os.listdir(params['input_dir']) if
                     params['filePattern'].match(f)]  # and re.search(input_format+"$",f,re.I)]
            files1 = [f for f in files if params['filePattern'].match(f).group(2) == params['l1']]
            files2 = [f for f in files if params['filePattern'].match(f).group(2) != params['l1'] and (
                        params['filePattern'].match(f).group(2) == params['l2'] or  params['l2'] == "*")]
            if params['verbose']:
                print("Files to process", files1)
            # processing input files
            for file1 in files1:
                params['inputFile1']=file1
                m = params['filePattern'].match(file1)
                
                name = m.group(1)
                for file2 in files2:
                    m = params['filePattern'].match(file2)
                    if m.group(1) == name:
                        params['l2']=m.group(2)
                        params['outputFileName']=""
                        align(params,preprocessor,encoder)
    if params['verbose']:
        print("Terminated in", time.monotonic() - t0, "s.")

    if params['printLog']:
        log.close()
