# -*- coding:utf8 -*-
"""

This module defines various functions to read input files and write aligned files.
Input formats : txt, xml, tsv
Output formats : txt, xml, tsv, ces, arc, tmx

Main functions :

    read_input_file(params, input_file, column=0, language="fr")
    
    write_alignable(sents, id_sents, intervals, index, output_dir, output_file, output_format)
    
    write_aligned_points(params, sents1, id_sents1, sents2, id_sents2, filtered_x, filtered_y, output_dir, output_file, output_format, anchor, print_ids=False, mean_score=0, file1="", file2="")

    add_anchor_in_output(input_dir, input_file1, input_file2, file_id1, file_id2, x, y, output_dir, direction)

"""


import os
import re
import sys
import xml.etree.ElementTree as ET
from lxml import etree
import csv

#######################################
# global parameters
seg_min_length = 5  # min length for an aligned segment (in order to avoid oversegmentation)
xml_id_offset = 0

# Rules that define a correct end of line, for line merging
merge_lines_regex = {
    'zh': r'[：，。？！”]\s*$',
    'fr': r'[?;:\.!"»…]\s*$',
    'ar': r'(\.|۔)\s*$',
}

########################################

def header(params,output_format):
    """
    returns the corresponding xml header
    
    """
    # arc format is adapted to yasa input
    if output_format=="arc":
        return """
        <text>
            <divid='d1'>
                <pid='d1p1'>\n"
                """

    # ces format is another standard for segmented files
    if output_format=="ces":
        return """
    <?xml version="1.0" encoding="utf-8"?>
    <cesAna>
        <chunkList>
            <chunk>
                <par>
        """
    
    # cesalign format is used to store alignment result
    if output_format=="ces_align":
        return f"""
    <?xml version="1.0" encoding="utf-8"?>
    <cesAlign type="seg" version="1.6">
        <cesHeader version="2.3" mean_score="__mean_score__">
            <translations>
                <translation lang="{params['l1']}" />
                <translation lang="{params['l2']}" />
            </translations>
        </cesHeader>

        <linkList>
            <linkGrp targType="seg">
                        """ 
   
    # tmx is a common xml format to encode aligned file (for translation memories)
    if output_format=="tmx":
        return f"""
    <?xml version="1.0" encoding="utf-8" ?>
    <!DOCTYPE tmx SYSTEM "tmx14.dtd">
    <tmx version="1.4">
      <header
        creationtool="AIlign"
        creationtoolversion="1.0"
        datatype="unknown"
        segtype="sentence"
        mean_score="__mean_score__"
      >
      </header>
      <body>
    """


def footer(output_format):

    # arc format is adapted to yasa input
    if output_format=="arc":
        return """
            </p>
        </div>
    </text>
               """
        
    if output_format=="ces":
        return """
                </par>
            </chunk>
        </chunkList>
    </cesAna>
                """
  
    if output_format=="ces_align":
        return """
            </linkGrp>
        </linkList>
    </cesAlign>
                """
    
    if output_format=="tmx":
        return """
      </body>
    </tmx>  
                """
    
######################################################################### Functions
# process basic xml entities
def toXML(s):
    s = re.sub(r'&', '&amp;', s);
    s = re.sub(r'<', '&lt;', s);
    s = re.sub(r'>', '&gt;', s);
    return s

######################################################################### reading / writing files
# reading input file

def read_input_file(params, input_file, split_sent, column=0, language="fr"):
    """Reads an input file and returns a list of sentences.

      Args:
        input_dir: The directory containing the input file.
        input_file: The name of the input file.
        input_format: The format of the input file.
        column: The column number of the input file that contains the text.
        language: The language of the input file.

      Returns:
        sents: a list of sentences.
        id_sents: the list of sentence ids (build upon segment ids)
        len_sents: the number of sentences
        seg2sents: a list of list of integer, that gives the 1-n correspondence
            between an original segment number and the list of final sentences
            - if split_sent, for one segment, we may have more than one sentences
            - if mergeSent, more than one segment may correspond to the same merged sentence
        nb_chars: the full number of characters that corresponds to the sentences
        pre_anchors: the list of sent number that corresponds to pre_anchors
        xml_root: the ET root element for xml formats
    """
    global seg_min_length, merge_lines_regex
    
    input_format=params['inputFormat'] 
    input_dir=params['inputDir']
    l1=params['l1']
    l2=params['l2']
    segmenter=None

    # parameter for sentence segmentation
    if split_sent:
        if params['useSentenceSegmenter']:
            from trankit import Pipeline

            if params['verbose']:
                print("Loading sentence segmenter from trankit")
            # names are defined here : https://trankit.readthedocs.io/en/latest/pkgnames.html
            names = {
                "en": "english",
                "de": "german",
                "es": "spanish",
                "fr": "french",
                "zh": "chinese",
                "ar": "arabic",
                "it": "italian"
            }
            try:
                segmenter = Pipeline(names[l1])
                segmenter.add(names[l2])
            except:
                print(
                    f"Error while loading sentence segmenter from trankit. Check that you have defined a name for languages {l1} and {l2} (line 240)")
        else:
            # Rules that define sentence segmentation
            split_sent_regex = {
                'zh': r'(?<=[：，。？！”])',
                'ar': r'(?<=\.|۔)',
                'fr': r'(?<=[.!?;:])\s+(?=[A-Z«"])|(?<=[!?;:])',  # grimm Baudry
                'de': r'(?<=[.!?;:’“]) (?=[A-Z«"„])|(?<=[!?;:])|(?=[‘“])',  # grimm KHM 1857
                'grc': r'(?<=[?;:.!"»…])\s',
                'la' : r'(?<=[?;:.!"»…])\s',
                'default': r'(?<=[?;:.!"»…]) (?=[A-Z])',
            }

            if split_sent and l1 not in split_sent_regex and not params["splitSentRegex"]:
                params['verbose'] and print(f"Default regex ", split_sent_regex["default"],
                                            f"will be used for sentence segmentation in {l1}")
                split_sent_regex[l1] = split_sent_regex['default']
            if split_sent and l2 not in split_sent_regex and not params["splitSentRegex"]:
                params['verbose'] and print(f"Default regex ", split_sent_regex["default"],
                                            f"will be used for sentence segmentation in {l2}")
                split_sent_regex[l2] = split_sent_regex['default']

    segs = []
    id_segs = []
    len_sents = 0
    seg2sents = []
    nb_chars = 0
    pre_anchors = [] # in XML format, it is possible to define anchors of prealignment
    xml_root = None

    try:
        input_file_path = os.path.join(input_dir, input_file) if input_dir else input_file
        f = open(input_file_path, encoding='utf8')
    except Exception as e :
        print("Error: a problem occurred while opening", input_file, e)
        sys.exit()

    # Reading according to input_format
    if input_format == "arc" or input_format == "ces":
        for line in f:
            line = line.strip()
            m = re.search(r'<s\b[^>]+id="([^"]*)"', line)
            if m and m.group(1):
                id_segs.append(m.group(1))
            else:
                id_segs.append("s" + str(lenSegs))
            segs.append(line)
            lenSegs += 1
            nb_chars += len(line)


    # The json input contains a sentences property, which is a list sentences, which are list of tokens
    # Each token is a list of conll features, col1->form, col9=blank_space
    elif input_format == "json":
        content = f.read()
        jsonObj = json.loads(content)
        segs = [
            "".join([tok[1] + tok[9] for tok in sent if len(tok) >= 10]) for sent in jsonObj["sentences"]
        ]
        for seg in segs:
            nb_chars += len(seg)
        id_segs = [str(i) for i in list(range(0, len(segs) + 1))]

    # the tsv format is an already aligned format. Sentence are extracted from a specific column
    elif input_format == "tsv":
        segs = []
        for i,line in enumerate(f):
            alignedSegs = re.split("\t", line)
            segs.append(alignedSegs[column])
            nb_chars += len(alignedSegs[column])
            if (params['alreadyAligned']):
                pre_anchors.append(i)
        id_segs = [str(i) for i in list(range(1, len(segs) + 1))]
        

    # in xml-conll, the conll sentences are encoded between <s></s> markup
    elif input_format == "xml-conll":
        content = f.read()
        try:
            xml_root = ET.fromstring(content)
        except :
            print("non conform XML :", os.path.join(input_dir, input_file))
            print(sys.exc_info()[0])
            # error_log.write("non conform XML :",os.path.join(input_dir, input_file),"\n")
            sys.exit()

        for s_elt in xml_root.findall('.//s'):
            s = "".join(s_elt.itertext())
            # suppression des tabulations et espaces répétés
            s = re.sub(r"\s+"," ",s)
            toks = []
            for line in re.split(r"\n", s):
                cols = re.split("\t", line)
                if len(cols) == 10:
                    toks.append(cols[1])
            seg = " ".join(toks)
            segs.append(seg)
            nb_chars += len(seg)

            if s_elt.attrib["id"]:
                id_segs.append(s_elt.attrib["id"])
            elif s_elt.attrib["xml:id"]:
                id_segs.append(s_elt.attrib["xml:id"])
            else:
                id_segs.append(str(len(segs)))

    # In XML format, the sentences are extracted using the text content of
    # the elements that are defined by xmlGuide (a list of tag or simple xpath expressions)
    elif input_format == "xml":
        content = f.read()
        content = re.sub(r'xmlns="[^"]*"|encoding="UTF-?8"', "", content)
        try:
            xml_root = etree.fromstring(content)
        except Exception as err :
            print("non conform XML :", os.path.join(input_dir, input_file))
            print(err)
            # error_log.write("non conform XML :",os.path.join(input_dir, input_file),"\n")
            sys.exit()
        segs = []
        # text element is default anchor 
        anchor_xpath= ".//" +params['anchorTag'] if params['anchorTag'] else ".//text"
        for prealigned_elt in xml_root.xpath(anchor_xpath):
            if  params['anchorTag']:
                # when an anchor or prealignment is found, feed the preAnchors list
                pre_anchors.append(len(segs))
            # special case where xmlGuide=anchorTag
            if params['anchorTag'] and params['anchorTag'] in params['xmlGuide']:
                prealigned_elts=[prealigned_elt]
            else:
                xpath = '|'.join([".//" + tag for tag in params['xmlGuide'] if tag!=""])
                prealigned_elts=prealigned_elt.xpath(xpath)

            for elt in prealigned_elts:
                content = "".join(elt.itertext())
                content = re.sub(r"[\s]+", " ", content)
                # if split_sent, new elements s must be added
                if split_sent:
                    sents=sentence_splitter(content,params,language,split_sent_regex,segmenter)
                    # deleting childs
                    for child in elt:
                        print("Because of split sent, sub element",child.tag,"will be removed")
                        elt.remove(child)
                    elt.text=""
                    # adding s elements as new children
                    i=1
                    for sent in sents:
                        s=etree.Element("s")
                        s.text=sent
                        elt.append(s)
                        
                        segs.append(sent)
                        params['verbose'] and print("Adding sentence n°",len(segs))

                        nb_chars += len(sent)
                        # recording id in id_segs
                        if 'id' in elt.attrib:
                            id_segs.append(elt.attrib["id"]+"_"+str(i))
                        elif "xml:id" in elt.attrib:
                            id_segs.append(elt.attrib["xml:id"]+"_"+str(i))
                        else:
                            id_segs.append(str(len(segs)))
                        i+=1
                else:
                    segs.append(content)
                    params['verbose'] and print("Adding sentence n°",len(segs))

                    nb_chars += len(content)
                    # recording id in id_segs
                    if 'id' in elt.attrib:
                        id_segs.append(elt.attrib["id"])
                    elif "xml:id" in elt.attrib:
                        id_segs.append(elt.attrib["xml:id"])
                    else:
                        id_segs.append(str(len(segs)))
    # Default format: one sentence per line
    else:
        print("Warning : default format TXT")
        for (i,line) in enumerate(f):
            line = line.strip()
            line = re.sub(r'\x0A|\x0D', '', line)
            nb_chars += len(line)
            segs.append(line)
            if (params['alreadyAligned']):
                pre_anchors.append(i)
        id_segs = [str(i) for i in list(range(1, len(segs) + 1))]

    # Here, the lines that corresponds to the same sentences may be merged
    # The corresponding sentence id will result in the concatenation of initial segment id
    if params['mergeLines']:
        if len(pre_anchors)>0:
            print("******* Attention avec le paramètre 'mergeLines' on ne peut prendre en compte les balises de préalignement")
            pre_anchors.clear()
        if params['verbose']:
            print("Line merging for ", language)
        sents = []
        id_sents = []
        num_sents = 0
        current_sent = []
        current_ids = []
        for (i, seg) in enumerate(segs):
            seg2sents.append([num_sents])
            current_ids.append(id_segs[i])
            current_sent.append(seg)
            # merging when the sentence ends with a separator
            if re.search(merge_lines_regex[language], seg) or seg.upper() == seg:
                id_sents.append("-".join(current_ids))
                sents.append(" ".join(current_sent))
                current_sent = []
                current_ids = []
                num_sents += 1
        if len(current_ids) > 0:
            sents.append(" ".join(current_sent))
            id_sents.append("-".join(current_ids))

    # here, segments can be split in smaller pieces (for xml format, already done !)
    elif split_sent and input_format != "xml":
        if params['verbose']:
            print("Sentence segmentation for ", language)
        sents = []
        id_sents = []
        new_pre_anchors=[]
        if params['useSentenceSegmenter']:
            segmenter.set_active(names[language])
        for (i, seg) in enumerate(segs):
            if i in pre_anchors:
                new_pre_anchors.append(len(sents))
            new_sents=sentence_splitter(seg,params,language,split_sent_regex,segmenter)

            seg2sents.append(list(range(len(sents), len(sents) + len(new_sents))))
            new_ids = [id_segs[i]]
            if len(new_sents) > 1:
                new_ids = [id_segs[i] + "_" + str(j) for j in range(len(new_sents))]
            sents.extend(new_sents)
            id_sents.extend(new_ids)
        # updating pre_anchors
        pre_anchors=new_pre_anchors

    # keeping the same segments as in the input
    else:
        sents = segs
        id_sents = id_segs
        seg2sents = [[j] for j in range(len(sents))]

    len_sents = len(sents)
    if params['verbose']:
        print(len_sents, "sentences for ", language)
        if params['veryVerbose']:
            print("\n".join(sents))

    f.close()

    if params['writeSegmentedInput']:
        input_file_pathSeg = re.sub(r"(.*)([._]\w+)[.]\w+$", r"\1.seg\2.txt", input_file_path)
        if input_file_pathSeg==input_file_path:
            input_file_pathSeg=input_file_path+".seg"
        seg_file = open(input_file_pathSeg, mode="w", encoding="utf8")
        seg_file.write("\n".join([ (f"<anchor/> {i}: " if i in pre_anchors else f"{i}: ")+sent for i,sent in enumerate(sents)]))
        seg_file.close()

    return (sents, id_sents, len_sents, seg2sents, nb_chars, pre_anchors, xml_root)


def sentence_splitter(seg,params,language,split_sent_regex,segmenter):
    """
    args :
        seg (str): the segment to split in sentences
        params (dict) : the global params
        language (str): the language
        split_sent_regex (dict) : the segmenting regex for each language
        segmenter (obj) : the trankit segmenter
    return:
        new_sents (list[str]): the list of strings
    
    """


    if params['useSentenceSegmenter']:
        segmenter.set_active(names[language])

    # use trankit for sentence segmentation
    if params['useSentenceSegmenter']:
        print("segmentation de ", seg)
        sentences = segmenter.ssplit(seg)['sentences']
        some_sents = [sent['text'] for sent in sentences]
    # or use a set of regex declared in split_sent
    else:
        if params["splitSentRegex"]:
            regex=params["splitSentRegex"]
        elif language in split_sent_regex:
            regex = split_sent_regex[language]
        else:
            regex = split_sent_regex["default"]
        some_sents = re.split(regex, seg)
    
    last_sent = ""
    new_sents = []
    # the splitted segment that are too small (< seg_min_length)
    # are grouped with the follower
    for sent in some_sents:
        if not re.match(r'^\s*$',sent):
            if len(last_sent + sent) > seg_min_length:
                new_sents.append(last_sent + " " + sent)
                last_sent = ""
            else:
                if last_sent == "":
                    last_sent = sent
                else:
                    last_sent += " " + sent
    if last_sent:
        new_sents.append(last_sent)
    return new_sents


# write only alignable intervals of l1 or l2 file
def write_alignable(sents, id_sents, intervals, index, output_dir, output_file, output_format):
    """
    Arguments :
        sents : List(str) : the sentence list
        id_sents : List(str) : the corresponding sentence ids
        intervals : List(List(int)) : the list of pairs [i..j] that defines corresponding intervals. The second axe is the language : 0 for l1, 1 for l2
        index : 0 or 1 for l1 or l2
        output_dir : str : the path of output dir
        output_file : str : the name of output file
        output_format : str : "ces" or "arc" or "txt"

    No return value, but the file output_file is written on the disk
    """


    try:
        output = open(os.path.join(output_dir, output_file), mode="w", encoding="utf8")
        # output header
        if output_format in ["ces","arc"]:
            output.write(header(params,output_format))

        # output sentences
        for interval in intervals:
            i1 = interval[0][index]
            i2 = interval[1][index]

            for i in range(i1, i2 + 1):
                if output_format == "ces" or output_format == "arc":
                    if input_format == "ces" or output_format == "arc":
                        id_sent = id_sents[i]
                    else:
                        id_sent = str(i + 1)
                    output.write("<s id=\"" + id_sent + "\">\n" + toXML(sents[i]) + "\n</s>\n")
                else:
                    output.write(sents[i] + "\n")

        # output footer
        if output_format in ["ces","arc"]:
            output.write(footer(output_format))

        output.close()
    except:
        print("Error: a problem occurred while writing", output_file)
        sys.exit()


# write aligned points
# if the anchor parameter is true then filtered_x and filtered_y are list of int
# if not, they are list of list of int (the grouped coordinate)
def write_aligned_points(params, sents1, id_sents1, sents2, id_sents2, filtered_x, filtered_y, output_dir, output_file,
                         output_format, anchor, print_ids=False, mean_score=0, file1="", file2=""):
    """
    Arguments :
        params : dict : the global parameters
        sents1 : List(str) : the L1 sentence list
        id_sents1 : List(str) : the corresponding sentence ids
        sents2 : List(str) : the L2 sentence list
        id_sents2 : List(str) : the corresponding sentence ids
        filtered_x : List(List(int)) OR List(int) if anchor=True
            if anchor == false : the X coordinates of groups in L1 (ex. :[[0],[1,2],[3],[4,5,6]])
            if anchor == true : the X coordinates of points in L1 (ex. [0, 2, 3, 5])
        filtered_y : List(List(int)) OR List(int) if anchor=True
            if anchor == false : the Y coordinates of groups in L2 (ex. :[[0,1],[2],[3],[4,5]])
            if anchor == true : the Y coordinates of points in L2 (ex. [1, 2, 3, 5])
        output_dir : str : the path of output dir
        output_file : str : the name of output file
        output_format : str : "tmx" or "ces" or "ids" or "txt"

    No return value, but the file output_file is written on the disk
    """

    l1=params['l1']
    l2=params['l2']
    input_format=params['inputFormat']

    if output_format == "txt2":
        output_file1 = output_file.replace(".txt2", "." + l1 + ".txt")
        output1 = open(os.path.join(output_dir, output_file1), mode="w", encoding="utf8")
        output_file2 = output_file.replace(".txt2", "." + l2 + ".txt")
        output2 = open(os.path.join(output_dir, output_file2), mode="w", encoding="utf8")
    else:
        if output_file[-4:] == "tsv2":
            output_file = output_file[:-1]
        output = open(os.path.join(output_dir, output_file), mode="w", encoding="utf8")
    # ~ output2=open(os.path.join(output_dir,output_file+".txt"),mode="w",encoding="utf8")
    # output header
    if output_format == "ces":
        output.write(re.sub(r'__mean_score__', f"{mean_score:.4f}", header(params,"ces_align")))
    elif output_format == "tmx":
        output.write(re.sub(r'__mean_score__', f"{mean_score:.4f}", header(params,"tmx")))
    elif output_format == "txt":
        output.write(f"Mean similarity:{mean_score}\n")
    elif output_format == "txt2":
        output1.write(f"Mean similarity:{mean_score}\n")
        output2.write(f"Mean similarity:{mean_score}\n")
    elif output_format == "tsv2":
        # ~ m=re.match('(.*)[.][^.]+[.][^.]+$',output_file)
        # ~ name=m.group(1)
        name1 = os.path.basename(file1)
        name2 = os.path.basename(file2)
        output.write(f"source={l1}/{collection_name}/{name1}	target={l2}/{collection_name}/{name2}\n\n")

    # output sentences
    for i in range(len(filtered_x)):
        if anchor:
            x = [filtered_x[i]]
            y = [filtered_y[i]]
        else:
            x = filtered_x[i]
            y = filtered_y[i]
        if output_format == "ces":
            if input_format == "ces" or input_format == "arc":
                id_sent1 = " ".join([id_sents1[x[j]] for j in range(len(x))])
                id_sent2 = " ".join([id_sents2[y[j]] for j in range(len(y))])
            else:
                id_sent1 = " ".join([str(x[j] + 1) for j in range(len(x))])
                id_sent2 = " ".join([str(y[j] + 1) for j in range(len(y))])
            output.write(f"\t\t<link xtargets=\"{id_sent1} ; {id_sent2}\"/>\n")
        elif output_format == "ids" or output_format == "tsv2":
            if input_format == "ces" or input_format == "arc":
                id_sent1 = " ".join([id_sents1[x[j]] for j in range(len(x))])
                id_sent2 = " ".join([id_sents2[y[j]] for j in range(len(y))])
            else:
                id_sent1 = " ".join([str(x[j] + 1) for j in range(len(x))])
                id_sent2 = " ".join([str(y[j] + 1) for j in range(len(y))])
                output.write(f"{id_sent1}\t{id_sent2}\n")
        elif output_format == "tmx":
            srcSegs = "".join(["\t\t<seg>" + toXML(sents1[x[j]]) + "</seg>\n" for j in range(len(x))])
            tgtSegs = "".join(["\t\t<seg>" + toXML(sents2[y[j]]) + "</seg>\n" for j in range(len(y))])

            output.write(f"<tu>\n")
            output.write(f"\t<tuv xml:lang=\"{l1}\">\n{srcSegs}\t</tuv>\n")
            output.write(f"\t<tuv xml:lang=\"{l2}\">\n{tgtSegs}\t</tuv>\n")
            output.write(f"</tu>\n")
        elif output_format == "txt":
            ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if print_ids else ""
            sent1 = ids1 + " ".join([sents1[x[j]] for j in range(len(x))])
            ids2 = "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if print_ids else ""
            sent2 = ids2 + " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(sent1 + "\n" + sent2 + "\n\n")
        elif output_format == "txt2":
            sent1 = " ".join(["[" + str(x[j]) + "] " + sents1[x[j]] for j in range(len(x))])
            output1.write(sent1 + "\n")
            sent2 = " ".join(["[" + str(y[j]) + "] " + sents2[y[j]] for j in range(len(y))])
            output2.write(sent2 + "\n")
        elif output_format == "tsv":
            if print_ids :
                sent1 = " ".join(["["+str(x[j])+"] "+sents1[x[j]] for j in range(len(x))])
                sent2 = " ".join(["["+str(y[j])+"] "+sents2[y[j]] for j in range(len(y))])
            else:
                sent1 = " ".join([sents1[x[j]] for j in range(len(x))])
                sent2 = " ".join([sents2[y[j]] for j in range(len(y))])
            output.write(f"{sent1}\t{sent2}\n")
            
        elif output_format == "bertalign":
            ids1 = "[" + ",".join([str(x[j]) for j in range(len(x))]) + "]"
            ids2 = "[" + ",".join([str(y[j]) for j in range(len(y))]) + "]"
            output.write(f"{ids1}:{ids2}\n")
        # ~ else:
            # ~ # default is TSV with no ID
            # ~ ids1 = "[" + " ".join([str(x[j]) for j in range(len(x))]) + "] " if print_ids else ""
            # ~ sent1 = ids1 + " ".join([sents1[x[j]] for j in range(len(x))])
            # ~ ids2 = "[" + " ".join([str(y[j]) for j in range(len(y))]) + "] " if print_ids else ""
            # ~ sent2 = ids2 + " ".join([sents2[y[j]] for j in range(len(y))])
            # ~ output.write(sent1 + "\t" + sent2 + "\n")

    # output footer
    if output_format in ["ces","tmx"]:
        output.write(footer(output_format))

    if output_format == "txt2":
        output1.close()
        output2.close()
    else:
        output.close()


# Writing anchors in xml files
def add_anchor_in_output(params, input_file1, input_file2, xml_root1, xml_root2, file_id1, file_id2, x, y):
    """
    Adds anchors in input xml files

    Args:
        input_dir: The directory containing the input file.
        file1: The name of the file1.
        file2: The name of the file2.
        xml_root1 : The ET element for file1 (may be modified if split_sent=True)
        xml_root2 : The ET element for file2 (may be modified if split_sent=True)
        x: the source coordinates
        y: the corresponding target coordinates

    Returns:
        write files output_dir/file1 and output_dir/file2 (or input_dir/file1 and input_dir/file2 if output_dir is empty)
        adding anchors
    """
    
    global xml_id_offset

    input_dir=params['inputDir']
    output_dir=params['outputDir']
    direction=params['direction']
    
    if xml_root1==None or xml_root2==None:
        print ("Impossible to add anchors to",input_file1," and ",input_file2,"because of XML format problem")
        return 

    segs = []
    if params["splitSent1"]:
        xpath1 = './/s'
    else:
        xpath1 = '|'.join(['.//' + tag for tag in params['xmlGuide']])
    if params["splitSent2"]:
        xpath2 = './/s'
    else:
        xpath2 = '|'.join(['.//' + tag for tag in params['xmlGuide']])
        
    # ~ sents1= xml_root1.findall(xpath)
    # ~ sents2= xml_root2.findall(xpath)
    sents1 = xml_root1.xpath(xpath1)
    sents2 = xml_root2.xpath(xpath2)

    hash_sign_for_corresp = "#" if params['hashSignInAnchor'] else ""

    for i in range(len(x)):
        if len(x[i]) > 0 and len(y[i]) > 0:
            xi = x[i][0]
            yi = y[i][0]
            corresp2_values = " ".join(hash_sign_for_corresp + file_id1 + str(xi + xml_id_offset) for xi in x[i])

            if direction == "1<->2" or direction == "2->1":
                anchor1 = etree.Element("anchor")
                anchor1.set("{http://www.w3.org/XML/1998/namespace}id", file_id1 + str(xi + xml_id_offset))
                anchor1.set("corresp", hash_sign_for_corresp + file_id2 + str(yi + xml_id_offset))
                prev = sents1[xi].getprevious()
                if prev is not None:
                    prev.addnext(anchor1)
                else:
                    parent = sents1[xi].getparent()
                    parent.insert(0, anchor1)

            if direction == "1<->2" or direction == "1->2":
                anchor2 = etree.Element("anchor")
                anchor2.set("{http://www.w3.org/XML/1998/namespace}id", file_id2 + str(yi + xml_id_offset))
                anchor2.set("corresp", corresp2_values)
                prev = sents2[yi].getprevious()
                if prev is not None:
                    prev.addnext(anchor2)
                else:
                    parent = sents2[yi].getparent()
                    parent.insert(0, anchor2)
        elif len(y[i]) > 0:
            yi = y[i][0]
            if direction == "1<->2" or direction == "1->2":
                anchor2 = etree.Element("anchor")
                anchor2.set("{http://www.w3.org/XML/1998/namespace}id", file_id2 + str(yi + xml_id_offset))
                prev = sents2[yi].getprevious()
                if prev is not None:
                    prev.addnext(anchor2)
                else:
                    parent = sents2[yi].getparent()
                    parent.insert(0, anchor2)

    # add anchor before each source sentence
    if direction == '1->2':
        for i in range(len(sents1)):
            sent = sents1[i]
            anchor1 = etree.Element("anchor")
            anchor1.set("{http://www.w3.org/XML/1998/namespace}id", file_id1 + str(i + xml_id_offset))
            prev = sents1[i].getprevious()
            if prev is not None:
                prev.addnext(anchor1)
            else:
                parent = sents1[i].getparent()
                parent.insert(0, anchor1)
    elif direction == '2->1':
        for i in range(len(sents2)):
            sent = sents2[i]
            anchor2 = etree.Element("anchor")
            anchor2.set("{http://www.w3.org/XML/1998/namespace}id", file_id2 + str(i + xml_id_offset))
            prev = sents2[i].getprevious()
            if prev is not None:
                prev.addnext(anchor2)
            else:
                parent = sents2[i].getparent()
                parent.insert(0, anchor2)

    # writing output files
    if output_dir == "" or output_dir == None:
        output_file_path1 = input_file_path1
        output_file_path2 = input_file_path2
    else:
        output_file_path1 = os.path.join(output_dir, os.path.basename(input_file1))
        output_file_path2 = os.path.join(output_dir, os.path.basename(input_file2))

    try:
        tree1 = etree.ElementTree(xml_root1)
        print(f"Writing {output_file_path1}")
        tree1.write(output_file_path1, encoding='utf-8', pretty_print=True)
    except:
        print("Error: a problem occurred while writing", output_file_path1)
        sys.exit()

    try:
        tree2 = etree.ElementTree(xml_root2)
        print(f"Writing {output_file_path2}")
        tree2.write(output_file_path2, encoding='utf-8', pretty_print=True)
    except:
        print("Error: a problem occurred while writing", output_file_path2)
        sys.exit()

# reading TSV or CES file with ids 
def read_alignment_file(params):
    """
    args : 
        params (dict): global params
    returns :
        x (list[list]): the list of num groups for x [[0,1],[2],[]...]
        y (list[list]): the list of num groups for y [[0],[1],[2,3]...]
    """
    aligned_file_name=params['alignedFileName']
    name_ext=os.path.splitext(aligned_file_name)
    x=[]
    y=[]
 
    if name_ext[1].lower() in ('.csv','.tsv'):
        delimiter= "\t" if name_ext[1].lower() == ".tsv" else ";"
        with open(aligned_file_name, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
            i=0
            j=0
            for row in csvreader:
                if len(row)==2:
                    source=row[0]
                    target=row[1]
                    ids1=re.findall(r'\[\d.*?\]',source)
                    ids2=re.findall(r'\[\d.*?\]',target)
                    l1=list(range(i,i+len(ids1)))
                    l2=list(range(j,j+len(ids2)))
                    if l1!=[] or l2!=[]:
                        x.append(l1)
                        y.append(l2)
                        i+=len(ids1)
                        j+=len(ids2)
                else:
                    print("Unreadable line in alignment file :",row)
            params['verbose'] and print("Aligned sentences: ",i,"x",j) 
    elif name_ext[1].lower() in ('.ces','.cesalign'):
        with open(aligned_file_name,encoding="utf8") as f:
            for line in f:
                m=re.search(r'<link xtargets\s*=\s*"(.*);(.*)"')
                if m:
                    l1=[ int(sid)-1 for sid in m.group(1).split(" ")]
                    l2=[ int(sid)-1 for sid in m.group(2).split(" ")]
                    x.append(l1)
                    y.append(l2)
                else:
                    print("Unreadable line in alignment file :",line)
    else:
        print("Warning : can only process csv/tsv files with ID, or CESAlign format")
        print(aligned_file_name,"will be ignored")
    

    return (x,y)
