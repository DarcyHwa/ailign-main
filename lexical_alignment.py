import os

import stanza
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extract_chunks import extract_chunks, extract_flat_chunks, extract_words
import json
import pyconll
import re
import sys

def convert_conll_list_to_string(doc):
    try:
        conll_string = ""
        for sentence in doc.sentences:
            for token in sentence.tokens:
                for i,word in enumerate(token.words):
                    # here, for compound token, use the compound form
                    # not the analysed forms
                    if i==0:
                        form =  word.parent.text
                    else:
                        form = ""
                    line = "\t".join([
                        str(word.id) if word.id is not None else '_',
                        form,
                        word.lemma if word.lemma is not None else '_',
                        word.upos if word.upos is not None else '_',
                        word.xpos if word.xpos is not None else '_',
                        word.feats if word.feats is not None else '_',
                        str(word.head) if word.head is not None else '0',
                        word.deprel if word.deprel is not None else '_',
                        word.deps if word.deps is not None else '_',
                        word.misc if word.misc is not None else '_'
                    ])
                    conll_string += line + '\n'
                    form=""
            conll_string += '\n'
        return conll_string
    except Exception as e:
        print(f"An error occurred in document_to_conll_string: {e}")
        return ""

def update_conll_ids(conll_string, start_id=0):
    """
    This function updates token ids and their corresponding head ids in a CONLL-U formatted string.
    Args:
        conll_string (str): The CONLL-U formatted string.
        start_id (int): The starting ID for the first token in the string.

    Returns:
        str: Updated CONLL-U string with continuous ids.
        int: The last id used in the updated CONLL string.
    """
    updated_conll = ""
    current_id = start_id

    lines = conll_string.split('\n')
    id_mapping = {}

    for line in lines:
        if line.strip() == "":
            updated_conll += line + '\n'
            continue

        parts = line.split('\t')
        if len(parts) < 10:
            continue  # Skip malformed lines

        # Increment the token's id and map old id to new id
        old_id = int(parts[0])
        current_id += 1
        id_mapping[old_id] = current_id
        parts[0] = str(current_id)

        # Update head if it's not zero (i.e., if it points to some other token)
        if parts[6] != '0':
            old_head_id = int(parts[6])
            parts[6] = str(id_mapping.get(old_head_id, 0))

        updated_line = '\t'.join(parts)
        updated_conll += updated_line + '\n'

    return updated_conll, current_id

def word_alignment(l1, l2, x, y, encoder, sents1, sents2, file_name, output_directory, outputFormats):
    # cesalign format is used to store alignment result
    ces_align_header = f"""<?xml version="1.0" encoding="utf-8"?>

    <cesAlign type="seg" version="1.6">

    <ces_header version="2.3" mean_score="__mean_score__">
        <translations>
            <translation lang="{l1}" />
            <translation lang="{l2}" />
        </translations>
    </ces_header>

    <linkList>
        <linkGrp targType="seg">

    """
    ces_align_footer = """
    </linkGrp>
    </linkList>

    </cesAlign>
    """
    # Load Stanza models for the specified languages
    nlp_l1 = stanza.Pipeline(lang=l1, processors='tokenize,mwt,pos,lemma,depparse')
    nlp_l2 = stanza.Pipeline(lang=l2, processors='tokenize,mwt,pos,lemma,depparse')

    # Initialize the list to store final word or chunk alignments
    alignments = []
    alignments_ids = []

    # Initialize the id counters
    last_id_l1 = 0
    last_id_l2 = 0
    
    langSrc = l1
    langTarget = l2



    # Iterate over each group of aligned sentences
    # The function zip(x, y) pairs each element of x with the corresponding element in y,
    # allowing the loop to process these pairs in tandem
    # La fonction zip(x, y) associe chaque élément de x avec l'élément correspondant dans y,
    # permettant à la boucle de traiter ces paires en tandem
    for group_x, group_y in zip(x, y):
        # Concatenate sentences in each group to form a single text for parsing
        text_l1 = ' '.join([sents1[i] for i in group_x])  
        text_l2 = ' '.join([sents2[i] for i in group_y])

        # Process texts with Stanza to get CoNLL-U formatted data
        doc_l1 = nlp_l1(text_l1)
        doc_l2 = nlp_l2(text_l2)

        # Use the function to convert documents to CoNLL format
        conll_l1 = convert_conll_list_to_string(doc_l1)
        conll_l2 = convert_conll_list_to_string(doc_l2)

        # Update IDs in the CoNLL strings
        conll_l1, last_id_l1 = update_conll_ids(conll_l1, last_id_l1)
        conll_l2, last_id_l2 = update_conll_ids(conll_l2, last_id_l2)

        # Now, you can extract chunks from the CoNLL data
        conll_l1_sentences = pyconll.load_from_string(conll_l1)
        conll_l2_sentences = pyconll.load_from_string(conll_l2)
        # Use the modified extract_words function
        words_l1 = extract_words(conll_l1_sentences)
        words_l2 = extract_words(conll_l2_sentences)

        # Compute embeddings for each word
        word_embeds_l1 = encoder.encode([word[0] for word in words_l1])
        word_embeds_l2 = encoder.encode([word[0] for word in words_l2])

        # Check if either of the embeddings is empty
        if word_embeds_l1.size == 0 or word_embeds_l2.size == 0:
            print("One of the languages has no words, skipping similarity calculation for this pair.")
            continue

        # Adjust dimensions if necessary
        if word_embeds_l1.ndim == 1:
            word_embeds_l1 = word_embeds_l1.reshape(1, -1)
        if word_embeds_l2.ndim == 1:
            word_embeds_l2 = word_embeds_l2.reshape(1, -1)

        similarity_matrix = cosine_similarity(word_embeds_l1, word_embeds_l2)

        # For each chunk in l1, find the best matching chunk in l2
        # first pass : pairing, associating each id1 to the best (id2,score) 
        id2_to_id1s={}
        for i, row in enumerate(similarity_matrix):
            best_match_index = np.argmax(row)
            if best_match_index not in id2_to_id1s:
                id2_to_id1s[best_match_index]=[]
            id2_to_id1s[best_match_index].append((i,row[best_match_index]))
        
        # second pass : resolving conflicts
        # if the same id2 is associated with different id1, the best association is conserved and other pairing are deleted
        id1_to_id2={}
        for id2 in id2_to_id1s:
            # reducing conflicts by keeping the best association for i2
            if len(id2_to_id1s[id2]) >= 2:
                best_match_pair=np.argmax([pair[1] for pair in id2_to_id1s[id2]])
                id2_to_id1s[id2]=[id2_to_id1s[id2][best_match_pair]]
            id1=id2_to_id1s[id2][0][0]
            id1_to_id2[id1]=id2
            
        # third pass : associating the missing id1
        for i, row in enumerate(similarity_matrix):
            if not i in id1_to_id2.keys():
                best_match_indices = list(np.argsort(row))
                best_match_indices.reverse()
                found=False
                for id2 in best_match_indices:
                    # if id2 is still free, it can be associated
                    if id2 not in id2_to_id1s:
                        id1_to_id2[i]=id2
                        id2_to_id1s[id2]=[(i,row[id2])]
                        found=True
                        break
                
            best_word_l2=["",""]
            best_match_score=0
            best_match_index=id1_to_id2.get(i,None)
            if  best_match_index != None:
                best_match_score = float(row[best_match_index])
                alignments_ids.append((words_l1[i][1], words_l2[best_match_index][1]))
                alignments.append({
                    'l1_word': words_l1[i][0],
                    'l2_word': words_l2[best_match_index][0],
                    'similarity': best_match_score
                })


        # old strategy with no injectivity
        # ~ for i, row in enumerate(similarity_matrix):
            # ~ best_match_index = np.argmax(row)
            # ~ best_match_score = float(row[best_match_index])
            # ~ alignments_ids.append((words_l1[i][1], words_l2[best_match_index][1]))
            # ~ alignments.append({
                # ~ 'l1_word': words_l1[i][0],
                # ~ 'l2_word': words_l2[best_match_index][0],
                # ~ 'similarity': best_match_score
            # ~ })

    if "json" in outputFormats:
        output_file_name_json = file_name + f"_{langTarget}-{langSrc}_word_ai.json"
        output_path_json = os.path.join(output_directory, output_file_name_json)
        with open(output_path_json, 'w', encoding='utf-8') as file:
            json.dump(alignments, file, ensure_ascii=False, indent=4)

    if "ces" in outputFormats:
        ces_align_body = ""
        for i, (id1, id2) in enumerate(alignments_ids):
            ces_align_body += f'<link xtargets="{id1} ; {id2}"/>\n'
        ces_align_content = ces_align_header + ces_align_body + ces_align_footer
        output_file_name = file_name + f"_{langTarget}-{langSrc}_word_ai.ces"
        output_path = os.path.join(output_directory, output_file_name)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(ces_align_content)

    if "txt" in outputFormats:
        aligned_txt_file_name = file_name + f"_{langTarget}-{langSrc}_word_ai.txt"
        output_path_formatted = os.path.join(output_directory, aligned_txt_file_name)
        with open(output_path_formatted, 'w', encoding='utf-8') as formatted_file:
            for (id1, id2), alignment in zip(alignments_ids, alignments):
                formatted_file.write(f"[{id1}] {alignment['l1_word']}\n")
                formatted_file.write(f"[{id2}] {alignment['l2_word']}\n")
                formatted_file.write('\n')
                
    if "tsv" in outputFormats:
        aligned_txt_file_name = file_name + f"_{langTarget}-{langSrc}_word_ai.tsv"
        output_path_formatted = os.path.join(output_directory, aligned_txt_file_name)
        with open(output_path_formatted, 'w', encoding='utf-8') as formatted_file:
            for (id1, id2), alignment in zip(alignments_ids, alignments):
                formatted_file.write(f"{alignment['l1_word']}\t")
                formatted_file.write(f"{alignment['l2_word']}\t")
                formatted_file.write(f"{alignment['similarity']}\n")

    return alignments

def chunk_alignment(l1, l2, x, y, encoder, sents1, sents2, file_name, output_directory, outputFormats):
    ces_align_header = f"""<?xml version="1.0" encoding="utf-8"?>

        <cesAlign type="seg" version="1.6">

        <ces_header version="2.3" mean_score="__mean_score__">
            <translations>
                <translation lang="{l1}" />
                <translation lang="{l2}" />
            </translations>
        </ces_header>

        <linkList>
            <linkGrp targType="seg">

        """
    ces_align_footer = """
        </linkGrp>
        </linkList>

        </cesAlign>
        """
    # Load Stanza models for the specified languages
    nlp_l1 = stanza.Pipeline(lang=l1, processors='tokenize,mwt,pos,lemma,depparse')
    nlp_l2 = stanza.Pipeline(lang=l2, processors='tokenize,mwt,pos,lemma,depparse')

    # Initialize the list to store final word or chunk alignments
    alignments = []
    alignments_ids = []

    # Initialize the id counters
    last_id_l1 = 0
    last_id_l2 = 0
    # split filename by / to get the last part of the path and then split by . to get the languages
    #filename_split= file_name.split("/")[-1].split("_")[1].split(".")
    langSrc = l1
    langTarget = l2

    # Iterate over each group of aligned sentences
    # The function zip(x, y) pairs each element of x with the corresponding element in y,
    # allowing the loop to process these pairs in tandem
    # La fonction zip(x, y) associe chaque élément de x avec l'élément correspondant dans y,
    # permettant à la boucle de traiter ces paires en tandem
    
    # hash that associates ids to tokens, globally
    tokens1={}
    tokens2={}
    numSent1=0
    numSent2=0
    for group_x, group_y in zip(x, y):
        # Concatenate sentences in each group to form a single text for parsing
        text_l1 = ' '.join([sents1[i] for i in group_x])  # Adjust indices for 0-based indexing
        text_l2 = ' '.join([sents2[i] for i in group_y])

        # Process texts with Stanza to get CoNLL-U formatted data
        doc_l1 = nlp_l1(text_l1)
        doc_l2 = nlp_l2(text_l2)

        # Use the function to convert documents to CoNLL format
        conll_l1 = convert_conll_list_to_string(doc_l1)
        conll_l2 = convert_conll_list_to_string(doc_l2)

        # Update IDs in the CoNLL strings
        conll_l1, last_id_l1 = update_conll_ids(conll_l1, last_id_l1)
        conll_l2, last_id_l2 = update_conll_ids(conll_l2, last_id_l2)
 
        # Now, you can extract chunks from the CoNLL data
        conll_l1_sentences = pyconll.load_from_string(conll_l1)
        conll_l2_sentences = pyconll.load_from_string(conll_l2)
        
        # recording the tokens in the hashes
        for sentence in conll_l1_sentences:
            numSent1+=1
            for token in sentence:
                token.misc["numSent"]=numSent1
                tokens1[token.id]=token
        for sentence in conll_l2_sentences:
            numSent2+=1
            for token in sentence:
                token.misc["numSent"]=numSent2
                tokens2[token.id]=token                
   
       
        # Chunks are lists of (chunk,ids) where ids are the corresponding token ids
        chunks_l1 = extract_flat_chunks(conll_l1_sentences)
        chunks_l2 = extract_flat_chunks(conll_l2_sentences)

        if not chunks_l1 or not chunks_l2:
            print("One of the languages has no chunks, skipping similarity calculation for this pair.")
            continue
        # Compute embeddings for each chunk
        chunk_embeds_l1 = encoder.encode([chunk[0] for chunk in chunks_l1])
        chunk_embeds_l2 = encoder.encode([chunk[0] for chunk in chunks_l2])

        if chunk_embeds_l1.ndim == 1:
            chunk_embeds_l1 = chunk_embeds_l1.reshape(1, -1)
        if chunk_embeds_l2.ndim == 1:
            chunk_embeds_l2 = chunk_embeds_l2.reshape(1, -1)

        # Find the best matching chunks based on cosine similarity
        similarity_matrix = cosine_similarity(chunk_embeds_l1, chunk_embeds_l2)

        # For each chunk in l1, find the best matching chunk in l2
        # first pass : pairing, associating each id1 to the best (id2,score) 
        id2_to_id1s={}
        for i, row in enumerate(similarity_matrix):
            best_match_index = np.argmax(row)
            if best_match_index not in id2_to_id1s:
                id2_to_id1s[best_match_index]=[]
            id2_to_id1s[best_match_index].append((i,row[best_match_index]))
        
        # second pass : resolving conflicts
        # if the same id2 is associated with different id1, the best association is conserved and other pairing are deleted
        id1_to_id2={}
        for id2 in id2_to_id1s:
            # reducing conflicts by keeping the best association for i2
            if len(id2_to_id1s[id2]) >= 2:
                best_match_pair=np.argmax([pair[1] for pair in id2_to_id1s[id2]])
                id2_to_id1s[id2]=[id2_to_id1s[id2][best_match_pair]]
            id1=id2_to_id1s[id2][0][0]
            id1_to_id2[id1]=id2
            
        # third pass : associating the missing id1
        for i, row in enumerate(similarity_matrix):
            if not i in id1_to_id2.keys():
                best_match_indices = list(np.argsort(row))
                best_match_indices.reverse()
                found=False
                for id2 in best_match_indices:
                    # if id2 is still free, it can be associated
                    if id2 not in id2_to_id1s:
                        id1_to_id2[i]=id2
                        id2_to_id1s[id2]=[(i,row[id2])]
                        found=True
                        break
                # ~ if not found:
                    # ~ print ("3. assoc",i,"-> NULL")
                
            best_chunk_l2=["",""]
            best_match_score=0
            best_match_index=id1_to_id2.get(i,None)
            if  best_match_index != None:
                best_chunk_l2=chunks_l2[best_match_index]
                best_match_score = float(row[best_match_index])
            
            alignments_ids.append((chunks_l1[i][1], best_chunk_l2[1]))
            alignments.append({
                'l1_chunk': chunks_l1[i][0],
                'l2_chunk': best_chunk_l2[0],
                'l1_chunk_ids' : chunks_l1[i][1],
                'l2_chunk_ids' : best_chunk_l2[1],
                'similarity': best_match_score
            })

    # print top 10 lexical alignments
    # ~ for alignment in alignments[:10]:
        # ~ print(f"Chunk in {l1}: {alignment['l1_chunk']}")
        # ~ print(f"Chunk in {l2}: {alignment['l2_chunk']}")
        # ~ print(f"Similarity score: {alignment['similarity']:.2f}")
        # ~ print("-" * 30)

    # File name for the raw data
    if "json" in outputFormats:
        output_file_name_json = file_name + f"_{langTarget}-{langSrc}_phrase_ai.json"
        output_path_json = os.path.join(output_directory, output_file_name_json)
        with open(output_path_json, 'w', encoding='utf-8') as file:
            json.dump(alignments, file, ensure_ascii=False, indent=4)
    if "ces" in outputFormats:
        ces_align_body = ""
        for i, (id1, id2) in enumerate(alignments_ids):
            ces_align_body += f'<link xtargets="{" ".join(id1)} ; {" ".join(id2)}"/>\n'
        ces_align_content = ces_align_header + ces_align_body + ces_align_footer


        output_file_name = file_name + f"_{langTarget}-{langSrc}_phrase_ai.ces"
        output_path = os.path.join(output_directory, output_file_name)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(ces_align_content)
    if "txt" in outputFormats:
        aligned_txt_file_name = file_name + f"_{langTarget}-{langSrc}_phrase_ai.txt"
        output_path_formatted = os.path.join(output_directory, aligned_txt_file_name)
        with open(output_path_formatted, 'w', encoding='utf-8') as formatted_file:
            for (ids1, ids2), alignment in zip(alignments_ids, alignments):
                formatted_ids1 = ' '.join(f"[{id}]" for id in ids1)  # Properly format multiple ids
                formatted_ids2 = ' '.join(f"[{id}]" for id in ids2)
                formatted_file.write(f"{formatted_ids1} {alignment['l1_chunk']}\n")
                formatted_file.write(f"{formatted_ids2} {alignment['l2_chunk']}\n")
                formatted_file.write('\n')
    if "tsv" in outputFormats:
        aligned_txt_file_name = file_name + f"_{langTarget}-{langSrc}_phrase_ai.tsv"
        output_path_formatted = os.path.join(output_directory, aligned_txt_file_name)
        with open(output_path_formatted, 'w', encoding='utf-8') as formatted_file:
            for (ids1, ids2), alignment in zip(alignments_ids, alignments):
                formatted_file.write(f"{alignment['l1_chunk']}\t")
                formatted_file.write(f"{alignment['l2_chunk']}\t")
                lemmas1=" ".join([tokens1[tokId].lemma for tokId in alignment['l1_chunk_ids']])
                lemmas2=" ".join([tokens2[tokId].lemma for tokId in alignment['l2_chunk_ids']])
                formatted_file.write(f"{lemmas1}\t")
                formatted_file.write(f"{lemmas2}\t")
                upos1=" ".join([tokens1[tokId].upos for tokId in alignment['l1_chunk_ids']])
                upos2=" ".join([tokens2[tokId].upos for tokId in alignment['l2_chunk_ids']])
                formatted_file.write(f"{upos1}\t")
                formatted_file.write(f"{upos2}\n")                

                
    return alignments
