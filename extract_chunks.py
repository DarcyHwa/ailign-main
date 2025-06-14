import requests
import json
import re
import sys

from urllib.parse import urlencode


# Retrieves CoNLL-U formatted data for the given text using the UDPipe service.
def get_conllu_data(text):
    # Endpoint URL for the UDPipe service
    endpoint = "http://lindat.mff.cuni.cz/services/udpipe/api/process"
    data = {
        'data': text
    }
    # Encodes the data and prepares the payload for the POST request
    encoded_data = urlencode(data)
    payload = 'tokenizer=&tagger=&parser=&model=french-gsd-ud-2.12-230717&' + encoded_data
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    try:
        # Sends the request and parses the response
        response = requests.post(endpoint, headers=headers, data=payload)
        # Raise an exception for 4xx or 5xx status codes
        response.raise_for_status()
        response_data = json.loads(response.text)
        result_conllu_data = response_data.get('result', '')
        return result_conllu_data
    except requests.exceptions.RequestException as e:
        # Handles any request errors
        raise Exception(f"Failed to get data: {str(e)}")

    # function for extracting chunks from a list of tokens.
def extract_words(sentences):
    words = []

    for sentence in sentences:
        i=0
        while i < len(sentence):
            token=sentence[i]
            i+=1
            # useless because of convert_conll_list_to_string (multi tokens are simplified) 
            # processing multi-word tokens
            # ~ m=re.search(r'(\d+)-(\d+)',token.id)
            # ~ if m:
                # ~ begin=int(m.groups(1))
                # ~ end=int(m.groups(2))
                # ~ # jump to the next token
                # ~ i+=end-begin+1
            # Append word text and its ID
            if token.form!="":
                words.append((token.form, token.id))

    return words

# Extracting flat chunks based on a simple heuristique
# We keep verb phrase and noun phrase, and the rest will give simple chunks
def extract_flat_chunks(sentences):
    chunks = []
    for sentence in sentences:
        # Create a mapping from each token to its head
        head_map = {token.id: token.head for token in sentence}
        
        i=0
        current_chunk=[]
        current_chunk_ids=[]
        while i < len(sentence):
            token=sentence[i]
            current_chunk.append(token.form)
            current_chunk_ids.append(token.id)            
            i+=1
            # processing multi-word tokens
            # useless because of convert_conll_list_to_string (multi tokens are simplified) 
            # ~ m=re.search(r'(\d+)-(\d+)',token.id)
            # ~ if m:
                # ~ # inherit the pos of the first compound token
                # ~ token.upos=sentence[i].upos
                # ~ begin=int(m.groups(1))
                # ~ end=int(m.groups(2))
                # ~ # jump to the next token
                # ~ i+=end-begin+1

            # noun phrase
            if token.upos in ('DET','NUM','NOUN','ADJ','PROPN'):
                # note that the inner tokens must not be DET (which opens a new noun phrase)
                while i < len(sentence) and sentence[i].lemma != "ne" and \
                    (sentence[i].upos in ('NUM','NOUN','ADV','ADJ','PROPN') or \
                    str(sentence[i].feats.get('VerbForm'))=="{'Part'}"):
                    token=sentence[i]
                    # ~ print(token.id,token.form, token.lemma)
                    i+=1
                    current_chunk.append(token.form)
                    current_chunk_ids.append(token.id)
                chunks.append((' '.join(current_chunk), current_chunk_ids))
                current_chunk=[]
                current_chunk_ids=[]

                continue
            
            # verb phrase
            if token.upos in ('AUX','VERB') or (token.lemma in ("ne","pas","point") and token.upos=="ADV"):
                while i < len(sentence) and (sentence[i].upos in ('AUX','VERB','PRON') or (sentence[i].lemma in ("ne","pas","point") and sentence[i].upos=="ADV")):
                    token=sentence[i]
                    # ~ print(token.id,token.form, token.lemma)
                    i+=1
                    current_chunk.append(token.form)
                    current_chunk_ids.append(token.id)
                chunks.append((' '.join(current_chunk), current_chunk_ids))
                current_chunk=[]
                current_chunk_ids=[]
                continue
                
            # the preposition or sconj is added to the current chunk
            # the other POS (PUNCT, PRON, SYM, INTJ) are added to the current_chunk
            # which is validated and reinitialized
            if token.upos not in ("ADP","SCONJ"):
                chunks.append((' '.join(current_chunk), current_chunk_ids))
                current_chunk=[]
                current_chunk_ids=[]
        if current_chunk!=[]:
            chunks.append((' '.join(current_chunk), current_chunk_ids))
            current_chunk=[]
            current_chunk_ids=[]
        # ~ print("|".join([ pair[0] for pair in chunks]))
    return chunks



def extract_chunks(sentences):
    chunks = []

    for sentence in sentences:
        # Create a mapping from each token to its head
        head_map = {token.id: token.head for token in sentence}

        for token in sentence:
            # Skip multi-word tokens
            if '-' in token.id:
                continue

            if token.upos == 'NOUN':
                related_tokens_info = []  # List to hold both id and text for related tokens

                # Find all tokens related to the noun (modifiers and the noun itself)
                related_tokens = [t for t in sentence if t.head == token.id or t.id == token.id]
                for related_token in related_tokens:
                    if related_token.deprel in ['det', 'amod', 'nmod', 'obl', 'case'] or related_token.id == token.id:
                        related_tokens_info.append(
                            (int(related_token.id), related_token.form))  # Store id as int for sorting

                # Sort the related tokens by id
                sorted_related_tokens_info = sorted(related_tokens_info, key=lambda x: x[0])

                # Check if there are multiple tokens to form a chunk
                if len(sorted_related_tokens_info) > 1:
                    chunk_text = ' '.join(
                        [info[1] for info in sorted_related_tokens_info])  # Concatenate text components
                    chunk_ids = [str(info[0]) for info in sorted_related_tokens_info]  # Convert ids back to strings
                    chunks.append((chunk_text, chunk_ids))  # Append the tuple with sorted text and ids

            elif token.upos == 'VERB':
                # Initialize the chunk for the verbal clause

                verb_chunk = []  # List to hold both id and text for verb and its dependents

                # Add the verb itself to the verb_chunk list
                verb_chunk.append((int(token.id.split('-')[0]), token.form))  # Convert id to int for sorting

                # Define a range around the verb to include closely surrounding dependents
                # Détermine l'indice de position du verbe dans la phrase, pour Définir
                # la limite supérieure de la plage des tokens à inclure dans le chunk de verbe
                #Cette plage est utilisée pour déterminer quels tokens doivent être considérés
                # comme faisant partie du chunk de verbe. L'idée est d'inclure les tokens qui sont étroitement
                # liés au verbe,généralement ceux immédiatement avant et après lui,
                # pour former une phrase verbale ou une clause cohérente
                verb_index = int(token.id.split('-')[0])  # Handle token range
                min_index = max(1, verb_index - 3)  # Adjust the range as needed
                max_index = min(len(sentence), verb_index + 3)

                # Iterate over tokens within the defined range
                for dependent in sentence:
                    if '-' in dependent.id:
                        continue

                    dependent_index = int(dependent.id.split('-')[0])  # Handle token range
                    if min_index <= dependent_index <= max_index and dependent.head == token.id:
                        # Check for prepositions related to oblique modifiers
                        # Identifies case marking tokens related to the oblique modifier.
                        if dependent.deprel == 'obl': #Is this token functioning as an oblique in its sentence?
                            for child in sentence:
                                if '-' not in child.id and child.head == dependent.id and child.deprel == 'case':
                                    verb_chunk.append((int(child.id.split('-')[0]), child.form))  # Add case marker to verb_chunk
                                    break
                        verb_chunk.append((int(dependent.id.split('-')[0]), dependent.form))  # Add dependent to verb_chunk

                # Sort the verb_chunk by id to ensure correct order
                sorted_verb_chunk = sorted(verb_chunk, key=lambda x: x[0])

                # Extract text and ids from the sorted list
                verb_chunk_text = [item[1] for item in sorted_verb_chunk]
                verb_chunk_ids = [str(item[0]) for item in sorted_verb_chunk]  # Convert ids back to strings

                # Join the chunk components and add to the list
                # combines all the elements of the verb_chunk_text list into a single string,
                # with each element separated by a space.
                # This creates a readable phrase from the individual tokens
                if len(verb_chunk_text) > 1:
                    chunks.append((' '.join(verb_chunk_text), verb_chunk_ids))

            elif token.upos in ['PRON', 'SCONJ', 'ADJ']:
                # Identify the head of the clause (the noun modified by the adjective clause)
                head_noun_id = head_map.get(token.id)
                head_noun = next((t for t in sentence if t.id == head_noun_id), None)

                if head_noun and head_noun.upos == 'NOUN':
                    # Initialize the list to hold both id and text for head noun and adjectival clause components
                    adj_chunk = []

                    # Add the head noun to the adj_chunk list
                    adj_chunk.append((int(head_noun.id), head_noun.form))  # Store id as int for sorting

                    # Add the token initiating the adjectival clause
                    adj_chunk.append((int(token.id), token.form))  # Store id as int for sorting

                    for relative in sentence:
                        if relative.head == token.id and relative.id != token.id:
                            adj_chunk.append((int(relative.id), relative.form))  # Store id as int for sorting

                    # Sort the adj_chunk by id to ensure correct order
                    sorted_adj_chunk = sorted(adj_chunk, key=lambda x: x[0])

                    # Extract text and ids from the sorted list
                    adj_chunk_text = [item[1] for item in sorted_adj_chunk]
                    adj_chunk_ids = [str(item[0]) for item in sorted_adj_chunk]  # Convert ids back to strings

                    # Join the chunk components and add to the list
                    if len(adj_chunk_text) > 1:
                        chunks.append((' '.join(adj_chunk_text), adj_chunk_ids))

    return chunks


# Processes the text, retrieving and parsing its CoNLL-U data, and extracting chunks.
def process_text(text):
    try:
        conllu_data = get_conllu_data(text)
        chunks = extract_chunks(conllu_data)
        words = extract_words(conllu_data)
        return [chunks, words]

    except Exception as e:
        print(f"Error processing text: {e}")
        return [[],[]]  # Return an empty list in case of error

# Main script execution if the script is run directly.
if __name__ == "__main__":
    # Sample text to process
    #text = "ChatGPT est capable de proposer des réponses courtes à des questions longues, de compléter des bonnes phrases françaises, de traduire des textes, d'écrire des articles et de tenir des conversations avec des humains."
    text = "Il navigue ensuite entre deux gigantesques bâtiments d’architecture chinoise, le théâtre et la salle de concert nationaux, puis monte un peu moins d’une centaine de marches pour enfin apercevoir l’immense statue de bronze représentant l’ancien président. Le mémorial surplombe la place. En son sein, la sculpture est gardée sans interruption par deux soldats, la relève s’effectuant toutes les heures, sous le regard généralement curieux des étrangers de passage. C’est cette philosophie imposée par des siècles de luttes sociales et politiques qui fait aujourd’hui l’objet d’une offensive idéologique de grande ampleur à la faveur des impératifs de la construction européenne."
    chunks, words = process_text(text)
    print(chunks)
    for phrase, numbers in chunks:
        numbers_str = ', '.join(numbers)  # Join the numbers into a string separated by commas
        print(f"{phrase}: [{numbers_str}]")
    for word, id in words:
        print(f"{word}: [{id}]")
