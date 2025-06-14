import os
import sys
import re
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
#~ import spacy

# TODO : CESalign format not supported yet (only txt and tsv)

# reading the command line arguments
parser = argparse.ArgumentParser(
	prog='eval_txt',
	formatter_class=argparse.RawDescriptionHelpFormatter,
	description='''\
A program that computes P,R and F score for a given pair of aligned files (SYS), compared to a given pair of reference aligned files (REF). 
	
Input : the corresponding files should be named following this pattern : NAME..*.ref.L1.txt, NAME..*.ref.L2.txt, NAME..*.L1.txt, NAME..*.L2.txt
	
For txt format, files should be formatted as raw utf8 text with one segment per line.
For txt2 format, files should be formatted as raw utf8 text with one segment per line, with both languages in the same file :
	Each corresponding pair is encoded by the sequence of three liens :
		- l1 segment
		- l2 segment
		- empty line.
	refCol1, refCol2, sysCol1, sysCol2 parameters indicates 0 for l1, and 1 for l2 lines
For tsv format, the aligned segments should appear on a same line, separated by tabulation. In this format
refCol1, refCol2, sysCol1, sysCol2 parameters indicates the column number to extraction 
corresponding language (first col=0)

'''
)

parser.add_argument('--l1', type=str, help='The source language (ISO : ex. "en" for English)', default='zh')
parser.add_argument('--l2', type=str, help='The target language (ISO : ex. "fr" for French)', default='fr')
parser.add_argument('--incompleteSys', help='The system alignment is not complete regarding the reference alignment (no synchronization is required)',action="store_true",default=False)

parser.add_argument('--logFile', type=str, help='The name of the log file to write the results (tsv format)', default='./evaluate.log')
parser.add_argument('--logFileEntry', type=str, help='The label of the new entry in the log file.', default='*')

parser.add_argument('--refFile1',help='Reference filename for language 1',type=str,default="")
parser.add_argument('--refFile2',help='Reference filename for language 2',type=str,default="")
parser.add_argument('--refFileFormat1',help='Format of reference filename for language 1 (tsv|txt|ces|txt2|tmx)',type=str,default="txt")
parser.add_argument('--refFileFormat2',help='Format of reference filename for language 1 (tsv|txt|ces|txt2|tmx)',type=str,default="txt")
parser.add_argument('--n4txt2',help='In txt2 format, indicates how lines are used for each group',type=int,default=3)
parser.add_argument('--refCol1',help='Column number for reference filename for language 1 (format tsv or ces)',type=int,default=1)
parser.add_argument('--refCol2',help='Column number for reference filename for language 2 (format tsv or ces)',type=int,default=2)


parser.add_argument('--sysFile1',help='System filename for language 1',type=str,default="")
parser.add_argument('--sysFile2',help='System filename for language 2',type=str,default="")
parser.add_argument('--sysFileFormat1',help='Format of system filename for language 1 (tsv|txt)',type=str,default="txt")
parser.add_argument('--sysFileFormat2',help='Format of system filename for language 1 (tsv|txt)',type=str,default="txt")
parser.add_argument('--sysCol1',help='Column number for system filename for language 1 (format tsv or ces)',type=int,default=1)
parser.add_argument('--sysCol2',help='Column number for system filename for language 2 (format tsv or ces)',type=int,default=2)

parser.add_argument('--verbose',help='Print trace',action="store_true",default=False)
parser.add_argument('--printToks',help='Print tokens with * when erroneous',action="store_true",default=False)

nlp={}
#~ nlp['zh'] = spacy.load("zh_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
#~ nlp['fr'] = spacy.load("fr_core_news_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

emptySegment=1

args = parser.parse_args()

l1=args.l1
l2=args.l2
incompleteSys=args.incompleteSys

logFile=args.logFile
logFileEntry=args.logFileEntry

refFile1=args.refFile1
refFile2=args.refFile2
refFileFormat1=args.refFileFormat1
refFileFormat2=args.refFileFormat2
n4txt2=args.n4txt2
refCol1=args.refCol1
refCol2=args.refCol2

sysFile1=args.sysFile1
sysFile2=args.sysFile2
sysFileFormat1=args.sysFileFormat1
sysFileFormat2=args.sysFileFormat2
sysCol1=args.sysCol1
sysCol2=args.sysCol2

verbose=args.verbose
printToks=args.printToks
printToks=True


##################################################################################"
# functions
def tokenize(s:str,lang='fr',nlp=None):
	if nlp:
		doc = nlp(s)
		tokens=[token.text for token in doc]
		return tokens
	else:
		if lang=="zh" :
			return [tok for tok in re.split(r"",s) if tok !="" and not re.match(r"^\s+$",tok)] 
		return [tok for tok in re.split(r'\s|\b|(?<=\W)',s) if tok !=""]

def readInputFile(inputFile,inputFormat,column=0,language="fr"):
	"""Reads an input file and returns a list of segments.

	  Args:
		inputFile: The name of the input file.
		inputFormat: The format of the input file.
		column: The column number of the input file that contains the text.
		language: The language of the input file.

	  Returns:
		A list of segments.
	"""
	global segMinLength
	
	segs=["<start>"]
	idSegs=[0]
	lenSegs=1

	try:
		if verbose: 
			print("Reading",inputFile)
		f = open(inputFile, encoding='utf8')
	except:
		print(sys.exc_info()[0]," A problem occurred while opening", inputFile)

		sys.exit()

	# Reading according to inputFormat
	if inputFormat == "arc" or inputFormat == "ces":
		for line in f:
			line = line.strip()
			m = re.search(r'<s\b[^>]+id="([^"]*)"', line)
			if m and m.group(1):
				idSegs.append(m.group(1))
			else:
				idSegs.append("s"+str(lenSegs))
			segs.append(line)
			lenSegs += 1

	# The json input contains a sentences property, which is a list sentences, which are list of tokens
	# Each token is a list of conll features, col1->form, col9=blank_space
	elif inputFormat == "json":
		content = f.read()
		jsonObj = json.loads(content)
		segs = [
			"".join([tok[column] + " " for tok in seg[0]]) for seg in jsonObj["sentences"]
		]
		idSegs=list(range(0,len(segs)+1))
		
	elif inputFormat == "tsv":
		segs = []
		for line in f:
			alignedSegs=re.split("\t",line)
			if len(alignedSegs)>column:
				segs.append(alignedSegs[column])
			else:
				print("Ill formated line:",line)

	elif inputFormat == "xml-conll":
		content = f.read()
		try:
			xmlRoot = ET.fromstring(content)
		except:
			print("non conform XML :",os.path.join(inputDir, inputFile))
			error_log.write("non conform XML :",os.path.join(inputDir, inputFile),"\n")
			sys.exit() 

		for sElt in xmlRoot.findall('.//s'):
			s="".join(sElt.itertext())
			toks=[]
			for line in re.split(r"\n",s):
				cols=re.split("\t",line)
				if len(cols)==10:
					toks.append(cols[1])
			seg=" ".join(toks)
			segs.append(seg)
			if sElt.attrib["id"]:
				idSegs.append(sElt.attrib["id"])
			else:
				idSegs.append(len(segs)-1)
	elif inputFormat == "tmx":
		content = f.read()
		# elimination of xml namespace
		content = re.sub("xml:lang","lang",content)
		try:
			xmlRoot = ET.fromstring(content)
		except:
			print("non conform XML :",os.path.join(inputDir, inputFile))
			error_log.write("non conform XML :",os.path.join(inputDir, inputFile),"\n")
			sys.exit() 

		for tu in xmlRoot.findall('.//tu'):
			seg=""
			tuvs=tu.findall('.//tuv[@lang="'+language+'"]')
			if (len(tuvs)==0):
				tuvs=tu.findall('.//tuv[@lang="'+language.upper()+'"]')
			for tuv in tuvs:
				seg+= "".join(tuv.itertext())
			segs.append(seg)
			idSegs.append(len(segs)-1)
	elif inputFormat == "txt2":
		for i,line in enumerate(f):
			line=line.strip()
			if i % n4txt2== column:
				segs.append(line)

		idSegs=list(range(0,len(segs)+1))
	# Default format: one segment per line
	else:
		for line in f:
			line=line.strip()
			segs.append(line)
		idSegs=list(range(0,len(segs)+1))
	
	f.close()
	return (segs,idSegs)

######################################################################################
# load testParam
logFileEntry="ailign2 (minDensityRatio=0.3,labseTheshold=0.4, margin=0.05, distNull=1,penalty_n_n=0.06,progressive penalty,useEncoder=False)"
(sysFile1,sysFileFormat1,sysCol1,l1)=("/home/kraifo/Documents/Projets de recherche/2022 - ACR GRIMM/corpus/5. aligned/tests/LeClanduSorghoRouge.fr-zh.txt.txt","txt2",1,"zh")
(sysFile2,sysFileFormat2,sysCol2,l2)=("/home/kraifo/Documents/Projets de recherche/2022 - ACR GRIMM/corpus/5. aligned/tests/LeClanduSorghoRouge.fr-zh.txt.txt","txt2",0,"fr")


if __name__=="__main__":
	
	# reading files
	(refSegs1,refIdSegs1)=readInputFile(refFile1,refFileFormat1,refCol1,l1)
	(refSegs2,refIdSegs2)=readInputFile(refFile2,refFileFormat2,refCol2,l2)
	(sysSegs1,sysIdSegs1)=readInputFile(sysFile1,sysFileFormat1,sysCol1,l1)
	(sysSegs2,sysIdSegs2)=readInputFile(sysFile2,sysFileFormat2,sysCol2,l2)
	
	print("Ref:",len(refSegs1),"segments pour",l1)
	print("Ref:",len(refSegs2),"segments pour",l2)
	print("Sys:",len(sysSegs1),"segments pour",l1)
	print("Sys:",len(sysSegs2),"segments pour",l2)
	
	if len(refSegs1)!=len(refSegs2):
		print("Error : Segment numbers do not match for reference files")
		sys.exit()
		
	if len(sysSegs1)!=len(sysSegs2):
		print("Error : Segment numbers do not match for system files")
		sys.exit()
	
	refFullText1=re.sub(r"\s","","".join(refSegs1))
	refFullText2=re.sub(r"\s","","".join(refSegs2))
	sysFullText1=re.sub(r"\s","","".join(sysSegs1))
	sysFullText2=re.sub(r"\s","","".join(sysSegs2))
	
	# testing if the sys files are complete
	if not incompleteSys:
		if refFullText1 != sysFullText1:
			print("Error : Synchronization problem : the full text must correspond exactly")
			i=0
			while refFullText1[i:i+100]==sysFullText1[i:i+100] and i+100<len(refFullText1):
				i=i+50
			print (i,refFullText1[i:i+100])
			print (i,sysFullText1[i:i+100])
			
			sys.exit()
		if refFullText2 != sysFullText2:
			print("Error : Synchronization problem : the full text must correspond exactly")
			i=0
			while refFullText2[i:i+100]==sysFullText2[i:i+100] and i+100<len(refFullText2):
				i=i+50
			print (i,refFullText2[i:i+100])
			print (i,sysFullText2[i:i+100])
			sys.exit()		
	
	############################################## reference alignment
	ref=set()
	
	# when sys files are not complete we need to do synchronization according to refTokens
	if incompleteSys:
		refTokens1=[]
		refTokens2=[]
	
	offsetTok1=0
	offsetTok2=0
	refX=[] # coordinates of first tok of each sentence1
	refY=[] # coordinates of first tok of each sentence2
	for i in range(len(refSegs1)):
		# adding points for plot 
		refX.append(offsetTok1)
		refY.append(offsetTok2)
		# reading and tokenizing segment
		seg1=refSegs1[i]
		seg2=refSegs2[i]
		toks1=tokenize(seg1,l1)
		toks2=tokenize(seg2,l2)
		
		if incompleteSys:
			refTokens1.extend(toks1)
			refTokens2.extend(toks2)
		
		# aligning with empty token
		if emptySegment:
			if toks1==[]:
				for pos2,tok2 in enumerate(toks2):
					key="-"+str(pos2+offsetTok2)
					ref.add(key)
			if toks2==[]:
				for pos1,tok1 in enumerate(toks1):
					key=str(pos1+offsetTok1)+"-"
					ref.add(key) 
		
		for pos1,tok1 in enumerate(toks1):
			for pos2,tok2 in enumerate(toks2):
				key=str(pos1+offsetTok1)+"-"+str(pos2+offsetTok2)
				ref.add(key)
		
		offsetTok1+=len(toks1)
		offsetTok2+=len(toks2)
	
	refNbTok1=offsetTok1
	refNbTok2=offsetTok2
	
	# plot
	plt.axis([1,offsetTok1,1,offsetTok2])
	plt.title(logFileEntry)
	plt.scatter(refX,refY,c="black",s=1)
	
	
	################################################ computing intersection with system alignment (at token level)
	ok1=set()
	ok2=set()
	offsetTok1=0
	offsetTok2=0
	sysNbTok1=0
	sysNbTok2=0
	sysXOk=[] # coordinates of first tok of each sys sentence1 (when ok)
	sysYOk=[] # coordinates of first tok of each sys sentence2 (when ok)
	sysXWr=[] # coordinates of first tok of each sys sentence1 (when wrong)
	sysYWr=[] # coordinates of first tok of each sys sentence1 (when wrong)

	for i in range(len(sysSegs1)):
		seg1=sysSegs1[i]
		seg2=sysSegs2[i]
		toks1=tokenize(seg1,l1)
		toks2=tokenize(seg2,l2)
		
		# synchronization toks1, toks2 must be searched through refTokens1 and refTokens2
		if incompleteSys:
			while refTokens1[offsetTok1:offsetTok1+len(toks1)] != toks1 and offsetTok1+len(toks1)<len(refTokens1):
				offsetTok1+=1
			while refTokens2[offsetTok2:offsetTok2+len(toks2)] != toks2 and offsetTok2+len(toks2)<len(refTokens2):
				offsetTok2+=1
			if  refTokens1[offsetTok1:offsetTok1+len(toks1)] != toks1:
				offsetTok1=0
				while refTokens1[offsetTok1:offsetTok1+len(toks1)] != toks1 and offsetTok1+len(toks1)<len(refTokens1):
					offsetTok1+=1
				if  refTokens1[offsetTok1:offsetTok1+len(toks1)] != toks1:
					print("Error : Synchronization has failed with",toks1)
					sys.exit()
				else:
					if verbose: 
						print("Rewinding for synchronization offsetTok1=",offsetTok1)
			if  refTokens2[offsetTok2:offsetTok2+len(toks2)] != toks2:
				offsetTok2=0
				while refTokens2[offsetTok2:offsetTok2+len(toks2)] != toks2 and offsetTok2+len(toks2)<len(refTokens2):
					offsetTok2+=1
				if  refTokens2[offsetTok2:offsetTok2+len(toks2)] != toks2:
					print("Error : Synchronization has failed with",toks2)
					sys.exit()
				else:
					if verbose:
						print("Rewinding for synchronization to offsetTok2=",offsetTok2)
		if verbose:
			print(f"offsetTok1={offsetTok1}, offsetTok2={offsetTok2}") 
		
		# aligning with empty token
		if emptySegment:
			if toks1==[]:
				for pos2,tok2 in enumerate(toks2):
					key="-"+str(pos2+offsetTok2)
					if key in ref:
						ok2.add(pos2+offsetTok2)
			if toks2==[]:
				for pos1,tok1 in enumerate(toks1):
					key=str(pos1+offsetTok1)+"-"
					if key in ref:
						ok1.add(pos1+offsetTok1)
		
		# adding non empty correspondances
		for pos1,tok1 in enumerate(toks1):
			for pos2,tok2 in enumerate(toks2):
				key=str(pos1+offsetTok1)+"-"+str(pos2+offsetTok2)
				if key in ref:
					ok1.add(pos1+offsetTok1)
					ok2.add(pos2+offsetTok2)
					if pos1==0 and pos2==0:
						sysXOk.append(offsetTok1)
						sysYOk.append(offsetTok2)
				elif pos1==0 and pos2==0:
					sysXWr.append(offsetTok1)
					sysYWr.append(offsetTok2)
		
		offsetTok1+=len(toks1)
		offsetTok2+=len(toks2)
		sysNbTok1+=len(toks1)
		sysNbTok2+=len(toks2)
	
	if printToks:
		offsetTok1=0
		offsetTok2=0
		f=open(sysFile1+".eval.tsv",mode="w",encoding="utf8")
		for i in range(len(sysSegs1)):
			seg1=sysSegs1[i]
			seg2=sysSegs2[i]
			toks1=tokenize(seg1,l1)
			toks2=tokenize(seg2,l2)
			for tok in toks1:
				star=""
				if offsetTok1 not in ok1:
					star="*"
				f.write(tok+star+" ")
				offsetTok1+=1
			f.write("\t")
			for tok in toks2:
				star=""
				if offsetTok2 not in ok2:
					star="*"
				f.write(tok+star+" ")
				offsetTok2+=1
			f.write("\n")
		f.close()
	
	plt.scatter(sysXOk,sysYOk,c="green",s=1)
	plt.scatter(sysXWr,sysYWr,c="red",s=1)
	
	
	# final p,r,f computation
	p1=len(ok1)/sysNbTok1
	r1=len(ok1)/refNbTok1
	f1=2*p1*r1/(p1+r1)
	
	p2=len(ok2)/sysNbTok2
	r2=len(ok2)/refNbTok2
	f2=2*p2*r2/(p2+r2)
	
	fineness1=len(sysSegs1)/len(refSegs1)
	fineness2=len(sysSegs2)/len(refSegs2)
	
	if refNbTok1==sysNbTok1 and refNbTok2==sysNbTok2:
		print (f"{l1} : {refNbTok1} tokens dans REF et dans SYS")
		print (f"{l2} : {refNbTok2} tokens dans REF et dans SYS")
	else:
		print (f"{l1} : {refNbTok1} tokens dans REF {sysNbTok1} tokens dans SYS")
		print (f"{l2} : {refNbTok2} tokens dans REF {sysNbTok2} tokens dans SYS")
		if not incompleteSys:
			print ("Error : tokens are not synchronized")
			for i in range(refNbTok2):
				print (refTokens[i],sysTokens[i])
				if refTokens[i]!=sysTokens[i]:
					print (refTokens[i+1],sysTokens[i+1])
					print (refTokens[i+2],sysTokens[i+2])
					print (refTokens[i+3],sysTokens[i+3])
					sys.exit()
			sys.exit()
	
	
	print (f"{l1} : P={p1:.4f} R={r1:.4f} F={f1:.4f} fineness={fineness1:.4f}")
	print (f"{l2} : P={p2:.4f} R={r2:.4f} F={f2:.4f} fineness={fineness2:.4f}")
	log=open(logFile,mode="a",encoding="utf8")
	log.write("\t".join([logFileEntry,l1,str(p1),str(r1),str(f1),str(fineness1)])+"\n")
	log.write("\t".join([logFileEntry,l2,str(p2),str(r2),str(f2),str(fineness2)])+"\n")
	log.close()
	plt.savefig(sysFile1+'.eval.png')
	plt.show()
