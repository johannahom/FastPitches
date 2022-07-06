from common.text.text_processing import TextProcessing
from common.utils import load_filepaths_and_text
import sys
import torch

'''
Written by Johannah O'Mahony @ CSTR University of Edinburgh

This script reads in the metadata file, counts words and counts labels
and provides a list of file in which the two don't match
'''

files = load_filepaths_and_text(sys.argv[0], dataset_path=None)
log = open("label_validation.log")
tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, get_counts=True)
non_words = ['#','%','&','*','+','-','/','[',']','(',')','!','\\','\'','\"','(',')', ' ', '.', '?', ',', ', ', '. ', '! ', '? ', '., ', '?, ', ',, ']


for file in files:

    #load CWT array and get len

    cwt_words = len(cwt_array)
    #transcribe text
    text, text_info = self.tp.encode_text(text)
    text_info = [x for x in text_info if x[0] not in non_words]
    fp_words = len(text_info)
    if fp_words != cwt_array:
       log.write(f"{filename}\t{len(cwt_array}\t{fp_words}\t{text}\n")

