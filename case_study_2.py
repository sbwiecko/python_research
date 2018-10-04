# -*- coding: utf-8 -*-
"""
Project Gutenberg is an online repository of publically 
available books in many languages.
All the books have been downloaded as zipped folders from the
edX servers, and extracted into the .\book folder.
The folder contains 4 folders, one for each language (FR, EN, GE and PO)
"""

##### Counting words

text = "This is my test text. We're keeping this text short to keep \
things manageable."

def count_words(text):
	"""
	Count the number of times each word occurs in text (str). Return
	dictionnary where keys are unique words and values are word counts.
	Skip punctuation.
	"""
	text = text.lower()
	skips = ['.', ',', ';',':',"'", '"'] # remove all punctuations
	# punctuations can lead to misleading counting...
	
	for ch in skips:
		text = text.replace(ch, "")
	
	word_counts = {}
	for word in text.split(" "):
		if word in word_counts: # if word already in the dict
			word_counts[word] += 1
		else:
			word_counts[word] = 1 # first instance of the word
	
	return word_counts

#### same with the Counter object from collections
	
from collections import Counter

def count_words_fast(text):
	"""
	Count the number of times each word occurs in text (str). Return
	dictionnary where keys are unique words and values are word counts.
	Skip punctuation.
	"""
	text = text.lower()
	skips = ['.', ',', ';',':',"'", '"'] # remove all punctuations
	# punctuations can lead to misleading counting...
	
	for ch in skips:
		text = text.replace(ch, "")
	
	#word_counts = {}
	#for word in text.split(" "):
	#	if word in word_counts: # if word already in the dict
	#		word_counts[word] += 1
	#	else:
	#		word_counts[word] = 1 # first instance of the word
	
	word_counts = Counter(text.split(" "))
	
	return word_counts # Counter object ~ dictionnary
	# count_words(text) == count_words_fast(text) returns True
	# count_words(text) is count_words_fast(text) returns False


#### Reading a book
def read_book(title_path):
	"""
	Read a book and retunr it as a string.
	"""
	
	with open(title_path, "r", encoding="utf8") as current_file:
		text = current_file.read()
		text = text.replace('\n','').replace('\r', '')
	return text

text = read_book("./books/English/shakespeare/Romeo and Juliet.txt")
print(len(text))

idx = text.find("What's in a name?")
sample_text = text[idx : idx + 500]
print(sample_text)

#### Word frequency statistics

def word_stats(word_counts):
	"""
	Return number of unique words and word frequencies.
	"""
	
	num_unique = len(word_counts) # number of unique words in the dict
	counts = word_counts.values() # list of word counts
	return (num_unique, counts)
	

text = read_book("./books/English/shakespeare/Romeo and Juliet.txt")
(num_unique, counts) = word_stats(count_words(text))
	
print(num_unique, sum(counts))  # show the number of unique words
								# and the total number of words

text = read_book("./books/German/shakespeare/Romeo und Julia.txt")
(num_unique, counts) = word_stats(count_words(text))
	
print(num_unique, sum(counts))
# the German version has less words but more unique words!


#### Reading multiple files
import os
book_dir = "./books"

for language in os.listdir(book_dir): # returns a list
	for author in os.listdir(book_dir + "/" + language): # concaternation:
		for title in os.listdir(
				book_dir + "/" + language + "/" + author):
			inputfile = book_dir + "/" + language + "/" + author + "/" + title # final path
			print(inputfile)
			
			text = read_book(inputfile)
			(num_unique, counts) = word_stats(count_words(text))
			
#### Use Pandas
import pandas as pd

table = pd.DataFrame(columns=['name', 'age'])
table.loc[1] = "James", 22
table.loc[2] = "Jess", 32

table

title_num = 1
stats = pd.DataFrame(columns=['language', 'author', 'title', 'length',
							  'unique'])

for language in os.listdir(book_dir):
	for author in os.listdir(book_dir + "/" + language):
		for title in os.listdir(book_dir + "/" + language + "/" + author):
			inputfile = book_dir + "/" + language + "/" + author + "/" + title
			# final path
			# print(inputfile)
			text = read_book(inputfile)
			(num_unique, counts) = word_stats(count_words(text))
			stats.loc[title_num] = language, author.capitalize, title.replace(
					".txt", ''), sum(counts), num_unique
			title_num += 1

print(stats.head())
print(stats.tail()) # table correctly populated


##### Plotting book statistics

stats['length']

import matplotlib.pyplot as plt

plt.plot(stats['length'], stats['unique'], 'bo')

plt.loglog(stats['length'], stats['unique'], 'bo')

# stratification

stats[stats['language'] == 'English']

len(stats[stats['language'] == 'French'])


plt.figure(figsize=(10,10))
subset = stats[stats['language'] == 'English']
plt.loglog(subset['length'], subset['unique'], 'o',
		   label="English", color="crimson")
subset = stats[stats['language'] == 'French']
plt.loglog(subset['length'], subset['unique'], 'o',
		   label="French", color="forestgreen")
subset = stats[stats['language'] == 'German']
plt.loglog(subset['length'], subset['unique'], 'o',
		   label="German", color="orange")
subset = stats[stats['language'] == 'Portuguese']
plt.loglog(subset['length'], subset['unique'], 'o',
		   label="Portuguese", color="blueviolet")
plt.legend()
plt.xlabel("Book length")
plt.ylabel("Number of unique words")
plt.savefig("../lang_plot.pdf")
###########