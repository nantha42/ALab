#import torch
import textblob
from icecream import ic
from esfunc import *
import csv

import re 
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")
cornell = "../../dataset/cornell/"

lines = {}
def load_lines():
	broken = 0
	filename =  cornell + "movie_lines.txt"
	with open(filename,"r") as hand:
		line = hand.readline()
		while line != "":
			record = line.split("+++$+++")
			id_ = record[0][:-1]
			line = record[-1]
			lines[id_] = line
			try:
				line = hand.readline()
			except:
				broken+=1
	


movies = {}
broken = 0
def load_movies():
	broken = 0
	filename = cornell+"movie_titles_metadata.txt"
	with open(filename,"r") as hand:
		line = hand.readline()
		while line != "":
			record = line.split("+++$+++")
			id_ = record[0]
			id_ = id_[:-1]
			if id_[0] == 'm':
				moviename = record[1]
				movies[id_] = moviename 
	
			try:
				line = hand.readline()
			except:
				broken+=1
				line = hand.readline()
	


movie_conversations = {}
broken = 0
count = 0;

def load_conversations():
	broken = 0

	filename = cornell+"movie_conversations.txt"
	for movie in movies.keys():
		movie_conversations[movie] = []
	
	with open(filename,"r") as hand:
		line = hand.readline()
		while line != "":
			record = line.split("+++$+++")
			id_ = record[-2]
			id_ = id_.split(" ")[1]
			conversation = record[-1].split("'")
			convs = []
			for c in conversation:
				if c[0] == 'L':
					convs.append(c)
			conversation = convs
			try:
				movie_conversations[id_].append(conversation)	
			except:
				movie_conversations[id_] = [conversation]	
	
			try:
				line = hand.readline()
			except:
				broken+=1
	
def get_words(movie_id):
	words = []
	for conv in movie_conversations[movie_id]:
		for l in conv:
			try:
				tokens = tokenizer(lines[l])
				words.extend(tokens)
			except:
				pass
	return words	


if __name__ == '__main__':
	load_lines()
	load_movies()
	load_conversations()

	words = []
	for movie in movies.keys():
		words.extend(get_words(movie))	

	print(len(list(set(words))))
	words = []
	words.extend(get_words('m1'))
	print(words[:100])
	print(len(list(set(words))))

	
