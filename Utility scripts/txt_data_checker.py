#!/usr/bin/python

import sys

maxwordcount = 0
maxwordfile = ""

totalwordcount = 0

if len(sys.argv) <= 1 :
    print("Please enter file names as arguments")
else:
    file_list = sys.argv
    
    file_list = file_list[1:]

    for filename in file_list:

        f = open(filename, "r", encoding="utf-8")

        linecount = 0
        wordcount = 0
        charactercount = 0

        for line in f:
            wordlist = line.split()
            linecount += 1
            wordcount += len(wordlist)
            charactercount += sum(len(word) for word in wordlist)

        if maxwordcount < wordcount:
            maxwordcount = wordcount
            maxwordfile = filename

        totalwordcount += wordcount

        print(f'\n{filename}')
        print(f'{linecount} lines')
        print(f'{wordcount} words')
        print(f'{charactercount} characters\n')

        f.close()

    print("----------")
    print(f"Max word count: {maxwordcount} in file {maxwordfile}")
    print(f"Total: {totalwordcount} words")
