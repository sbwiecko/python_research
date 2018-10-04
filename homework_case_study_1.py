# -*- coding: utf-8 -*-
"""
Homework: Case Study 1
A cipher is a secret code for a language.  In this case study, 
we will explore a cipher that is reported by contemporary Greek 
historians to have been used by Julius Caesar to send secret 
messages to generals during times of war.

https://courses.edx.org/courses/course-v1:HarvardX+PH526x+1T2018
"""

"""
The Caesar cipher shifts each letter of a message to another letter in the 
alphabet located a fixed distance from the original letter. If our 
encryption key were 1, we would shift h to the next letter i, i to the next 
letter j, and so on. If we reach the end of the alphabet, which for us is 
the space character, we simply loop back to a. To decode the message, we 
make a similar shift, except we move the same number of steps backwards in 
the alphabet
"""

##### EXERCICE 1
"""
Create a string called alphabet consisting of the lowercase letters 
of the space character space ' ', concatenated with string.ascii_lowercase 
at the end
"""

import string
alphabet = " " + string.ascii_lowercase


##### EXERCICE 2
"""
Create a dictionary with keys consisting of the characters in alphabet, 
and values consisting of the numbers from 0 to 26. Store this as positions.
"""

positions = {pos: char for char, pos in enumerate(alphabet)}
# or create a loop - idx=0;for char in alphabet: positions[char]=idx;idx+=1

##### EXERCICE 3
"""
Use positions to create an encoded message based on message where each 
character in message has been shifted forward by 1 position, as defined 
by positions. Note that you can ensure the result remains within 0-26 
using result % 27. Store this as encoded_message.
"""

message = "hi my name is caesarz"
encoded_message = ""

for char in message:
	pos = positions[char]
	encoded_message += alphabet[(pos+1)%27]
	#the modulo ensures we stay in the 0-26 range


##### EXERCICE 4
"""
Modify this code to define a function encoding that takes a message 
as input as well as an int encryption key to encode a message with 
the Caesar cipher by shifting each letter in message by key positions.
Your function should return a string consisting of these encoded letters.
Use encode to encode message using key = 3, and save the result as 
encoded_message. Print encoded_message.
"""

def encoding(input_message, key=1):
	encoded_message = ""

	for char in input_message:
		pos = positions[char]
		encoded_message += alphabet[(pos + key)%27]
	return encoded_message

print(encoding(message, key=3))

##### EXERICE 5
"""
Use encoding to decode encoded_message using key = -3. Store your 
decoded message as decoded_message. Print decoded_message. 
Does this recover your original message?
"""

encoded_message = encoding(message, 3)

decoded_message = encoding(encoded_message, -3)
print(decoded_message)
# incredible that when we index with -1, we index position 26
###################################