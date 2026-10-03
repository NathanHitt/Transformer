import regex as re
import heapq
from collections import defaultdict

test_string = "hello! こんにちは"
# converts a unicode character into a sequence of bytes
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(type(utf8_encoded))

#unocide code points (21-bit integers with 159,801 valid values)
print(list(utf8_encoded))


# One byte does not necessarily correspnd to one Unicode character:
print(len(test_string))
print(len(utf8_encoded))

# Decode 
# (convert sequence of coe points to integers in range 0 -255, much more reasonable vocabulary size)
print(utf8_encoded.decode("utf-8"))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
print(re.findall(PAT, "some text that i'll pre-tokenize"))


# we should use re.finditer in our code instead to avoid storing pre-tokenized words as we construct mapping



# simple tokenizer:
from collections import Counter

corpus = """low low low low low lower lower widest widest widest newest newest newest newest newest newest """

# pretokenize: 
pre_tokenized = corpus.split(" ")
print(pre_tokenized)
# count = Counter(pre_tokenized)
# print(count)


# transform into bytes
count = []
for token in pre_tokenized:
    temp = []
    for char in token:
        temp.append(char.encode('utf-8'))

    temp = tuple(temp)
    count.append(temp)
        


pre_tokens = Counter(count)
print(pre_tokens)


#def pre_tokenize(corpus)



# for byte, c in count.items():
#     pre_tokens[byte] = -c

#You can add bytes to make pairs
#print(count[0][0] + count[0][1])



# # derive pair frequency counts and \

pair_counter = defaultdict(int)
for item, count in pre_tokens.items():


    for index in range(0, len(item) - 1, 1):
        bp1, bp2 = item[index], item[index + 1]
        pair_counter[bp1 + bp2] += count


max_bp = None
for bp, count in pair_counter.items():

    if (max_bp == None) or count > max_bp[1] or (count == max_bp[1] and bp > max_bp[0]):
        max_bp = (bp, count)


print(max_bp)
max_bp = max_bp[0]


new_pre_token_to_freq = {}

# construct a new counter object with all the valid items + the merged ones
for pre_token, count in pre_tokens.items():
    new_pre_token = []
    index = 0
    while index < len(pre_token) - 1:
        bp1, bp2 = pre_token[index], pre_token[index + 1]
        
        if max_bp != bp1 + bp2:
            new_pre_token.append(bp1)
            index += 1
        else:
            new_pre_token.append(max_bp)
            index += 2

    # check if theres still an element left
    if index == len(pre_token) - 1:
        new_pre_token.append(pre_token[index])

    new_pre_token_to_freq[tuple(new_pre_token)] = count

print("New pre token to frequency: ", new_pre_token_to_freq)
            


    





