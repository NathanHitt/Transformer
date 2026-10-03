import regex as re
import heapq
from collections import Counter
from collections import defaultdict
from find_chunk_boundaries import find_chunk_boundaries
from multiprocessing import Pool

def pre_tokenize(corpus: str, pre_tokens, pattern):
    matches = re.finditer(pattern, corpus)

    for match in matches:
        temp = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        pre_tokens[temp] += 1

    return pre_tokens


def construct_pair_counter(pre_tokens_to_freq):
    pair_counter = defaultdict(int)
    
    for item, count in pre_tokens_to_freq.items():

        # count up co-occuring pairs
        for index in range(0, len(item) - 1):
            bp1, bp2 = item[index], item[index + 1]
            pair_counter[(bp1, bp2)] += count

    return pair_counter


def find_max_bp(pair_counter):
    max_bp = None
    for bp, count in pair_counter.items():
        if (max_bp == None) or count > max_bp[1] or (count == max_bp[1] and bp > max_bp[0]):
            max_bp = (bp, count)

    max_bp = max_bp[0]

    return max_bp

def perform_train(pre_tokens_to_freq, pair_counter):

    max_bp = find_max_bp(pair_counter)

    # perform merge
    new_pre_token_to_freq = defaultdict(int)

    for pre_token, count in pre_tokens_to_freq.items():
        new_pre_token = []
        index = 0
        while index < len(pre_token):

            # edge case
            if index == len(pre_token) - 1:
                new_pre_token.append(pre_token[index])
                break

            bp1, bp2 = pre_token[index], pre_token[index + 1]

            # base case
            if max_bp != (bp1, bp2):
                new_pre_token.append(bp1)
                index += 1
            else:
                merge_token = bp1 + bp2

                # first update pair counter
                if index - 1 >= 0:
                    pair_counter[(pre_token[index - 1], bp1)] -= count
                    pair_counter[(pre_token[index - 1], merge_token)] += count

                if index + 2 < len(pre_token):
                    pair_counter[(bp2, pre_token[index + 2])] -= count
                    pair_counter[(merge_token, pre_token[index + 2])] += count

                pair_counter[(bp1, bp2)] -= count

                new_pre_token.append(merge_token)
                index += 2

        new_pre_token_to_freq[tuple(new_pre_token)] += count

    return new_pre_token_to_freq, max_bp, pair_counter

def pretokenize_chunk(start, end, path, pattern, special_pattern):

    with open(path, 'rb') as f:
        f.seek(start)
        chunks = f.read(end - start).decode("utf-8", errors='ignore')

    sub_chunks= re.split(special_pattern, chunks)
    pre_tokens = defaultdict(int)
    
    for sub_chunk in sub_chunks: 

        # update global pre_tokens frequency dict:
        pre_tokens = pre_tokenize(sub_chunk, pre_tokens, pattern)

    return pre_tokens


#corpus = """low low low low low lower lower widest widest widest newest newest newest newest newest newest """
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SIMPLE_PAT = r"\S+"
PATH = "/Users/nathanhittesdorf/Desktop/CS336 Language Modeling/assignment1-basics/data/tiny_ex.txt"
special_tokens = ["<|endoftext|>"]
num_processes = 4

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], pattern = PAT, num_processes =4):

    pre_tokens = defaultdict(int)
    special_pattern = "|".join(re.escape(token) for token in special_tokens)

    # Load the data into non overlapping chunks, process those
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, desired_num_chunks=num_processes, split_special_token=b"<|endoftext|>")
        cutoffs = list(zip(boundaries[:-1], boundaries[1:])) 

    # Initate per chunk BPE
    with Pool(num_processes) as p:
        jobs = []

        for start, end in cutoffs:
            jobs.append(p.apply_async(pretokenize_chunk, args = (start, end, input_path, pattern, special_pattern)))

        results = [job.get() for job in jobs]

    pre_tokens = Counter()

    for result in results:
        pre_tokens.update(result)

    pre_tokens = defaultdict(int, pre_tokens)

    # initialize pair_counter
    pair_counter = construct_pair_counter(pre_tokens)

    # lets create the initial vocabulary, a dit of [int, byte] mapping from token ID to bytes
    vocab = {}

    for num in range(0, 256):
        vocab[num] = bytes([num])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode()

    merges = []

    #print(f"Initial Pretokens map: {pre_tokens}")

    while len(vocab) < vocab_size:


        pre_tokens, max_bp, pair_counter = perform_train(pre_tokens, pair_counter)
        
        bp1, bp2 = max_bp
        vocab[len(vocab)] = bp1 + bp2
        # append BPE merges produced for training in order of creation
        merges.append(max_bp)

    return vocab, merges




if __name__ == '__main__':
    vocab, merges = train_bpe(input_path = PATH, vocab_size = 258, special_tokens = special_tokens, pattern=SIMPLE_PAT)
    print(vocab)
    print(merges)