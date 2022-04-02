from collections import defaultdict
import numpy as np


class WordNgramModel:
    def __init__(self, n):
        self.n = n
        self.d = defaultdict(lambda: defaultdict(int))

    def train(self, train_file):
        with open(train_file) as f:
            words = f.read().split()

        for i in range(self.n + 1, len(words)):
            context = " ".join(words[i - self.n : i])
            current = words[i]
            self.d[context][current] += 1

    def sample(self, context):
        counts = self.d[context]
        counts_sum = sum(counts.values())
        probs = [x / counts_sum for x in counts.values()]
        return np.random.choice(list(counts.keys()), p=probs)

    def generate(self, prefix, length):
        outputs = [prefix]
        for i in range(length):
            new_word = self.sample(prefix)
            outputs.append(new_word)
            prefix = " ".join(prefix.split()[1:]) + " " + new_word
        return " ".join(outputs)


class CharNgramModel:
    def __init__(self, n):
        self.n = n
        self.d = defaultdict(lambda: defaultdict(int))

    def train(self, train_file):
        with open(train_file) as f:
            text = f.read()

        for i in range(self.n + 1, len(text)):
            context = text[i - self.n : i]
            current = text[i]
            self.d[context][current] += 1

    def sample(self, context):
        counts = self.d[context]
        counts_sum = sum(counts.values())
        probs = [x / counts_sum for x in counts.values()]
        return np.random.choice(list(counts.keys()), p=probs)

    def generate(self, prefix, length):
        outputs = [prefix]
        for i in range(length):
            new_char = self.sample(prefix)
            outputs.append(new_char)
            prefix = prefix[1:] + new_char
        return "".join(outputs)


if __name__ == "__main__":

    model = CharNgramModel(8)
    model.train("shakespeare.txt")
    print(model.generate("SCENE VI", 500))
