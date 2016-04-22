#!/usr/bin/env python

import json

SENTENCE_PAIR_DATA = True

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

def convert_binary_bracketing(parse):
    transitions = []
    tokens = []
    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                tokens.append(word.lower())
                transitions.append(0)
    return tokens, transitions

def load_data(path):
    print "Loading", path
    examples = []
    with open(path, 'r') as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            (example["premise_tokens"], example["premise_transitions"]) = convert_binary_bracketing(loaded_example["sentence1_binary_parse"])
            (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert_binary_bracketing(loaded_example["sentence2_binary_parse"])
            examples.append(example)
    print "Processed {0} examples".format(len(examples))
    return examples, None


def loadExampleSentences(path):
    with open(path, 'r') as f:
        examples = []
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            premiseTokens, _ = convert_binary_bracketing(loaded_example["sentence1_binary_parse"])
            hypothesisTokens, _ = convert_binary_bracketing(loaded_example["sentence2_binary_parse"])
            example = [premiseTokens, hypothesisTokens]
            examples.append(example)
    print "Processed {0} examples".format(len(examples))
    return examples


def loadExampleLabels(path):
    print "Loading gold labels from {0}".format(path)
    with open(path, 'r') as f:
        examplesLabels = []
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            label = loaded_example["gold_label"]
            examplesLabels.append(label)
    return examplesLabels


if __name__ == "__main__":
    pass
    # Demo:
    #examples = load_data('snli-data/snli_1.0_dev.jsonl')
    #print examples[0]
