def segmentation_accuracy(path, words):
    text = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                length = len(word)
                for i in range(len(word) - 1):
                    if word[i] == "ل" and word[i + 1] == "ا":
                        print("lam alef", word)
                        length -= 1
                text.append(length)

    if len(text) == len(words):
        print("word segmentation accuracy=100%")
        correct = 0
        for i in range(len(words)):
            if len(words[i]) - 1 == text[i]:
                correct += 1
        print("character segmentation accuracy=", (correct / len(words)) * 100)
    else:
        print("word segmentation failed", len(text), len(words))

