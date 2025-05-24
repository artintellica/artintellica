import torch
import matplotlib.pyplot as plt

# Scenario: P(Spam) = 0.4, P(Word|Spam) = 0.7, P(Word|Not Spam) = 0.2
p_spam = torch.tensor(0.4)  # P(Spam)
p_not_spam = 1 - p_spam  # P(Not Spam)
p_word_given_spam = torch.tensor(0.7)  # P(Word|Spam)
p_word_given_not_spam = torch.tensor(0.2)  # P(Word|Not Spam)

# Law of Total Probability: P(Word) = P(Word|Spam)P(Spam) + P(Word|Not Spam)P(Not Spam)
p_word = p_word_given_spam * p_spam + p_word_given_not_spam * p_not_spam

# Bayes' Theorem: P(Spam|Word) = P(Word|Spam)P(Spam) / P(Word)
p_spam_given_word = (p_word_given_spam * p_spam) / p_word

print(f"P(Spam|Word) = {p_spam_given_word.item():.3f}")

# Visualize probabilities
labels = ["P(Spam)", "P(Not Spam)", "P(Word|Spam)", "P(Word|Not Spam)", "P(Spam|Word)"]
values = [
    p_spam.item(),
    p_not_spam.item(),
    p_word_given_spam.item(),
    p_word_given_not_spam.item(),
    p_spam_given_word.item(),
]
plt.bar(labels, values)
plt.title("Probabilities in Spam Classifier")
plt.show()
