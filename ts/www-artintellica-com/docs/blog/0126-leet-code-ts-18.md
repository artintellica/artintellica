+++
title = "LeetCode in TypeScript, Part 8: Word Break"
author = "Artintellica"
date = "2025-06-24"
code = "https://github.com/artintellica/artintellica/tree/main/ts/blog/0126-leet-code-ts-18"
+++

Welcome back to our LeetCode journey with TypeScript! In this blog post, we're
diving into a classic medium-difficulty problem: **Word Break**. This problem is
a fantastic way to explore dynamic programming (DP), a critical concept for
technical interviews, especially at top-tier companies like Google and Meta. If
you're preparing for a Silicon Valley job interview, understanding how to tackle
problems like this can set you apart. We'll break down the problem, explain it
in simple terms, and provide a complete TypeScript solution with detailed
commentary. Let's get started!

## Introduction to the Word Break Problem

The Word Break problem is defined as follows: Given a string `s` and a
dictionary of words `wordDict`, determine if `s` can be segmented into a
space-separated sequence of one or more dictionary words. In other words, can
you break the string into pieces where each piece is a valid word from the
dictionary?

For example:

- If `s = "leetcode"` and `wordDict = ["leet", "code"]`, the answer is `true`
  because "leetcode" can be split into "leet" and "code".
- If `s = "applepenapple"` and `wordDict = ["apple", "pen"]`, the answer is
  `true` because it can be split into "apple", "pen", and "apple".
- If `s = "catsandog"` and `wordDict = ["cats", "dog", "sand", "and", "cat"]`,
  the answer is `false` because there's no way to split it into valid words from
  the dictionary.

This problem tests your ability to use dynamic programming to solve a string
segmentation challenge. Dynamic programming is a technique where you break a
complex problem into smaller subproblems and store their results to avoid
redundant computations. It's a staple in coding interviews because it often
appears in problems involving optimization or decision-making, like this one.

## ELI5: Explaining Word Break Like You're Five

Imagine you have a long word, like "icecream", and a little book of smaller
words, like ["ice", "cream"]. Your job is to see if you can cut the long word
into pieces that are all in your book. So, can you cut "icecream" into "ice" and
"cream"? Yes, because both pieces are in the book!

Now, think of checking every possible way to cut the word. Start from the
beginning: Can I cut off the first letter? The first two letters? And so on. For
each cut, check if that piece is in your book. If it is, then check if the rest
of the word can also be cut into pieces from the book. To make this faster, you
remember (or write down) which parts you've already figured out, so you don't
have to check them again. This "remembering" trick is what we call dynamic
programming!

In the end, if you can cut the whole word into pieces that are all in the book,
you say "Yes, it works!" Otherwise, you say "No, it doesn't."

## Solving Word Break with Dynamic Programming in TypeScript

Let's solve this problem using dynamic programming. The idea is to create a
boolean array `dp` where `dp[i]` represents whether the substring `s[0...i-1]`
can be segmented into words from the dictionary. We'll build this array step by
step.

### Approach

1. Create a DP array of size `s.length + 1`, initialized to `false`. The extra
   slot is for the empty string, which is always valid (`dp[0] = true`).
2. For each position `i` in the string (from 1 to `s.length`), check every
   possible substring ending at `i` by looking at previous positions `j` (from 0
   to `i-1`).
3. If `dp[j]` is `true` (meaning `s[0...j-1]` is valid) and the substring
   `s[j...i-1]` is in the dictionary, then set `dp[i] = true`.
4. Finally, `dp[s.length]` tells us if the entire string can be segmented.

This approach ensures we check all possible ways to split the string while
reusing results from smaller subproblems, making it efficient with a time
complexity of $O(n^2)$, where $n$ is the length of the string.

### TypeScript Solution

Below is the complete TypeScript solution with detailed comments explaining each
step:

```typescript
function wordBreak(s: string, wordDict: string[]): boolean {
  // Convert wordDict to a Set for O(1) lookup time
  const wordSet = new Set<string>(wordDict);

  // Create a DP array where dp[i] means s[0...i-1] can be segmented
  const dp: boolean[] = new Array(s.length + 1).fill(false);

  // Empty string is always valid
  dp[0] = true;

  // Iterate through each position i in the string
  for (let i = 1; i <= s.length; i++) {
    // Check every possible substring ending at i by looking at previous positions j
    for (let j = 0; j < i; j++) {
      // If s[0...j-1] is valid and s[j...i-1] is in the dictionary
      if (dp[j] && wordSet.has(s.substring(j, i))) {
        dp[i] = true;
        break; // No need to check further once we find a valid split
      }
    }
  }

  // Return whether the entire string can be segmented
  return dp[s.length];
}

// Test cases to demonstrate the solution
function runTests(): void {
  console.log("Test 1:", wordBreak("leetcode", ["leet", "code"])); // Expected: true
  console.log("Test 2:", wordBreak("applepenapple", ["apple", "pen"])); // Expected: true
  console.log(
    "Test 3:",
    wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]),
  ); // Expected: false
}

// Run the tests
runTests();
```

### Explanation of the Code

- **Type Safety**: We're using TypeScript's type annotations (`string` and
  `string[]`) to ensure the input types are correct. The `Set<string>` ensures
  type-safe dictionary lookups.
- **DP Array**: The `dp` array is typed as `boolean[]` for clarity. It stores
  whether each prefix of the string is valid.
- **Efficiency**: Using a `Set` for `wordDict` makes lookups $O(1)$, which is
  crucial since we do many lookups. The overall time complexity is $O(n^2)$, and
  space complexity is $O(n)$.
- **Test Cases**: I've included test cases to verify the solution works as
  expected. You can run this code in a TypeScript environment (e.g., with
  `ts-node`) to see the output.

### Output of Test Cases

When you run the code, you'll get:

```
Test 1: true
Test 2: true
Test 3: false
```

These results match the expected behavior: "leetcode" and "applepenapple" can be
segmented, but "catsandog" cannot.

## Additional Exercise: Word Break with Word List Output

As a bonus, let's extend the problem to not just return whether the string can
be segmented, but also return one possible segmentation (i.e., the list of
words). This is a common follow-up question in interviews to test if you can
adapt your solution.

### Approach for Word List Output

We'll modify the DP solution to track the words used in the segmentation.
Instead of just storing a boolean in `dp`, we'll store the list of words (or an
empty array if invalid) that form the segmentation up to that point.

### TypeScript Solution for Word List Output

```typescript
function wordBreakWithList(s: string, wordDict: string[]): string[] | null {
  // Convert wordDict to a Set for O(1) lookup time
  const wordSet = new Set<string>(wordDict);

  // Create a DP array where dp[i] stores the list of words for s[0...i-1]
  // If no valid segmentation, store null
  const dp: (string[] | null)[] = new Array(s.length + 1).fill(null);

  // Empty string has an empty list of words
  dp[0] = [];

  // Iterate through each position i in the string
  for (let i = 1; i <= s.length; i++) {
    // Check every possible substring ending at i by looking at previous positions j
    for (let j = 0; j < i; j++) {
      if (dp[j] !== null) {
        const word = s.substring(j, i);
        if (wordSet.has(word)) {
          // If this is the first valid segmentation found, or we want to store it
          dp[i] = [...dp[j], word];
          break; // Take the first valid segmentation for simplicity
        }
      }
    }
  }

  // Return the list of words if the entire string can be segmented, else null
  return dp[s.length];
}

// Test cases to demonstrate the solution with word list output
function runTestsWithList(): void {
  console.log("Test 1:", wordBreakWithList("leetcode", ["leet", "code"])); // Expected: ["leet", "code"]
  console.log("Test 2:", wordBreakWithList("applepenapple", ["apple", "pen"])); // Expected: ["apple", "pen", "apple"]
  console.log(
    "Test 3:",
    wordBreakWithList("catsandog", ["cats", "dog", "sand", "and", "cat"]),
  ); // Expected: null
}

// Run the tests
runTestsWithList();
```

### Explanation of the Extended Solution

- **DP Array**: Now `dp` is typed as `(string[] | null)[]`, storing either a
  list of words or `null` if no segmentation is possible.
- **Tracking Words**: When a valid split is found, we copy the word list from
  `dp[j]` and append the current word `s[j...i-1]`.
- **Output**: The final result at `dp[s.length]` is either the list of words or
  `null` if no segmentation exists.

### Output of Test Cases for Word List

When you run the code, you'll get:

```
Test 1: ["leet", "code"]
Test 2: ["apple", "pen", "apple"]
Test 3: null
```

This matches the expected behavior, showing one possible segmentation when it
exists and `null` when it doesn't.

## Conclusion

In this blog post, we've tackled the **Word Break** problem, a medium-difficulty
LeetCode challenge that’s a favorite in Silicon Valley interviews at companies
like Google and Meta. We introduced the problem and explained it in simple terms
using an ELI5 analogy of cutting words into pieces from a book. We then provided
two complete TypeScript solutions: one to check if segmentation is possible
using dynamic programming, and a follow-up to output the actual list of words in
a segmentation. Key takeaways include:

- Dynamic programming is a powerful technique to solve string segmentation by
  breaking it into subproblems and storing results in a `dp` array.
- TypeScript’s type safety helps ensure our code is robust, with clear typing
  for inputs, outputs, and data structures like `Set<string>`.
- The time complexity of the solution is $O(n^2)$, making it efficient for
  typical interview constraints.
- Extending the problem to output the word list shows how to adapt DP solutions
  for follow-up questions, a common interview scenario.

By mastering problems like Word Break, you're building a strong foundation in
dynamic programming, a critical skill for coding interviews. In the next post,
we'll explore another exciting LeetCode problem to continue leveling up your
skills. Stay tuned, and happy coding!
