+++
title = "LeetCode in TypeScript, Part 1.6: Longest Substring Without Repeating Characters"
author = "Artintellica"
date = "2025-06-23"
+++

Welcome back to our series on solving LeetCode problems with TypeScript, tailored for Silicon Valley job interviews! In this post, we're tackling a medium-difficulty problem that frequently appears in technical interviews at top tech companies like Meta and Google: **Longest Substring Without Repeating Characters**. This problem is a fantastic way to explore the sliding window technique and the use of hash maps, both of which are essential tools in an interview candidate's toolkit. Whether you're preparing for a coding interview or just sharpening your algorithmic skills, this problem offers valuable insights into string manipulation and optimization. Let’s dive in!

## Problem Introduction

The problem statement is straightforward: given a string, find the length of the longest substring without repeating characters. For example, in the string `"abcabcbb"`, the longest substring without repeating characters is `"abc"`, which has a length of 3. In `"bbbbb"`, the answer is `"b"` with a length of 1, and in `"pwwkew"`, the answer is `"wke"` with a length of 3.

This problem tests your ability to handle strings efficiently and introduces the **sliding window** technique, a common approach for problems involving arrays or strings where you need to find a subarray or substring with specific properties. It also requires the use of a hash map (or set) to track characters, making it a great exercise in balancing time and space complexity.

**Why It's Relevant**: This problem is a favorite in interviews because it combines multiple concepts—string traversal, window management, and data structure usage—into a single, moderately challenging question. Solving it efficiently demonstrates a solid grasp of algorithmic thinking, which is exactly what companies like Meta and Google look for.

**Key Concepts**: Strings, Sliding Window, Hash Maps.

## ELI5: Explain Like I'm 5

Imagine you’re walking along a path, and you want to collect a bunch of different toys in a row without picking up the same toy twice. You have a little bag (that’s like our "window") to hold the toys you’ve picked up. You start walking and pick up toys one by one. If you see a toy you already have in your bag, you have to stop and figure out how many different toys you collected before that repeat. Then, you drop some toys from the start of your collection to make room for the new one and keep going.

In this problem, the "path" is the string, the "toys" are the characters, and the "bag" is our sliding window. We’re trying to find the longest stretch of the path where all the toys (characters) are different. We use a little notebook (a hash map) to remember which toys we’ve already picked up and where we saw them, so we know when to drop some toys and start a new collection.

The goal is to find the longest group of different toys you can collect in one go without any repeats. That number—the size of the biggest group—is our answer!

## Approach to Solving the Problem

To solve this, we’ll use the sliding window technique. A sliding window is like a frame that moves over the string, expanding or shrinking based on certain conditions. Here, our window represents a substring without repeating characters. We’ll maintain two pointers: a `left` pointer for the start of the window and a `right` pointer for the end. As we move the `right` pointer to include more characters, we check if the new character is already in our window. If it is, we move the `left` pointer to just after the last occurrence of that character, effectively shrinking the window to avoid duplicates.

To track characters and their positions efficiently, we’ll use a hash map (in TypeScript, a `Map` or an object) to store the last index where we saw each character. This helps us quickly decide where to move the `left` pointer when we encounter a repeat. We’ll also keep track of the maximum length of the window seen so far.

### Time and Space Complexity
- **Time Complexity**: $O(n)$, where $n$ is the length of the string. Each character is added and removed from the window at most once.
- **Space Complexity**: $O(min(m, n))$, where $m$ is the size of the character set (e.g., 26 for lowercase letters), and $n$ is the string length. This is the space used by the hash map to store characters.

## TypeScript Solution

Let’s implement this solution in TypeScript. I’ll include detailed comments to explain each step, and we’ll use TypeScript’s type annotations to make the code clear and safe.

```typescript
function lengthOfLongestSubstring(s: string): number {
    // Initialize a map to store the last index of each character
    const charIndexMap: Map<string, number> = new Map();
    
    // Initialize variables for the left pointer, max length, and loop through string with right pointer
    let left: number = 0;
    let maxLength: number = 0;
    
    // Iterate over the string using the right pointer
    for (let right = 0; right < s.length; right++) {
        const currentChar: string = s[right];
        
        // Check if the current character is already in the window
        if (charIndexMap.has(currentChar)) {
            // Move the left pointer to the position just after the last occurrence of currentChar
            // Use Math.max to ensure we don't move left pointer backward
            left = Math.max(left, charIndexMap.get(currentChar)! + 1);
        }
        
        // Update the last seen index of the current character
        charIndexMap.set(currentChar, right);
        
        // Update the maximum length if the current window is larger
        maxLength = Math.max(maxLength, right - left + 1);
    }
    
    return maxLength;
}
```

### How It Works
- We use a `Map` to store characters and their last seen indices.
- The `left` pointer marks the start of our current window, and `right` marks the end as we iterate through the string.
- If we encounter a character already in our map, we update `left` to just after its last occurrence (ensuring we don’t move backward with `Math.max`).
- After each iteration, we update the `maxLength` by calculating the size of the current window (`right - left + 1`).
- Finally, we return the largest window size found.

### Example Walkthrough
Let’s test it with the string `"abcabcbb"`:
- At `right=0`, char=`a`, map={a:0}, window=`a`, maxLength=1
- At `right=1`, char=`b`, map={a:0,b:1}, window=`ab`, maxLength=2
- At `right=2`, char=`c`, map={a:0,b:1,c:2}, window=`abc`, maxLength=3
- At `right=3`, char=`a`, already in map at index 0, so `left=1`, map={a:3,b:1,c:2}, window=`bca`, maxLength=3
- Continue this process until the end, and the final `maxLength` is 3.

## Testing the Solution
Let’s write a small test suite in TypeScript to verify our solution works for various cases.

```typescript
function runTests(): void {
    const testCases: { input: string; expected: number }[] = [
        { input: "abcabcbb", expected: 3 }, // "abc"
        { input: "bbbbb", expected: 1 },    // "b"
        { input: "pwwkew", expected: 3 },  // "wke"
        { input: "", expected: 0 },        // empty string
        { input: "dvdf", expected: 3 },    // "vdf"
    ];

    testCases.forEach((test, index) => {
        const result: number = lengthOfLongestSubstring(test.input);
        console.log(
            `Test ${index + 1}: Input="${test.input}", Expected=${test.expected}, Got=${result}, ${
                result === test.expected ? "PASS" : "FAIL"
            }`
        );
    });
}

// Run the tests
runTests();
```

### Output
Running the above test suite should produce something like:
```
Test 1: Input="abcabcbb", Expected=3, Got=3, PASS
Test 2: Input="bbbbb", Expected=1, Got=1, PASS
Test 3: Input="pwwkew", Expected=3, Got=3, PASS
Test 4: Input="", Expected=0, Got=0, PASS
Test 5: Input="dvdf", Expected=3, Got=3, PASS
```

These test cases cover common scenarios, including strings with repeats, all identical characters, empty strings, and tricky cases like `"dvdf"` where the repeat isn’t at the start of the current window.

## Conclusion

In this blog post, we tackled the **Longest Substring Without Repeating Characters** problem, a medium-difficulty LeetCode challenge that’s a staple in Silicon Valley interviews at companies like Meta and Google. We explored the sliding window technique, using two pointers to maintain a dynamic window of unique characters, and leveraged a hash map (`Map` in TypeScript) to track character positions efficiently. Our TypeScript solution demonstrated type safety and clarity, with a time complexity of $O(n)$ and space complexity of $O(min(m, n))$.

Key takeaways:
- The sliding window technique is powerful for string and array problems where you need to find a substructure with specific constraints.
- Hash maps are invaluable for quick lookups, helping us avoid nested loops and achieve linear time complexity.
- Writing test cases ensures our solution handles edge cases like empty strings or repeating patterns.

By mastering problems like this, you’re building a strong foundation for technical interviews, especially for roles that test algorithmic problem-solving. In the next post, we’ll dive into another exciting problem to further hone our skills. Stay tuned, and keep coding!
