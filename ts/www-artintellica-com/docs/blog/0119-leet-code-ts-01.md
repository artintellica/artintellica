+++
title = "LeetCode in TypeScript, Part 1: Solving Two Sum (Easy)"
author = "Artintellica"
date = "2025-06-23"
+++

Welcome to the first post in our series on tackling LeetCode problems with TypeScript! If you're preparing for technical interviews in Silicon Valley, you're likely aware that mastering data structures and algorithms is crucial. LeetCode offers a fantastic platform to hone these skills, and in this series, we'll solve problems ranging from elementary to advanced, all while leveraging TypeScript's powerful type system to write clean, maintainable code.

In this post, we're starting with one of the most iconic and frequently asked interview questions: **Two Sum**. This problem is often used as a warm-up question at companies like Amazon and Google, testing your ability to manipulate arrays and optimize solutions using hash maps. Let's dive in and explore how to solve it step by step with TypeScript.

## Problem Introduction: Two Sum

The Two Sum problem is straightforward but packs a punch in terms of teaching core concepts. Here's the problem statement:

- **Problem**: Given an array of integers `nums` and an integer `target`, return the indices of the two numbers such that they add up to `target`. You may assume that each input would have exactly one valid solution, and you may not use the same element twice.
- **Example**:
  - Input: `nums = [2, 7, 11, 15]`, `target = 9`
  - Output: `[0, 1]` (because `nums[0] + nums[1] = 2 + 7 = 9`)
- **Why It's Relevant**: This problem tests your understanding of array traversal and optimization techniques. It's a common first question in interviews to gauge your problem-solving approach and familiarity with basic data structures like hash maps.

The key concepts we'll cover include arrays, hash maps, and time complexity analysis, aiming for an efficient solution with $O(n)$ time complexity.

## ELI5: Understanding Two Sum Like You're Five

Imagine you have a box of toy blocks, each with a number written on it. Your mom gives you a special number, say 9, and asks you to find two blocks that add up to that number. You also need to tell her where those blocks are in the box (their positions).

- **First Way (Slow)**: You pick one block, then check every other block to see if they add up to 9. If not, you pick another block and check again. This takes a long time if you have lots of blocks!
- **Better Way (Fast)**: You have a magic notebook. As you look at each block, you write down what number you need to find to make 9 (like if you pick a block with 2, you write "need 7"). Then, every time you pick a new block, you check your notebook to see if you've seen the number you need before. If you have, bingo! You know where both blocks are.

The fast way is like using a hash map—a special tool to remember numbers and their positions quickly. This makes finding the two blocks much faster, even if you have a huge box of blocks!

## Solving Two Sum in TypeScript

Let's implement the Two Sum solution in TypeScript. We'll use a hash map approach to achieve $O(n)$ time complexity, where $n$ is the length of the input array. TypeScript's type system will help us define the input and output types clearly, making our code more robust.

### Step-by-Step Approach
1. Create a hash map to store numbers we've seen and their indices.
2. Iterate through the array. For each number, calculate the complement (target minus the current number).
3. Check if the complement exists in the hash map. If it does, we've found our pair of indices.
4. If not, add the current number and its index to the hash map.
5. Return the indices of the two numbers that sum to the target.

### Full Solution Code

Here's the complete TypeScript solution with detailed comments:

```typescript
function twoSum(nums: number[], target: number): number[] {
    // Create a hash map to store number-to-index mappings
    const numMap: Map<number, number> = new Map();
    
    // Iterate through the array
    for (let i = 0; i < nums.length; i++) {
        const currentNum = nums[i];
        // Calculate the complement needed to reach the target
        const complement = target - currentNum;
        
        // Check if complement exists in the map
        if (numMap.has(complement)) {
            // Return the indices of the two numbers
            return [numMap.get(complement)!, i];
        }
        
        // If not found, add the current number and its index to the map
        numMap.set(currentNum, i);
    }
    
    // No solution found (though problem guarantees one exists)
    return [];
}

// Test the function
const nums = [2, 7, 11, 15];
const target = 9;
console.log(twoSum(nums, target)); // Output: [0, 1]
```

### Explanation of the Code
- **Type Safety**: We explicitly type the input `nums` as `number[]` and the output as `number[]` (an array of two indices). This ensures TypeScript catches any type mismatches during development.
- **Hash Map**: We use `Map<number, number>` to store numbers as keys and their indices as values. `Map` is a built-in TypeScript/JavaScript data structure that's perfect for quick lookups.
- **Time Complexity**: The solution runs in $O(n)$ time because we only traverse the array once, and hash map operations (get/set) are $O(1)$ on average.
- **Space Complexity**: We use $O(n)$ extra space to store the hash map.

### Why This Matters in Interviews
In an interview setting, starting with a brute-force approach (nested loops, $O(n^2)$ time) is fine, but you should quickly pivot to the hash map solution to demonstrate optimization skills. Be prepared to explain edge cases, like:
- What if the array has negative numbers? (Our solution handles them fine.)
- What if there are duplicate numbers? (Our solution works as long as there's exactly one valid pair.)

## Additional Exercise: Two Sum with Multiple Solutions

Sometimes, interviewers might tweak the problem to ask for all pairs of indices that sum to the target (not just one pair). Let's solve this variation in TypeScript as an additional exercise to deepen our understanding.

### Problem Variation
- Given an array of integers `nums` and an integer `target`, return all pairs of indices where the numbers add up to `target`. Numbers can be used only once per pair, but the array might have duplicates.

### Full Solution Code for Variation

```typescript
function twoSumAllPairs(nums: number[], target: number): number[][] {
    // Create a hash map to store number-to-indices mappings (handling duplicates)
    const numMap: Map<number, number[]> = new Map();
    const result: number[][] = [];
    
    // Populate the map with all indices for each number
    for (let i = 0; i < nums.length; i++) {
        const currentNum = nums[i];
        if (!numMap.has(currentNum)) {
            numMap.set(currentNum, []);
        }
        numMap.get(currentNum)!.push(i);
    }
    
    // Iterate through the array to find pairs
    for (let i = 0; i < nums.length; i++) {
        const currentNum = nums[i];
        const complement = target - currentNum;
        
        // Check if complement exists
        if (numMap.has(complement)) {
            const complementIndices = numMap.get(complement)!;
            // Add pairs, ensuring we don't reuse the same index
            for (const j of complementIndices) {
                if (j > i) { // Only consider indices after i to avoid duplicates
                    result.push([i, j]);
                }
            }
        }
    }
    
    return result;
}

// Test the function
const numsVariation = [1, 5, 5, 1, 3];
const targetVariation = 6;
console.log(twoSumAllPairs(numsVariation, targetVariation)); // Output: [[0, 1], [0, 2], [3, 1], [3, 2]]
```

### Explanation of the Variation
- **Hash Map for Duplicates**: We store arrays of indices for each number to handle duplicates (e.g., multiple 5s in the array).
- **Avoiding Reuse**: We ensure `j > i` to avoid using the same index twice in a pair and to prevent duplicate pairs in the result.
- **Time Complexity**: This runs in $O(n^2)$ in the worst case due to potentially many pairs, but the hash map still helps with lookups.

This variation shows how to adapt the original solution to a slightly different requirement, a common scenario in interviews where follow-up questions test your flexibility.

## Conclusion

In this blog post, we've tackled the classic LeetCode problem **Two Sum**, a staple in Silicon Valley technical interviews at companies like Amazon and Google. We started with an introduction to the problem, broke it down with an ELI5 analogy of finding toy blocks, and implemented a solution in TypeScript using a hash map for $O(n)$ time complexity. We also explored a variation to find all possible pairs, showcasing how to handle follow-up questions.

Key takeaways:
- Use hash maps (`Map` in TypeScript) to optimize array problems from $O(n^2)$ to $O(n)$ time.
- TypeScript's type system helps define clear input/output types, making code safer and more readable.
- Be prepared for variations in interviews—think about duplicates, multiple solutions, and edge cases.

In the next post, we'll build on these concepts with another easy problem, likely focusing on strings or linked lists. Stay tuned, and happy coding! If you have questions or want to see another variation, drop a comment below.
