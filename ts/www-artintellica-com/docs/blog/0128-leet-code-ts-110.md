+++
title = "LeetCode in TypeScript, Part 1.10: Finding the Median of Two Sorted Arrays"
author = "Artintellica"
date = "2025-06-24"
code = "https://github.com/artintellica/artintellica/tree/main/ts/blog/0128-leet-code-ts-110"
+++

Welcome back to our LeetCode journey in TypeScript! In this post, we're diving
into one of the more challenging problems often encountered in technical
interviews at top-tier Silicon Valley companies like Google: **Median of Two
Sorted Arrays**. This problem is classified as "Hard" on LeetCode, and it’s a
fantastic test of your ability to handle binary search, arrays, and edge cases
under pressure. Whether you're preparing for a final-round interview or just
looking to sharpen your algorithmic skills, this problem will push your limits
and teach you valuable techniques.

We'll break down the problem, explain it in simple terms, provide a detailed
TypeScript solution, and analyze the approach step by step. Let's get started!

## Problem Introduction: Median of Two Sorted Arrays

The problem is straightforward to state but tricky to solve efficiently. You're
given two sorted arrays of integers, `nums1` and `nums2`, of lengths `m` and `n`
respectively. Your task is to find the median of the combined sorted array
formed by merging these two arrays. The median is the middle value in a sorted
list of numbers. If the total number of elements is odd, the median is the
middle element. If it's even, the median is the average of the two middle
elements.

For example:

- If `nums1 = [1, 3]` and `nums2 = [2]`, the combined sorted array is
  `[1, 2, 3]`, and the median is `2`.
- If `nums1 = [1, 2]` and `nums2 = [3, 4]`, the combined sorted array is
  `[1, 2, 3, 4]`, and the median is `(2 + 3) / 2 = 2.5`.

The challenge lies in solving this efficiently without merging the arrays
explicitly, as that would take $O(m + n)$ time and space. Instead, we aim for a
solution with $O(\log(\min(m, n)))$ time complexity using binary search.

**Why It's Relevant**: This problem is a favorite in final interview rounds
because it tests deep understanding of binary search, partitioning arrays, and
handling edge cases. It’s not just about coding; it’s about thinking creatively
under constraints.

## ELI5: Understanding the Median of Two Sorted Arrays

Imagine you have two sorted lists of numbers, like two stacks of cards already
in order. One stack has numbers like 1, 3, and the other has 2, 4, 6. You want
to find the "middle" number if you combined both stacks into one big sorted
stack. But here’s the catch: you don’t want to actually combine them because
that takes too long. Instead, you want to figure out the middle by just looking
at parts of each stack.

Think of it like cutting each stack into two parts. You guess where to cut the
first stack so that the left part has some numbers, and the right part has the
rest. Then, you cut the second stack so that the left parts of both stacks
together have about half the total cards. If the numbers at the edges of your
cuts make sense (the biggest number on the left is smaller than the smallest
number on the right for both cuts), you’ve found the right spot! The median is
then based on those edge numbers.

This "cutting" idea is what binary search helps us do. We keep guessing where to
cut until we find the perfect spot, and we do it fast by halving our search
space each time.

## Approach to Solving the Problem

To solve this problem efficiently, we use a binary search approach to partition
the two arrays into two halves such that the left halves of both arrays contain
elements less than or equal to the right halves. Here's the high-level idea:

1. **Partitioning**: We try to partition `nums1` and `nums2` into left and right
   parts. For `nums1`, pick a partition point `x`, so the left part is
   `nums1[0..x-1]` and the right part is `nums1[x..m-1]`. Similarly, for
   `nums2`, pick a partition point `y`, so the left part is `nums2[0..y-1]` and
   the right part is `nums2[y..n-1]`.
2. **Balance**: Ensure the number of elements in the combined left parts equals
   (or is one less than) the number in the combined right parts. This means
   `x + y` should be roughly half of `m + n`.
3. **Validity Check**: Check if the partition is valid by ensuring the largest
   element in the left parts is less than or equal to the smallest element in
   the right parts. Mathematically, `nums1[x-1] <= nums2[y]` and
   `nums2[y-1] <= nums1[x]`.
4. **Binary Search**: Use binary search on the smaller array to find the correct
   partition point `x`, then compute `y` as `((m + n + 1) / 2) - x`. Adjust the
   search range based on whether the partition is valid.
5. **Median Calculation**: Once the correct partition is found, compute the
   median based on whether the total length is odd or even.

The time complexity is $O(\log(\min(m, n)))$ because we perform binary search on
the smaller array. Space complexity is $O(1)$ as we use only a few variables.

## TypeScript Solution: Median of Two Sorted Arrays

Below is the complete TypeScript solution to the problem. I've added detailed
comments to explain each step. We'll assume the input arrays are `number[]`
types for clarity and type safety.

```typescript
function findMedianSortedArrays(nums1: number[], nums2: number[]): number {
  // Ensure nums1 is the shorter array for efficiency in binary search
  if (nums1.length > nums2.length) {
    [nums1, nums2] = [nums2, nums1];
  }

  const m: number = nums1.length;
  const n: number = nums2.length;
  let left: number = 0;
  let right: number = m;

  while (left <= right) {
    // Partition points for nums1 and nums2
    const x: number = Math.floor((left + right) / 2);
    const y: number = Math.floor((m + n + 1) / 2) - x;

    // Get left and right values for nums1
    const left1: number = x === 0 ? Number.NEGATIVE_INFINITY : nums1[x - 1];
    const right1: number = x === m ? Number.POSITIVE_INFINITY : nums1[x];

    // Get left and right values for nums2
    const left2: number = y === 0 ? Number.NEGATIVE_INFINITY : nums2[y - 1];
    const right2: number = y === n ? Number.POSITIVE_INFINITY : nums2[y];

    // Check if partition is valid
    if (left1 <= right2 && left2 <= right1) {
      // If total length is odd, median is the max of left parts
      if ((m + n) % 2 === 1) {
        return Math.max(left1, left2);
      } else {
        // If total length is even, median is average of max(left) and min(right)
        return (Math.max(left1, left2) + Math.min(right1, right2)) / 2;
      }
    } else if (left1 > right2) {
      // If left1 is too big, move partition left in nums1
      right = x - 1;
    } else {
      // If left2 is too big, move partition right in nums1
      left = x + 1;
    }
  }

  throw new Error("Input arrays are not sorted");
}
```

## Explanation of the Code

- **Swapping Arrays**: We ensure `nums1` is the shorter array by swapping if
  necessary. This optimizes the binary search by minimizing iterations.
- **Binary Search Setup**: We search for the partition point `x` in `nums1`
  between `0` and `m`. For each `x`, we compute `y` to balance the left and
  right halves.
- **Edge Cases with Infinity**: If a partition point is at the start or end of
  an array, we use `Number.NEGATIVE_INFINITY` or `Number.POSITIVE_INFINITY` to
  handle comparisons safely.
- **Partition Validation**: We check if `left1 <= right2` and `left2 <= right1`.
  If true, we’ve found the correct partition and compute the median.
- **Median Calculation**: For odd total length, the median is the maximum of the
  left parts. For even length, it’s the average of the maximum of left parts and
  minimum of right parts.
- **Adjustment**: If the partition isn’t valid, we adjust the binary search
  range based on which side needs correction.

## Testing the Solution

Let’s test the solution with a few examples to verify it works correctly.

**Example 1**:

```typescript
const nums1: number[] = [1, 3];
const nums2: number[] = [2];
console.log(findMedianSortedArrays(nums1, nums2)); // Output: 2
```

- Combined sorted array: `[1, 2, 3]`
- Median: `2` (middle element since total length is 3, odd)

**Example 2**:

```typescript
const nums1: number[] = [1, 2];
const nums2: number[] = [3, 4];
console.log(findMedianSortedArrays(nums1, nums2)); // Output: 2.5
```

- Combined sorted array: `[1, 2, 3, 4]`
- Median: `(2 + 3) / 2 = 2.5` (average of two middle elements since total length
  is 4, even)

**Example 3**:

```typescript
const nums1: number[] = [0, 0];
const nums2: number[] = [0, 0];
console.log(findMedianSortedArrays(nums1, nums2)); // Output: 0
```

- Combined sorted array: `[0, 0, 0, 0]`
- Median: `(0 + 0) / 2 = 0`

## Key Takeaways and Interview Tips

When discussing this problem in an interview:

- Explain the binary search approach clearly, focusing on why merging arrays
  ($O(m + n)$) is inefficient compared to partitioning ($O(\log(\min(m, n)))$).
- Draw diagrams to show how partitions work, especially for edge cases like
  empty arrays or arrays of different sizes.
- Mention how you handle edge cases with infinities to avoid index out-of-bounds
  errors.
- Be prepared to discuss why you chose the shorter array for binary search (to
  minimize iterations).

## Conclusion

In this blog post, we tackled the challenging LeetCode problem "Median of Two
Sorted Arrays" using TypeScript. We explored a binary search-based solution that
achieves $O(\log(\min(m, n)))$ time complexity by partitioning the arrays
instead of merging them. Through an ELI5 explanation, we likened the problem to
cutting stacks of cards to find the middle without combining them. The
TypeScript code provided a robust, type-safe implementation with detailed
comments and test cases to solidify understanding.

Key points to remember:

- Binary search on the smaller array optimizes the solution.
- Proper partitioning ensures left elements are less than or equal to right
  elements across both arrays.
- Edge cases are handled using infinities to avoid errors.
- This problem tests advanced algorithmic thinking, making it a staple in
  final-round interviews at companies like Google.

Stay tuned for more LeetCode challenges in TypeScript as we continue to build
skills for Silicon Valley interviews. Happy coding!
