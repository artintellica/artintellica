+++
title = "LeetCode in TypeScript, Part 1.7: Merge Intervals"
author = "Artintellica"
date = "2025-06-24"
+++

Welcome back to our LeetCode series in TypeScript! Today, we're tackling a
medium-difficulty problem called **Merge Intervals**. This problem is a favorite
in technical interviews, especially at companies like Amazon and Microsoft,
because it tests your ability to handle real-world data problems involving
ranges or schedules. Whether you're optimizing meeting times or managing
resource allocations, merging overlapping intervals is a practical skill. In
this post, we'll break down the problem, explain it in simple terms, provide a
TypeScript solution, and analyze the key concepts.

## Introduction to Merge Intervals

The Merge Intervals problem is defined as follows: Given an array of intervals
where each interval is represented as `[start, end]`, merge all overlapping
intervals and return an array of non-overlapping intervals that cover all the
intervals in the input. For example, if the input is
`[[1,3], [2,6], [8,10], [15,18]]`, the output should be
`[[1,6], [8,10], [15,18]]` because `[1,3]` and `[2,6]` overlap and can be merged
into `[1,6]`.

This problem is relevant in interviews because it combines multiple fundamental
concepts: sorting (to arrange intervals for easier comparison), array
manipulation, and logical reasoning about overlaps. It also mimics real-world
scenarios like scheduling or resource management, which are common in software
engineering tasks. Solving this problem efficiently demonstrates your ability to
think algorithmically and handle edge cases.

## ELI5: Merge Intervals Explained Like You're Five

Imagine you have a bunch of colored strips of paper on a table, and each strip
represents a time period, like from 1 o'clock to 3 o'clock. Some strips overlap
because they cover the same time, like one from 2 o'clock to 6 o'clock. Your job
is to combine the overlapping strips into one big strip so that you have fewer
strips, but they still cover all the same times. So, if two strips overlap, you
glue them together to make one strip that starts at the earliest time and ends
at the latest time of the two.

To do this, first, you line up all the strips from the earliest start time to
the latest. Then, you go through them one by one. If the next strip starts
before the current one ends, they overlap, so you stretch the current strip to
cover the later end time. If they don't overlap, you keep the current strip as
it is and move to the next one as a new strip. At the end, you have a neat set
of strips with no overlaps!

## Key Concepts and Approach

Before diving into the code, let's outline the key concepts and the approach
we'll use to solve this problem:

- **Sorting**: We sort the intervals by their start times. This ensures we
  process intervals in order and can easily check for overlaps.
- **Overlap Check**: Two intervals overlap if the start time of the second
  interval is less than or equal to the end time of the first interval.
- **Merging**: If there's an overlap, update the end time of the current
  interval to be the maximum of the current end and the next interval's end. If
  there's no overlap, add the current interval to the result and move to the
  next one.

The time complexity of this solution will be $O(n \log n)$ due to the sorting
step, where $n$ is the number of intervals. The space complexity is $O(1)$ or
$O(n)$ depending on whether the sorting is done in-place or requires additional
space, excluding the space needed for the output.

## TypeScript Solution for Merge Intervals

Let's implement the solution in TypeScript. We'll define the input as an array
of number arrays (each inner array representing an interval `[start, end]`), and
we'll return a merged array of intervals. Here's the complete solution with
detailed comments:

```typescript
function merge(intervals: number[][]): number[][] {
  // If the input is empty, return an empty array
  if (intervals.length === 0) {
    return [];
  }

  // Sort intervals based on start time
  intervals.sort((a, b) => a[0] - b[0]);

  // Initialize the result array with the first interval
  const merged: number[][] = [intervals[0]];

  // Iterate through the sorted intervals starting from the second one
  for (let i = 1; i < intervals.length; i++) {
    const currentInterval = intervals[i];
    const lastMerged = merged[merged.length - 1];

    // Check if the current interval overlaps with the last merged interval
    if (currentInterval[0] <= lastMerged[1]) {
      // If overlap, update the end time of the last merged interval
      lastMerged[1] = Math.max(lastMerged[1], currentInterval[1]);
    } else {
      // If no overlap, add the current interval to the result
      merged.push(currentInterval);
    }
  }

  return merged;
}

// Test cases
console.log(
  merge([
    [1, 3],
    [2, 6],
    [8, 10],
    [15, 18],
  ]),
); // Output: [[1,6],[8,10],[15,18]]
console.log(
  merge([
    [1, 4],
    [4, 5],
  ]),
); // Output: [[1,5]]
console.log(
  merge([
    [1, 4],
    [0, 4],
  ]),
); // Output: [[0,4]]
console.log(
  merge([
    [1, 4],
    [2, 3],
  ]),
); // Output: [[1,4]]
```

### Explanation of the Code

1. **Input Check**: If the input array is empty, return an empty array to handle
   edge cases.
2. **Sorting**: We sort the intervals by start time using the `sort` method.
   This ensures we process intervals in ascending order of start times.
3. **Initialization**: Start with the first interval in the `merged` array as
   the baseline.
4. **Iteration and Merging**:
   - For each subsequent interval, compare its start time with the end time of
     the last interval in `merged`.
   - If there's an overlap (`currentInterval[0] <= lastMerged[1]`), update the
     end time of `lastMerged` to the maximum of the two end times.
   - If there's no overlap, add the current interval to `merged`.
5. **Return**: The `merged` array contains the final non-overlapping intervals.

### Time and Space Complexity

- **Time Complexity**: $O(n \log n)$ due to the sorting step. The iteration
  after sorting is $O(n)$.
- **Space Complexity**: $O(1)$ for the algorithm itself (excluding the output
  array), as we modify the input or use minimal extra space. However, depending
  on the JavaScript engine, sorting might use $O(n)$ space.

## Additional Exercise: Merge Intervals with Edge Cases

To solidify our understanding, let's consider a variation or additional test
cases that might come up in an interview. Suppose we need to handle unsorted
intervals with more complex overlaps or edge cases like fully contained
intervals. The solution above already handles these, but let's explicitly test
more scenarios.

```typescript
function testMergeIntervals(): void {
  const testCases = [
    {
      input: [
        [1, 3],
        [2, 6],
        [8, 10],
        [15, 18],
      ],
      expected: [
        [1, 6],
        [8, 10],
        [15, 18],
      ],
    },
    {
      input: [
        [1, 4],
        [4, 5],
      ],
      expected: [[1, 5]],
    },
    {
      input: [
        [1, 4],
        [0, 4],
      ],
      expected: [[0, 4]],
    },
    {
      input: [
        [1, 4],
        [2, 3],
      ],
      expected: [[1, 4]],
    },
    {
      input: [
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [1, 10],
      ],
      expected: [[1, 10]],
    },
    { input: [], expected: [] },
  ];

  testCases.forEach((test, index) => {
    const result = merge(test.input);
    console.log(`Test ${index + 1}: Input: ${JSON.stringify(test.input)}`);
    console.log(`Expected: ${JSON.stringify(test.expected)}`);
    console.log(`Got: ${JSON.stringify(result)}`);
    console.log(
      `Pass: ${JSON.stringify(result) === JSON.stringify(test.expected)}\n`,
    );
  });
}

// Run the tests
testMergeIntervals();
```

### Explanation of Additional Test Cases

- **Fully Overlapping Intervals**: `[[1,10]]` with other intervals inside like
  `[[2,3],[4,5]]` tests if all smaller intervals are merged into one.
- **Empty Input**: Ensures the function handles empty arrays gracefully.
- **Unsorted Input**: `[[1,4],[0,4]]` confirms sorting works correctly to merge
  intervals regardless of input order.

The solution provided earlier handles all these cases correctly because of the
sorting step and the overlap check logic.

## Conclusion

In this blog post, we've explored the Merge Intervals problem, a
medium-difficulty LeetCode challenge that's highly relevant for technical
interviews at companies like Amazon and Microsoft. We started with a basic
introduction to the problem, which involves combining overlapping time ranges
into a simplified set of non-overlapping intervals. Using an ELI5 analogy, we
likened it to gluing overlapping paper strips together. We then implemented a
solution in TypeScript, leveraging sorting and iterative merging with a time
complexity of $O(n \log n)$. We also tested the solution with various edge cases
to ensure robustness.

Key takeaways include:

- Sorting intervals by start time is crucial for efficiently detecting overlaps.
- Merging logic relies on comparing the start of the current interval with the
  end of the last merged interval.
- TypeScript's type system, while not heavily used here, ensures type safety for
  input and output arrays.

This problem tests fundamental skills in array manipulation, sorting, and
logical reasoning—skills that are essential for coding interviews. In the next
post, we'll tackle another exciting LeetCode challenge, building on these
concepts. Keep practicing, and happy coding!
