+++
title = "LeetCode in TypeScript, Part 1.4: Merging Two Sorted Lists"
author = "Artintellica"
date = "2025-06-23"
+++

Welcome back to our series on solving LeetCode problems with TypeScript,
tailored for cracking those Silicon Valley job interviews! Today, we're diving
into an "Easy" yet fundamental problem: **Merge Two Sorted Lists**. This problem
is a classic when it comes to understanding linked lists, a data structure that
often pops up in technical interviews at companies like Amazon. If you're
preparing for a coding interview, mastering linked list operations like merging
is essential, as they test your ability to manipulate pointers and maintain
order in data.

In this post, we'll break down the problem, explain it in simple terms, provide
a TypeScript solution with detailed comments, analyze the time and space
complexity, and wrap up with key takeaways. Let's get started!

## Problem Introduction: Merge Two Sorted Lists

The problem statement is straightforward: You are given the heads of two sorted
linked lists, `list1` and `list2`. Your task is to merge these two lists into a
single sorted list. The merged list should maintain the ascending order of the
original lists. The input lists are singly linked lists, meaning each node
points to the next node, and the last node points to `null`.

For example:

- Input: `list1 = [1,2,4]`, `list2 = [1,3,4]`
- Output: `[1,1,2,3,4,4]`

This problem is relevant because it tests your understanding of linked list
traversal and pointer manipulation—skills that are often assessed in interviews.
It also introduces the concept of merging, which can be extended to more complex
problems like merging k-sorted lists or sorting algorithms like merge sort.

Key concepts we'll cover:

- Linked list structure and traversal.
- Pointer manipulation to build a new list.
- Iterative vs. recursive approaches to merging.

## ELI5: Merging Two Sorted Lists Explained Like You're Five

Imagine you have two lines of kids, each standing in order from shortest to
tallest. One line is `list1` (say, kids with heights 1, 2, 4), and the other is
`list2` (kids with heights 1, 3, 4). Your job is to make one big line where
everyone is still in order from shortest to tallest.

Here's how you do it:

1. Look at the first kid in each line.
2. Pick the shorter one to stand at the front of the new line.
3. Move forward in the line you picked from.
4. Repeat until you've picked all the kids from both lines.

For example:

- Start: Compare 1 (from `list1`) and 1 (from `list2`). Pick one of them (say,
  from `list1`). New line: [1].
- Next: Compare 2 (from `list1`) and 1 (from `list2`). Pick 1 from `list2`. New
  line: [1,1].
- Next: Compare 2 (from `list1`) and 3 (from `list2`). Pick 2. New line:
  [1,1,2].
- Keep going until the new line is [1,1,2,3,4,4].

In programming, each "kid" is a node in a linked list with a value (height) and
a pointer to the next node (the next kid). We use pointers to keep track of
where we are in each list and to build the new merged line.

## TypeScript Solution: Merging Two Sorted Lists

Let's implement this in TypeScript. First, we need to define the structure of a
linked list node using a class or interface. Then, we'll write a function to
merge the two lists iteratively (a recursive solution is also possible, but
iterative is more space-efficient).

### Step 1: Define the Linked List Node

```typescript
// Definition for singly-linked list node
class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}
```

This `ListNode` class represents a node in the linked list with a value (`val`)
and a pointer to the next node (`next`), which can be `null` if it's the last
node. TypeScript's type system ensures that `next` is either a `ListNode` or
`null`, adding type safety.

### Step 2: Iterative Solution to Merge Two Lists

We'll use an iterative approach with a dummy node to simplify the merging
process. The dummy node acts as a placeholder for the head of the merged list,
making it easier to handle edge cases (like when one list is empty).

```typescript
function mergeTwoLists(
  list1: ListNode | null,
  list2: ListNode | null,
): ListNode | null {
  // Create a dummy node to serve as the head of the merged list
  const dummy = new ListNode(0);
  // Use a current pointer to build the merged list
  let current = dummy;

  // Traverse both lists while neither is empty
  while (list1 !== null && list2 !== null) {
    // Compare values of the current nodes from list1 and list2
    if (list1.val <= list2.val) {
      // If list1's value is smaller or equal, add it to the merged list
      current.next = list1;
      // Move to the next node in list1
      list1 = list1.next;
    } else {
      // If list2's value is smaller, add it to the merged list
      current.next = list2;
      // Move to the next node in list2
      list2 = list2.next;
    }
    // Move the current pointer forward
    current = current.next;
  }

  // If list1 still has nodes, append them to the merged list
  if (list1 !== null) {
    current.next = list1;
  }
  // If list2 still has nodes, append them to the merged list
  if (list2 !== null) {
    current.next = list2;
  }

  // Return the head of the merged list (skip the dummy node)
  return dummy.next;
}
```

### Explanation of the Code

- **Dummy Node**: We start with a dummy node to avoid special handling for the
  head of the merged list. Its `next` pointer will point to the actual start of
  the merged list.
- **Comparison and Merging**: We compare the values of the current nodes from
  `list1` and `list2`. The smaller (or equal) value is linked to the merged
  list, and we advance the pointer in the corresponding input list.
- **Remaining Nodes**: Once one list is exhausted, we append the remaining nodes
  from the other list directly to the merged list.
- **Return**: Finally, we return `dummy.next`, which is the head of the merged
  list.

### Time and Space Complexity

- **Time Complexity**: $O(n + m)$, where $n$ and $m$ are the lengths of `list1`
  and `list2`, respectively. We traverse each list once.
- **Space Complexity**: $O(1)$, as we only use a few pointers (dummy and
  current) regardless of input size. We're reusing the existing nodes without
  creating new ones.

### Testing the Solution

Let's test this with the example from earlier: `list1 = [1,2,4]` and
`list2 = [1,3,4]`.

```typescript
// Helper function to create a linked list from an array (for testing)
function arrayToList(arr: number[]): ListNode | null {
  if (arr.length === 0) return null;
  const dummy = new ListNode(0);
  let current = dummy;
  for (const val of arr) {
    current.next = new ListNode(val);
    current = current.next;
  }
  return dummy.next;
}

// Helper function to convert linked list back to array (for output)
function listToArray(list: ListNode | null): number[] {
  const result: number[] = [];
  let current = list;
  while (current !== null) {
    result.push(current.val);
    current = current.next;
  }
  return result;
}

// Test the merge function
const list1 = arrayToList([1, 2, 4]);
const list2 = arrayToList([1, 3, 4]);
const merged = mergeTwoLists(list1, list2);
console.log(listToArray(merged)); // Output: [1, 1, 2, 3, 4, 4]
```

This test confirms our solution works as expected, producing a sorted merged
list.

## Conclusion: Key Takeaways

In this blog post, we tackled the LeetCode problem "Merge Two Sorted Lists," a
common interview question that tests linked list manipulation and pointer logic.
Here's what we learned:

- **Linked Lists Basics**: We defined a `ListNode` class in TypeScript to
  represent nodes in a singly linked list, leveraging type safety for `next`
  pointers.
- **Merging Logic**: We used an iterative approach with a dummy node to merge
  two sorted lists, comparing values and linking nodes in ascending order.
- **Efficiency**: Our solution runs in $O(n + m)$ time with $O(1)$ space, making
  it optimal for interviews where performance matters.
- **Practical Testing**: We included helper functions to test the solution,
  ensuring it handles real input correctly.

This problem is a stepping stone to more complex linked list challenges and
merging scenarios, often seen in Amazon interviews. By mastering this, you're
building a solid foundation for data structure questions. In the next post,
we'll explore another fundamental topic to further sharpen your skills. Stay
tuned, and happy coding!
